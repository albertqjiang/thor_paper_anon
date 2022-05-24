import json
import logging
import time

import gin
import jax
import numpy as np
import optax
from smart_open import open
from tqdm import tqdm

import metric_logging
from data.tfrecord_loader import TFRecordInputs
from jobs.core import Job
from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.transformer_shard import CausalTransformer
from mesh_transformer.util import clip_by_global_norm, additive_weight_decay
from mesh_transformer_utils.device_train import train_step, eval_step, save
from mesh_transformer_utils.tokenization import TokenizerWrapper
from utils.general_utils import readable_num

logger = logging.getLogger(__name__)


class ValidationDataset:
    def __init__(self, dataset_name, dataset):
        self.name = dataset_name
        self.ds = dataset


class TrainTransformerJob(Job):
    def __init__(self, model_config_path, train_set, val_sets, run_locally,
                 train_inputs_cls=TFRecordInputs,
                 val_inputs_cls=TFRecordInputs,
                 save_model_bucket=None,
                 save_model_dir=None, tune_model_path=None,
                 permanent_checkpoints_at=(),
                 keep_n=200,
                 delete_old=False):
        self.model_config = self.load_model_config(model_config_path)
        self.model_config_original = self.model_config.copy()

        self.train_set = train_set
        self.train_inputs_cls = train_inputs_cls

        self.val_sets = val_sets
        self.val_inputs_cls = val_inputs_cls

        self.run_locally = run_locally
        self.save_model_bucket = save_model_bucket
        self.save_model_dir = save_model_dir
        self.tune_model_path = tune_model_path
        self.permanent_checkpoints_at = permanent_checkpoints_at

        self.tokenizer = TokenizerWrapper.from_file_or_gpt(
            self.model_config.get('bpe_path'))
        self.keep_n = keep_n
        self.delete_old = delete_old

    def execute(self):
        total_time_start = time.time()

        metric_logging.log_property('model_config', self.model_config)

        self.model_config["optimizer"], self.model_config[
            "scheduler"] = self.get_optimizer_and_scheduler()

        # Extract parameters
        val_batches = self.model_config["val_batches"]
        val_every = self.model_config["val_every"]
        ckpt_every = self.model_config["ckpt_every"]
        total_steps = self.model_config["total_steps"]

        gradient_accumulation_steps = self.model_config[
            "gradient_accumulation_steps"]

        cores_per_replica = self.model_config["cores_per_replica"]
        per_replica_batch = self.model_config["per_replica_batch"]

        devices, mesh_shape, tpu_size = self.init_tpu_devices()

        # pick initial ckpt - based on tuning vs train from scratch
        initial_ckpt_state_path, train_loader, fine_tuning, step = \
            self.get_checkpoint_metadata()

        # set up datasets
        train_dataset, val_datasets = self.load_datasets(tpu_size)

        # tok/sec metrics
        sequences_per_step = gradient_accumulation_steps * (
                per_replica_batch * tpu_size // cores_per_replica)
        tokens_per_step = self.model_config['seq'] * sequences_per_step

        # load + run
        with jax.experimental.maps.Mesh(devices, ('dp', 'mp')):
            logging.info("initializing network")
            network = CausalTransformer(self.model_config)

            if initial_ckpt_state_path:
                self.load_ckpt(network, initial_ckpt_state_path, fine_tuning,
                               devices)

            starting_step = self.compile_train_and_eval_fns(network,
                                                            train_dataset,
                                                            val_datasets[0],
                                                            step)

            train_losses = []
            for step in range(starting_step, total_steps+1):
                if step == starting_step or step % val_every == 0:
                    for val_dataset in val_datasets:
                        val_loss = self.evaluate(network, val_dataset, step,
                                                 val_batches,
                                                 print_examples_for_debug=(
                                                         step == starting_step))

                if not self.run_locally:
                    self.save_checkpoint_if_needed(network, step, total_steps,
                                                   ckpt_every, val_loss,
                                                   cores_per_replica)

                start = time.time()
                train_batch = train_dataset.get_samples()
                if step == starting_step:
                    self.log_dataset_for_debug(train_batch, f'train_{step}')

                loss, last_loss, grad_norm, grad_norm_micro, weights_l2 = \
                    train_step(
                        network, train_batch
                    )
                step += 1

                train_losses.append(loss)

                if step == starting_step + 1 or step % 50 == 0:
                    avg_loss = np.array(train_losses).mean()
                    train_losses = []
                    self.log_training_metrics(network,
                                              avg_loss, last_loss, grad_norm,
                                              grad_norm_micro, weights_l2,
                                              sequences_per_step,
                                              tokens_per_step,
                                              step,
                                              step_time=time.time() - start)

        return readable_num(time.time() - total_time_start)

    def save_checkpoint_if_needed(self, network, step, total_steps, ckpt_every,
                                  val_loss, cores_per_replica):
        should_save_ckpt = (step % ckpt_every == 0 or step == total_steps)
        # TODO: do proper ckpt saving as described in the benchmarking doc

        is_permanent_ckpt = step in self.permanent_checkpoints_at

        if should_save_ckpt or is_permanent_ckpt:
            logging.info(f"saving a checkpoint for step {step}")
            save(network, self.model_config_original, step,
                 self.save_model_bucket,
                 self.save_model_dir,
                 mp=cores_per_replica,
                 # aux={"train_loader": train_dataset.get_state()}, TODO
                 keep_n=self.keep_n,
                 delete_old=self.delete_old,
                 permanent=is_permanent_ckpt
                 )

    def set_up_special_token_ids(self):
        sep_id, eos_id = \
            self.tokenizer.sep_token_id, self.tokenizer.eos_token_id
        with gin.unlock_config():
            gin.bind_parameter('TFRecordInputs.sep_id',
                               sep_id)
            gin.bind_parameter('TFRecordInputs.eos_id',
                               eos_id)

    def evaluate(self, network, val_dataset, step, val_batches,
                 print_examples_for_debug=False):
        val_loss = []
        val_accuracy = []
        val_seq_accuracy = np.array([0.0, 0.0])

        for i in tqdm(range(val_batches),
                      desc=f"validation for step {step}, ds {val_dataset.name}",
                      total=val_batches):
            val_batch = val_dataset.ds.get_samples()
            if i == 0 and print_examples_for_debug:
                self.log_dataset_for_debug(val_batch,
                                           f'val_{step}_{val_dataset.name}')

            val_l, val_acc, val_seq_acc = eval_step(network, val_batch)
            val_loss.append(val_l)
            val_accuracy.append(val_acc)
            val_seq_accuracy += val_seq_acc
        val_dataset.ds.reset()

        val_loss = np.array(val_loss).mean()
        val_accuracy = np.array(val_accuracy).mean()
        val_seq_accuracy = val_seq_accuracy[0] / val_seq_accuracy[1]
        logging.info(
            f"validation loss for step {step}: {val_loss},"
            f" validation accuracy: {val_accuracy},"
            f" validation seq accuracy: {val_seq_accuracy}")

        metric_logging.log_scalar(
            f'eval_{val_dataset.name}/loss', step, float(val_loss))
        metric_logging.log_scalar(f'eval_{val_dataset.name}/tok_accuracy', step,
                                  float(val_accuracy))
        metric_logging.log_scalar(f'eval_{val_dataset.name}/seq_accuracy', step,
                                  float(val_seq_accuracy))
        return val_loss

    def load_datasets(self, tpu_size):
        logging.info('Loading datasets')

        cores_per_replica = self.model_config["cores_per_replica"]
        per_replica_batch = self.model_config["per_replica_batch"]
        gradient_accumulation_steps = self.model_config[
            "gradient_accumulation_steps"]

        train_dataset = self.train_inputs_cls(
            self.train_set,
            batch_size=(
                gradient_accumulation_steps,
                per_replica_batch *
                tpu_size // cores_per_replica
            ),
            tokenizer=self.tokenizer,
        )

        global_val_batch = per_replica_batch * tpu_size // cores_per_replica

        val_datasets = [ValidationDataset(ds_config['dataset_name'],
                                          self.val_inputs_cls(
                                              ds_config['index_fname'],
                                              batch_size=(global_val_batch,),
                                              tokenizer=self.tokenizer,
                                              seq2seq=ds_config['seq2seq'],
                                          )) for ds_config in self.val_sets]

        return train_dataset, val_datasets

    @staticmethod
    def load_ckpt(network, initial_ckpt_state_path, fine_tuning, devices):
        logging.info("loading network")
        init_sched_state = network.state["opt_state"][-1]

        start = time.time()
        network.state = read_ckpt(network.state,
                                  initial_ckpt_state_path,
                                  devices.shape[1],
                                  load_opt=not fine_tuning)  # TODO: check

        if fine_tuning:
            # overwrite the loaded scheduler step with zeros
            # this makes fine-tuning use the lr schedule in
            network.state["opt_state"][-1] = init_sched_state

        logging.info(f"network loaded in {time.time() - start:.06}s")

    @staticmethod
    def compile_train_and_eval_fns(network, train_dataset, val_dataset,
                                   step):
        logging.info('compiling train fn')
        start = time.time()
        train_step(
            network, train_dataset.get_samples()
        )
        step += 1
        logging.info(f"Train fn compiled in {time.time() - start:.06}s")

        logging.info('compiling eval fn')
        start = time.time()
        eval_step(network, val_dataset.ds.get_samples())
        val_dataset.ds.reset()
        logging.info(f"Eval fn compiled in {time.time() - start:.06}s")
        return step

    @staticmethod
    def load_model_config(model_config_path):
        params = json.load(open(model_config_path))

        assert params["cores_per_replica"] <= 8
        assert params["pe"] in ["fixed", "rotary", "t5"]

        params.setdefault("gradient_accumulation_steps", 1)
        params.setdefault("noise_scale_alpha", 0.01)

        return params

    def get_optimizer_and_scheduler(self):
        gradient_accumulation_steps = self.model_config[
            "gradient_accumulation_steps"]
        warmup_steps = self.model_config["warmup_steps"]
        anneal_steps = self.model_config["anneal_steps"]
        lr = self.model_config["lr"]
        end_lr = self.model_config["end_lr"]
        weight_decay = self.model_config["weight_decay"]

        scheduler = util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr)

        opt = optax.chain(
            optax.scale(1 / gradient_accumulation_steps),
            clip_by_global_norm(1),
            optax.scale_by_adam(),
            additive_weight_decay(weight_decay),
            optax.scale(-1),
            optax.scale_by_schedule(scheduler)
        )

        return opt, scheduler

    def init_tpu_devices(self):
        start = time.time()
        tpu_size = jax.device_count()
        cores_per_replica = self.model_config["cores_per_replica"]

        if tpu_size < cores_per_replica:
            msg = f"each shard needs a separate device, but device count" \
                  f" ({tpu_size}) < shard count ({cores_per_replica})"
            raise ValueError(msg)
        logging.info(f"jax devices: {tpu_size}")
        logging.info(f"jax runtime initialized in {time.time() - start:.06}s")

        mesh_shape = (tpu_size // cores_per_replica, cores_per_replica)
        devices = np.array(jax.devices()).reshape(mesh_shape)
        return devices, mesh_shape, tpu_size

    def get_checkpoint_metadata(self):
        initial_ckpt_state_path = None
        train_loader = None
        step = 0

        if self.tune_model_path:
            logging.info(
                '`--tune_model_path` passed: we are beginning a fine-tuning run'
            )
            fine_tuning = True
            initial_ckpt_state_path = self.tune_model_path
        else:
            logging.info(
                '`--tune_model_path` not passed: we are continuing'
                ' a fine-tuning run from a checkpoint'
                ' (or we are not fine-tuning)')
            fine_tuning = False
            initial_ckpt_model_dir = self.save_model_dir
            initial_ckpt_path = \
                f"gs://{self.save_model_bucket}/{initial_ckpt_model_dir}"
            meta_path = f"{initial_ckpt_path}/meta.json"

            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                ckpt_step = meta["checkpoints"][-1]
                initial_ckpt_state_path = \
                    f"{initial_ckpt_path}/step_{ckpt_step}/"
                logging.info(
                    f"state will be restored from checkpoint {ckpt_step}")

                step = ckpt_step
                train_loader = meta['aux'][str(ckpt_step)].get("train_loader",
                                                               None)
            except Exception as e:
                # no checkpoint, start at zero
                logging.info(
                    f"{e}. No checkpoint to load at {initial_ckpt_path}. "
                    f"Training from scratch.")

        return initial_ckpt_state_path, train_loader, fine_tuning, step

    def log_training_metrics(self, network, loss, last_loss, grad_norm,
                             grad_norm_micro, weights_l2,
                             sequences_per_step, tokens_per_step, step,
                             step_time):

        steps_per_sec = 1 / step_time
        tokens_per_sec = tokens_per_step * steps_per_sec
        sequences_processed = sequences_per_step * step
        tokens_processed = tokens_per_step * step
        gradient_accumulation_steps = self.model_config[
            "gradient_accumulation_steps"]

        # compute summary stats about the gradient

        # converts from grads-summed-over-microbatch
        # (what `CasualTransformer.train` computes)
        # to grads-averaged-over-microbatch (what we want)
        #
        # (when taking gradient steps, the same conversion
        # happens inside the optimizer
        #  via optax.scale(1 / gradient_accumulation_steps))
        grad_norm = grad_norm / gradient_accumulation_steps

        # compute G_noise and S_noise
        # from "An Empirical Model of Large-Batch Training" Appendix A.1
        # here, B_big = gradient_accumulation_steps,
        # and B_small = 1 for convenience
        gbsmall = grad_norm_micro ** 2
        gbbig = grad_norm ** 2
        G_noise = (gradient_accumulation_steps * gbbig - gbsmall) / (
                gradient_accumulation_steps - 1
        )
        S_noise = (gbsmall - gbbig) / (
                1 - 1 / gradient_accumulation_steps)

        noise_scale_stats = {
            "~noise/G_noise": G_noise,
            "~noise/S_noise": S_noise,
        }

        # heuristic to avoid reporting G_noise in very early training when
        # gradients are large (these take a long time to wash out of the
        # moving average that defines B_simple)
        use_step_in_noise_avgs = gbbig < 2

        G_noise_avg, S_noise_avg = None, None
        noise_scale_alpha = self.model_config[
            "noise_scale_alpha"]  # alpha parameter for
        # the exponential moving averages used to compute B_simple
        if use_step_in_noise_avgs:
            # compute moving averages of G_noise and S_noise, for B_simple
            if G_noise_avg is None:
                G_noise_avg = G_noise
            else:
                G_noise_avg = (1 - noise_scale_alpha) * G_noise_avg + \
                              noise_scale_alpha * G_noise

            if S_noise_avg is None:
                S_noise_avg = S_noise
            else:
                S_noise_avg = (1 - noise_scale_alpha) * S_noise_avg + \
                              noise_scale_alpha * S_noise

            B_simple = S_noise_avg / G_noise_avg

            noise_scale_stats.update(
                {
                    "~train/noise/G_noise_avg": G_noise_avg,
                    "~train/noise/S_noise_avg": S_noise_avg,
                    "~train/noise/B_simple": B_simple,
                }
            )

        wandb_stats = {
            "train/loss": loss,
            "~train/last_loss": last_loss,
            "~train/steps_per_sec": steps_per_sec,
            "~train/tokens_per_sec": tokens_per_sec,
            "train/grad_norm": grad_norm,
            "train/weights_l2": weights_l2,
            "train/learning_rate": float(self.model_config["scheduler"](
                network.state["opt_state"][-1].count[0].item())),
            "~sequences_processed": sequences_processed,
            "~tokens_processed": tokens_processed,
        }
        wandb_stats.update(noise_scale_stats)
        metric_logging.log_dict_as_scalars(step, wandb_stats)

    def log_dataset_for_debug(self, example, name):
        def flatten_batch_dims(x):
            return np.reshape(x, (-1, x.shape[-1]))
        batch = flatten_batch_dims(example["input"])
        mask = flatten_batch_dims(example["mask"])
        for i, (example, mask) in enumerate(zip(batch, mask)):
            metric_logging.log_text(f'{name}_{i}', str(example.tolist()))
            metric_logging.log_text(f'{name}_{i}_detokenized',
                                    self.tokenizer.decode(example))
            metric_logging.log_text(f'{name}_{i}_mask', str(mask.tolist()))
            metric_logging.log_text(f'{name}_{i}_mask_sum', str(mask.sum()))
