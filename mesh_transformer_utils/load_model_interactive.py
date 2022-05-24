"""This module implements an easy way of creating interactive CasualTransformer
instance. Usage:

model = InteractiveTransformer(<checkpoint_path>, <batch_size>)
model.start()
output = model.predict(<input_str>)
model.close()
"""

import json
import time

import jax
import numpy as np
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleus_sample, typical_sample
from mesh_transformer.transformer_shard import CausalTransformer
from mesh_transformer.util import clip_by_global_norm
from mesh_transformer_utils.inference import generate_target
from mesh_transformer_utils.tokenization import TokenizerWrapper

from smart_open import open


def load_model_params(checkpoint_dir, params_override=None):
    with open(f"{checkpoint_dir}/model_config.json", "r") as f:
        params = json.load(f)

    opt = optax.chain(
        optax.scale(1 / 1),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )
    params["optimizer"] = opt  # dummy optimizer, not needed for inference
    if params_override is not None:
        for key, value in params_override.items():
            params[key] = value
    return params


def load_network(checkpoint_dir, params, devices, mesh_shape, step_to_load=None):
    network = CausalTransformer(params)

    start = time.time()
    if step_to_load is None:
        with open(f"{checkpoint_dir}/meta.json", "r") as f:
            meta = json.load(f)
        ckpt_step = meta["checkpoints"][-1]
    else:
        assert isinstance(step_to_load, int)
        ckpt_step = step_to_load
    print(f"using checkpoint {ckpt_step}")
    network.state = read_ckpt(network.state,
                              f"{checkpoint_dir}/step_{ckpt_step}/",
                              devices.shape[1])
    print(f"network loaded in {time.time() - start:.06}s")

    local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
    del network.state["opt_state"]
    network.state = network.move_xmap(network.state,
                                      np.zeros(local_shards))
    return network


class InteractiveMode:
    def __init__(self, checkpoint_dir, batch_size, step_to_load=None, sampler="nucleus"):
        if sampler == "nucleus":
            sampler = nucleus_sample
        elif sampler == "typical":
            sampler = typical_sample
        else:
            raise AttributeError("sampler parameter can take only one of the values [nucleus, typical]")

        params = load_model_params(checkpoint_dir, {"sampler": sampler})

        per_replica_batch = params["per_replica_batch"]
        cores_per_replica = params["cores_per_replica"]
        assert cores_per_replica <= 8

        start = time.time()
        print(f"jax devices: {jax.device_count()}")
        print(f"jax runtime initialized in {time.time() - start:.06}s")

        mesh_shape = (
            jax.device_count() // cores_per_replica, cores_per_replica)
        self.devices = np.array(jax.devices()).reshape(mesh_shape)

        self.total_batch = batch_size or per_replica_batch * \
                      jax.device_count() // cores_per_replica

        with jax.experimental.maps.Mesh(self.devices, ('dp', 'mp')):
            self.network = load_network(checkpoint_dir, params, self.devices, mesh_shape, step_to_load)

            self.tokenizer = TokenizerWrapper.from_file_or_gpt(
                params.get('bpe_path'))

    def predict(self, input_seq, gen_length, sampler_options):
        with jax.experimental.maps.Mesh(self.devices, ('dp', 'mp')):
            input_tokens = self.tokenizer.encode(input_seq)

            output = generate_target(self.network, input_tokens,
                                     self.total_batch,
                                     gen_length=gen_length,
                                     sampler_options=sampler_options,
                                     return_logits=True)

            res, logits = output
            log_softmax = jax.nn.log_softmax(np.squeeze(logits), -1)
            log_probs = np.take_along_axis(log_softmax, np.expand_dims(res, -1), axis=2)
            return [([self.tokenizer.decode([element]) for element in sample], np.squeeze(log_prob).tolist())
                    for sample, log_prob in zip(res, log_probs)]


def extract_output_and_logprob(outputs, eos_token, remove_repetitions, return_logits=False):
    str_list, acc_log_prob_list = [], []
    for token_list, log_prob_list in outputs:
        acc_log_prob = 0
        found = False
        for i, token in enumerate(token_list):
            acc_log_prob += log_prob_list[i]
            if token == eos_token:
                str_list.append("".join(token_list[:i]))
                found = True
                break
        if not found:
            str_list.append("".join(token_list))
        acc_log_prob_list.append(acc_log_prob)

    if remove_repetitions:
        dup_free_str_dict = {}
        for str_, log_prob in zip(str_list, acc_log_prob_list):
            if str_ not in dup_free_str_dict or dup_free_str_dict[str_] > log_prob:
                dup_free_str_dict[str_] = log_prob
        return [(key, value) for key, value in dup_free_str_dict.items()]
    return [(e1, e2) for e1, e2 in zip(str_list, acc_log_prob_list)]


def extract_output(output, eos_token, remove_repetitions):
    def split_on_eos(text):
        str_list = []
        for d in text.split(eos_token):
            if len(d.strip()) > 2:
                str_list.append(d[2:])
        return str_list
    out = [split_on_eos(x)[0] for x in output]
    if remove_repetitions:
        return list(set(out))
    else:
        return out


class InteractiveTransformer:
    def __init__(
        self,
        checkpoint_dir,
        batch_size,
        # input_init_token, input_end_token,
        output_end_token,
        return_logits=False,
        step_to_load=None,
        sampler="nucleus",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        # self.input_init_token = input_init_token
        # self.input_end_token = input_end_token
        self.output_end_token = output_end_token
        self.sampler = sampler
        self.return_logits = return_logits
        self.step_to_load = step_to_load
        self._model = None

    def start(self):
        self._model = InteractiveMode(
            self.checkpoint_dir,
            self.batch_size,
            step_to_load=self.step_to_load,
            sampler=self.sampler,
        )

    def predict(
        self,
        input_str,
        gen_length,
        sampler_options,
        input_max_len=1024,
        remove_repetitions=True,
    ):
        assert self._model is not None, "You must start model before predicting"
        if len(input_str) > input_max_len:
            input_str = input_str[:input_max_len]
        outputs = self._model.predict(
            # wrap_input(input_str, self.input_init_token, self.input_end_token),
            input_str,
            gen_length,
            sampler_options,
        )
        return extract_output_and_logprob(
            outputs,
            self.output_end_token,
            remove_repetitions,
            return_logits=self.return_logits,
        )

    def close(self):
        self._model.close()
        self._model = None
