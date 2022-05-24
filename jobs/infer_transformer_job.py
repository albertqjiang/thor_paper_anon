import json
import time

import jax
import numpy as np
import optax
import pandas as pd
from smart_open import open

from jobs.core import Job
from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleus_sample
from mesh_transformer.transformer_shard import CausalTransformer
from mesh_transformer.util import clip_by_global_norm
from mesh_transformer_utils.inference import generate_target
from data.tfrecord_loader import TFRecordNewInputs
from mesh_transformer_utils.tokenization import TokenizerWrapper


def print_topk_predictions(decoded_target, output,
                           tokenizer, EOS_ID, k=4):
    all_preds = []
    for idx, o in enumerate(output):
        eos_positions = np.where(o == EOS_ID)[0]
        if len(eos_positions) == 0:
            eos_positions = [len(o)]

        eos_id = eos_positions[0]
        out = o[:eos_id]

        decoded_out = tokenizer.decode(out)
        acc = (decoded_out == decoded_target)
        all_preds.append((acc, decoded_out))
    topk_preds = sorted(all_preds, key=lambda x: x[0],
                        reverse=True)[:k]
    for i, (acc, o) in enumerate(topk_preds):
        print(f'Step {i}: "{o}", correct: {acc}')

    return topk_preds[0][0]


class InferenceTransformerJob(Job):

    def __init__(self, val_set, checkpoint_dir, batch_size=None):
        self.val_set = val_set
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size

    def load_params(self):
        params = json.load(open(f"{self.checkpoint_dir}/model_config.json"))

        opt = optax.chain(
            optax.scale(1 / 1),
            clip_by_global_norm(1),
            optax.scale_by_adam(),
            optax.additive_weight_decay(0),
            optax.scale(-1),
            optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
        )
        params["optimizer"] = opt  # dummy optimizer, not needed for inference
        # but required by the codebase

        params["sampler"] = nucleus_sample
        return params

    def load_network(self, params, devices, mesh_shape):
        network = CausalTransformer(params)

        start = time.time()
        with open(f"{self.checkpoint_dir}/meta.json", "r") as f:
            meta = json.load(f)

        ckpt_step = meta["checkpoints"][-1]
        print(f"using checkpoint {ckpt_step}")
        network.state = read_ckpt(network.state,
                                  f"{self.checkpoint_dir}/step_{ckpt_step}/",
                                  devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state,
                                          np.zeros(local_shards))
        return network

    @staticmethod
    def split_sequence_to_input_target_examples(seq, SEP_ID, EOS_ID):
        inp = seq[0]
        sep_pos = np.where(inp == SEP_ID)[0]
        eos_pos = np.where(inp == EOS_ID)[0]
        eos_pos = np.concatenate([np.array([-1]), eos_pos])

        for prev_end, input_end, target_end in zip(eos_pos[:-1], sep_pos,
                                                   eos_pos[1:]):
            yield inp[prev_end + 1:input_end + 1], inp[input_end + 1:target_end]

    def execute(self):
        params = self.load_params()

        per_replica_batch = params["per_replica_batch"]
        cores_per_replica = params["cores_per_replica"]
        assert cores_per_replica <= 8

        seq = params["seq"]

        start = time.time()
        print(f"jax devices: {jax.device_count()}")
        print(f"jax runtime initialized in {time.time() - start:.06}s")

        mesh_shape = (
            jax.device_count() // cores_per_replica, cores_per_replica)
        devices = np.array(jax.devices()).reshape(mesh_shape)

        total_batch = self.batch_size or per_replica_batch * \
                      jax.device_count() // cores_per_replica

        with jax.experimental.maps.Mesh(devices, ('dp', 'mp')):
            network = self.load_network(params, devices, mesh_shape)

            tokenizer = TokenizerWrapper.from_file_or_gpt(
                params.get('bpe_path'))

            val_set = TFRecordNewInputs(
                f'assets/data/{list(self.val_set.items())[0][1]}',
                batch_size=(1,),
                sample_size=seq
            )

            val_generator = val_set.sample_once()
            examples_correct, input_lens, target_lens, decoded_inp_lens =\
                [], [], [], []

            start = time.time()
            for seq in val_generator:
                for proof_state_sep, target in \
                        self.split_sequence_to_input_target_examples(
                            seq, tokenizer.sep_token_id,
                            tokenizer.eos_token_id):
                    if proof_state_sep.shape[0] > 2048:
                        continue

                    input_lens.append(proof_state_sep.shape[0])
                    target_lens.append(target.shape[0])

                    decoded_input = tokenizer.decode(proof_state_sep)
                    print(f'Input: "{decoded_input}"')
                    decoded_inp_lens.append(len(decoded_input))

                    decoded_target = tokenizer.decode(target)
                    print(f'Target: "{[decoded_target]}",'
                          f' target len: {len(target)},'
                          f' decoded len: {len(decoded_target)}')

                    output = generate_target(network, proof_state_sep,
                                             total_batch,
                                             gen_length=64)

                    correct = print_topk_predictions(decoded_target, output,
                                                     tokenizer,
                                                     tokenizer.eos_token_id,
                                                     k=1)
                    examples_correct.append(correct)

            print(f'Avg step accuracy:'
                  f' {100 * sum(examples_correct) / len(examples_correct)} %')
            print(f"Total evaluation time: {time.time() - start:.06}s")

            def get_stats(arr):
                return pd.DataFrame(arr).describe()
            print(f'Proof state input lengths: {get_stats(decoded_inp_lens)}\n'
                  f'Proof state stats: {get_stats(input_lens)}\n'
                  f'Proof step stats: {get_stats(target_lens)}')
