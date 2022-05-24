import argparse
import json
import time

import gin
import numpy as np

from mesh_transformer.checkpoint import write_ckpt
from smart_open import open
from google.cloud import storage


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="""
    To use, download the full checkpoint archive, extract and upload to a GCS bucket, and set that as --tune-model-path
    Modify the config file:
        - set `model_dir` to where the checkpoints should be written during training
        - set `train_set`, `val_set` to index files for your data
        - set `tpu_size` to 8 (if on a v3-8)
        - set `warmup_steps`, `anneal_steps`, `lr`, `end_lr` to the lr schedule for your finetuning run
        - the global step will reset to 0, keep that in mind when writing your lr schedule
        - set `name` to specify the name of the Weights & Biases run
        - set `wandb_project` to specify the Weights & Biases project to log to
    To prepare data in the expected data format:
        - use the script `create_finetune_tfrecords.py` in this repo to create data in the expected format
        - upload the .tfrecords files to GCS
        - save their GCS paths to a index file under `data/`, see existing files for examples
    """,
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument("--tune-model-path", type=str, default=None, help="Base model to finetune")
    parser.add_argument("--fresh-opt", default=False, action="store_true", help="Use a newly initialized optimizer, ignoring any optimizer state saved in the base checkpoint")

    args = parser.parse_args()
    return args


def save(network, config, step, bucket, path, mp, aux=None, keep_n=3, delete_old=True,
         permanent=False):
    assert path
    client = storage.Client()

    if aux is None:
        aux = {}

    try:
        with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
            meta = json.load(f)
    except:
        # create metadata file
        with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
            json.dump({
                "step": 0,
                "checkpoints": [],
                "permanent_checkpoints": [],
                "aux": {}
            }, f)

    # do sharded checkpoint writing
    start = time.time()
    res = []
    for shard_id in range(mp):
        write_ckpt(network.state, f"gs://{bucket}/{path}/step_{step}/", shard_id)

    print(f"Wrote checkpoint in {time.time() - start:.06}s")

    with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
        meta = json.load(f)

    meta["step"] = step
    if permanent:
        meta["permanent_checkpoints"].append(step)
    else:
        meta["checkpoints"].append(step)
    all_aux = meta.get("aux", {})

    while len(meta["checkpoints"]) > keep_n:
        ckpt_to_delete = meta["checkpoints"].pop(0)

        try:
            del all_aux[str(ckpt_to_delete)]
        except:
            print(f"failed to delete the aux state for {step}")

        if delete_old:
            print(f"deleting checkpoint {ckpt_to_delete}")
            for blob in client.list_blobs(bucket, prefix=f"{path}/step_{ckpt_to_delete}/"):
                # print(f"deleting {blob.name}")
                assert path in blob.name
                blob.delete()
        else:
            print(f"keeping checkpoint {ckpt_to_delete}")

    all_aux[step] = aux
    meta["aux"] = all_aux

    with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
        json.dump(meta, f)

    with open(f"gs://{bucket}/{path}/model_config.json", "w") as f:
        json.dump(config, f)


def train_step(network, data):
    input_batch = data["input"]
    mask_batch = data["mask"]

    inp = input_batch[:, :, :-1]
    tgt = input_batch[:, :, 1:]
    tgt_mask = mask_batch[:, :, 1:]

    inputs = {
        "obs": inp,
        "target": tgt,
        "mask": tgt_mask,
    }

    loss, last_loss, grad_norm, grad_norm_micro, weights_norm = \
        network.train(inputs)

    return (
        np.array(loss).mean(),
        np.array(last_loss).mean(),
        np.array(grad_norm).mean(),
        np.array(grad_norm_micro).mean(),
        np.array(weights_norm).mean(),
    )


def sequence_accuracy(seq2seq_mask, correct):
    correct = np.where(correct > 0, 1., 0.)  # Remove weight scaling
    split_indices = 1 + np.flatnonzero(seq2seq_mask[:-1] != seq2seq_mask[1:])

    mask_split = np.split(seq2seq_mask, split_indices)
    correct_split = np.split(correct, split_indices)

    def get_min_vals(list_of_arrays):
        return np.array(list(map(lambda x: x.min(), list_of_arrays)))

    mask_split_max, correct_split_max = map(get_min_vals,
                                            [mask_split, correct_split])
    n_sequences = mask_split_max[mask_split_max == 1].sum()
    correct_sequences = correct_split_max[mask_split_max == 1].sum()
    return np.array([correct_sequences, n_sequences])


def eval_step(network, data):
    input_batch = data["input"]
    mask_batch = data["mask"]

    inp = input_batch[:, :-1]
    tgt = input_batch[:, 1:]
    tgt_mask = mask_batch[:, 1:]

    inputs = {
        "obs": inp,
        "target": tgt,
        "mask": tgt_mask,
    }

    out = network.eval(inputs)
    loss = np.array(out["loss"])
    correct = np.array(out['correct'])

    # Pair: (correct, total)
    seq_accuracy = sum(sequence_accuracy(mask, correct_example) for
                       mask, correct_example in zip(tgt_mask, correct))

    return loss.mean(), correct.mean(), seq_accuracy
