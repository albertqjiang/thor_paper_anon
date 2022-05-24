import os
import json
import math
import jax

from jobs.train_transformer_job import TrainTransformerJob

from smart_open import open


def train_one_epoch(
    tf_records_path,
    base_config_path,
    model_config_path,
    base_name,
    iteration_k: int,
    tune_model_path=None,
):
    """
    Trains one epoch of the model.
    """
    base_bucket_path = f"gs://n2formal-public-data-europe/expert_iteration/{base_name}"
    # Get info about the tfrecords
    tf_records_name = tf_records_path.split("/")[-1]
    # Upload the tf records to the bucket, and save the path as an index file
    tf_records_bucket_location = (
        f"{base_bucket_path}/iteration_{iteration_k}/data/{tf_records_name}"
    )
    open(tf_records_bucket_location, "wb").write(open(tf_records_path, "rb").read())
    train_set = (
        f"assets/data/expert_iteration_{base_name}_iteration_{iteration_k}_train.index"
    )
    open(train_set, "w").write(tf_records_bucket_location)
    val_sets = [
        {
            "dataset_name": "first_step_eval",
            "index_fname": "assets/data/pisa_last_1_val_50k.index",
            "seq2seq": True,
        }
    ]

    # How many steps should I train?
    number_of_sequences = int(tf_records_name.strip(".tfrecords").strip("train_"))
    base_config = json.load(open(base_config_path))
    sequences_per_step = (
        base_config["tpu_size"] * base_config["gradient_accumulation_steps"]
    )
    number_of_steps_in_one_epoch = math.ceil(number_of_sequences / sequences_per_step)

    # Dump the right config
    base_config["warmup_steps"] = 10
    base_config["anneal_steps"] = number_of_steps_in_one_epoch - 10
    base_config["total_steps"] = number_of_steps_in_one_epoch
    base_config["lr"] = 1e-4
    base_config["end_lr"] = 1e-4
    base_config["tpu_size"] = jax.device_count()
    json.dump(base_config, open(model_config_path, "w"))

    # Set neptune API token
    os.system(
        'export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cH'
        'M6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzMTdhZmZhZS1kZjk1LTRhYjgtYTBhYS1hNThkZDVlYTJmNDEifQ=="'
    )

    run_locally = False
    save_model_bucket = "n2formal-public-data-europe"
    save_model_dir = f"expert_iteration/{base_name}/iteration_{iteration_k}/models"

    if tune_model_path:
        pass
    else:
        # If in iteration > 1, find the last model and load it
        meta_path = f"{base_bucket_path}/iteration_{iteration_k-1}/models/meta.json"
        meta = json.load(open(meta_path))
        last_checkpoint = meta["checkpoints"][-1]
        tune_model_path = f"{base_bucket_path}/iteration_{iteration_k-1}/models/step_{last_checkpoint}/"

    job = TrainTransformerJob(
        model_config_path=model_config_path,
        train_set=train_set,
        val_sets=val_sets,
        run_locally=run_locally,
        save_model_bucket=save_model_bucket,
        save_model_dir=save_model_dir,
        tune_model_path=tune_model_path,
        keep_n=1,
        delete_old=True,
    )
    job.execute()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Runs the expert iteration agent.")
    parser.add_argument("--tf_records_path", type=str, required=True)
    parser.add_argument("--base_config_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--base_name", type=str, required=True)
    parser.add_argument("--iteration_k", type=int, required=True)
    parser.add_argument("--tune_model_path", type=str, required=False, default=None)
    args = parser.parse_args()
    train_one_epoch(
        tf_records_path=args.tf_records_path,
        base_config_path=args.base_config_path,
        model_config_path=args.model_config_path,
        base_name=args.base_name,
        iteration_k=args.iteration_k,
        tune_model_path=args.tune_model_path,
    )
