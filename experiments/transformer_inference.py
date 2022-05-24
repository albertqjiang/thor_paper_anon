from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # run parameters:
    'run.job_class': '@jobs.InferenceTransformerJob',
    'InferenceTransformerJob.val_set': {'new_val': 'new_val_50k.index'},
    'InferenceTransformerJob.checkpoint_dir': 'gs://atp-checkpoints-scratch/job_2021_12_08-10_01_36_AM', # 30M
    # 'InferenceTransformerJob.checkpoint_dir': 'gs://atp-checkpoints-scratch/job_2021_12_09-05_48_24_PM', # 200M

    'InferenceTransformerJob.batch_size': 8,
}

params_grid = {'idx': [0]}

experiments_list = create_experiments_helper(
    experiment_name='transformer-local-job',
    project_name='atp-debug',
    script='python3 -m runner --mrunner --config_file=configs/empty.gin',
    python_path='',
    tags=['solving'],
    base_config=base_config, params_grid=params_grid
)
