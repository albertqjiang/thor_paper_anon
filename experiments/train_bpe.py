from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    #run parameters:
    'run.job_class': '@jobs.TrainBPEJob',
    'TrainBPEJob.vocab_size': 6144,
    'TrainBPEJob.special_tokens': ['<|endoftext|>', 'Cambridge'],
    'TrainBPEJob.dataset_path': 'gs://atp-checkpoints-scratch/state_only_txt/state_only_train.txt',
    'TrainBPEJob.save_dir': '.',
    'TrainBPEJob.save_bucket_dir': 'gs://atp-checkpoints-scratch/tokenizers',
    'use_neptune': False
}

params_grid = {'learning_rate': [0]}

experiments_list = create_experiments_helper(
    experiment_name='train_bpe',
    project_name='tomaszodrzygozdz/ATP',
    script='python3 -m runner --mrunner --config_file=configs/empty.gin',
    python_path='',
    tags=['solving'],
    base_config=base_config, params_grid=params_grid
)
