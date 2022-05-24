from datetime import datetime

from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # run parameters:
    'run.job_class': '@jobs.CreateDataJob',
    # 'CreateDataJob.raw_input_dir': '../data_samples',
    'CreateDataJob.raw_input_dir': None,
    # 'CreateDataJob.raw_input_dir': 'gs://n2formal-public-data-europe/seq2seq_all/with_state',
    # 'CreateDataJob.txt_data_dir': 'create_finetune_in',
    'CreateDataJob.txt_data_dir': 'gs://atp-data-scratch/mathematics_dataset-v1.0',
    'CreateDataJob.processed_output_dir': 'gs://atp-data-scratch/mathematics_tokenizer_50k',
    # 'CreateDataJob.processed_output_dir': 'gs://atp-data-scratch/data_tokenizer_6144',
    # 'CreateDataJob.tokenizer_path': 'assets/tokenizers/tokenizer-6144-tokens.json',
    'CreateDataJob.tokenizer_path': None,
    'CreateDataJob.seq2seq': False,
    'use_neptune': True
}

params_grid = {'idx': [0]}

experiments_list = create_experiments_helper(
    experiment_name='transformer-local-job',
    project_name='atp-debug',
    script='python3 -O -m runner --mrunner --config_file=configs/empty.gin',
    python_path='',
    tags=['dataset'],
    base_config=base_config, params_grid=params_grid
)
