from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': True,
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/tokenizer_50k_local.json',

    # 'TrainTransformerJob.train_inputs_cls': '@ProofsWithContextInputs',
    # 'TrainTransformerJob.train_set': 'assets/data/episodic_train.index',
    # 'TrainTransformerJob.val_inputs_cls': '@ProofsWithContextInputs',
    # 'TrainTransformerJob.val_sets': [{
    #     'dataset_name': 'pisa_episodic',
    #     'index_fname': 'assets/data/episodic_val.index',
    #     'seq2seq': True,
    # }],
    # 'ProofsWithContextInputs.max_len': 513,

    'TrainTransformerJob.train_set': 'assets/data/pisa_state_only_train_50k.index',

    'TrainTransformerJob.val_sets': [{
        'dataset_name': 'pisa_episodic',
        'index_fname': 'assets/data/pisa_state_only_train_50k.index',
        'seq2seq': True,
    }],
    'TFRecordLoader.shuffle': True,
    'use_neptune': False
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
