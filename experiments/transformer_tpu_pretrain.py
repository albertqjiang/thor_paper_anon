from datetime import datetime

from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/30m_non_emb_50k_pretrain.json',
    # 'TrainTransformerJob.model_config_path': 'assets/transformer_configs/200m_non_emb_50k_pretrain.json',
    # 'TrainTransformerJob.model_config_path': 'assets/transformer_configs/800m_non_emb_50k_pretrain.json',
    'TrainTransformerJob.save_model_bucket': 'atp-checkpoints-scratch',
    'TrainTransformerJob.save_model_dir': f'job_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',

    'TrainTransformerJob.train_set': ['assets/pretrain_data/github_train.index',
                                      'assets/pretrain_data/arxiv_train.index'],
    'TrainTransformerJob.train_inputs_cls': '@train/TFRecordMixtureInputs',
    'train/TFRecordMixtureInputs.seq2seq': False,

    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_state_only',
         'index_fname': 'assets/data/pisa_state_only_val_50k.index',
         'seq2seq': True},
        {'dataset_name': 'github',
         'index_fname': 'assets/pretrain_data/github_val.index',
         'seq2seq': False},
        {'dataset_name': 'arxiv',
         'index_fname': 'assets/pretrain_data/arxiv_val.index',
         'seq2seq': False},
    ],

    'TFRecordLoader.shuffle': True,
    'use_neptune': True
}

params_grid = {'idx': [0]}

experiments_list = create_experiments_helper(
    experiment_name='transformer-local-job',
    project_name='pmtest/labelleisabelle',
    script='python3 -m runner --mrunner --config_file=configs/empty.gin',
    python_path='',
    tags=['pretrain', 'train'],
    base_config=base_config, params_grid=params_grid
)
