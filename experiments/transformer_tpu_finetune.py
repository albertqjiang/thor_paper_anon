from datetime import datetime

from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    # 'TrainTransformerJob.model_config_path': 'assets/transformer_configs/30m_non_emb_50k.json',
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/200m_non_emb_50k.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_200m/step_100000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/metric_state_only_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_state_only_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_eval',
         'index_fname': 'assets/data/pisa_state_only_val_50k.index',
         'seq2seq': True},
        {'dataset_name': 'first_step_eval',
         'index_fname': 'assets/data/eval_first_steps_valid.index',
         'seq2seq': True}
    ],
    'use_neptune': True
}

config_800M = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    # 'TrainTransformerJob.model_config_path': 'assets/transformer_configs/30m_non_emb_50k.json',
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/800m_non_emb_50k.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_800m/step_200000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/800m_last_1_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_last_1_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_last_1_eval',
         'index_fname': 'assets/data/pisa_last_1_val_50k.index',
         'seq2seq': True},
        {'dataset_name': 'first_step_eval',
         'index_fname': 'assets/data/eval_first_steps_valid.index',
         'seq2seq': True}
    ],
    'use_neptune': True
}

config_800M_large_lr_and_bs = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/800m_non_emb_50k_large_lr_and_bs.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_800m/step_200000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/800m_last_1_large_lr_and_bs_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_last_1_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_last_1_eval',
         'index_fname': 'assets/data/pisa_last_1_val_50k.index',
         'seq2seq': True},
        {'dataset_name': 'first_step_eval',
         'index_fname': 'assets/data/eval_first_steps_valid.index',
         'seq2seq': True}
    ],
    'use_neptune': True
}

config_800M_even_larger_lr_and_bs_and_dropout = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/800m_non_emb_50k_even_larger_lr_and_bs.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_800m/step_200000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/800m_last_1_even_larger_lr_and_bs_and_dropout_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_last_1_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_last_1_eval',
         'index_fname': 'assets/data/pisa_last_1_val_50k.index',
         'seq2seq': True}
    ],
    'use_neptune': True
}

mix_config = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    # 'TrainTransformerJob.model_config_path': 'assets/transformer_configs/30m_non_emb_50k.json',
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/200m_non_emb_50k.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_200m/step_100000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    # 'TrainTransformerJob.save_model_dir': f'models/mix_last_1_and_trimmed_proof_and_state_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    # 'TrainTransformerJob.save_model_dir': f'models/mix_with_state_only_and_trimmed_proof_and_state_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.save_model_dir': f'models/mix_with_state_only_and_last_1_and_trimmed_proof_and_state_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    # 'TrainTransformerJob.train_set': 'assets/data/pisa_mix_last_1_and_trimmed_proof_and_state_train_50k.index',
    # 'TrainTransformerJob.train_set': 'assets/data/pisa_mix_with_state_only_and_trimmed_proof_and_state_train_50k.index',
    'TrainTransformerJob.train_set': 'assets/data/pisa_mix_with_state_only_and_last_1_and_trimmed_proof_and_state_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_current_mix_eval',
         # 'index_fname': 'assets/data/pisa_mix_last_1_and_trimmed_proof_and_state_val_50k.index',
         # 'index_fname': 'assets/data/pisa_mix_with_state_only_and_trimmed_proof_and_state_val_50k.index',
         'index_fname': 'assets/data/pisa_mix_with_state_only_and_last_1_and_trimmed_proof_and_state_val_50k.index',
         'seq2seq': True},
        {'dataset_name': 'pisa_eval',
         'index_fname': 'assets/data/pisa_state_only_val_50k.index',
         'seq2seq': True},
        {'dataset_name': 'first_step_eval',
         'index_fname': 'assets/data/eval_first_steps_valid.index',
         'seq2seq': True}
    ],
    'use_neptune': True
}

large_lr_and_bs_config ={
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    # 'TrainTransformerJob.model_config_path': 'assets/transformer_configs/30m_non_emb_50k.json',
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/200m_non_emb_50k_large_lr_and_bs.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_200m/step_100000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/large_lr_and_bs_metric_state_only_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_state_only_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_eval',
         'index_fname': 'assets/data/pisa_state_only_val_50k.index',
         'seq2seq': True},
        {'dataset_name': 'first_step_eval',
         'index_fname': 'assets/data/eval_first_steps_valid.index',
         'seq2seq': True}
    ],
    'use_neptune': True
}

large_lr_and_bs_last_1_learn_hammer_config ={
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/200m_non_emb_50k_large_lr_and_bs.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_200m/step_100000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/large_lr_and_bs_last_1_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_last_1_learn_hammer_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_eval',
         'index_fname': 'assets/data/pisa_last_1_learn_hammer_val_50k.index',
         'seq2seq': True},
        {'dataset_name': 'first_step_eval',
         'index_fname': 'assets/data/eval_first_steps_valid.index',
         'seq2seq': True}
    ],
    'use_neptune': True
}

even_larger_lr_and_bs_last_1_learn_hammer_800m_config = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/800m_non_emb_50k_even_larger_lr_and_bs.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_800m/step_200000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/800m_even_larger_lr_and_bs_last_1_learn_hammer_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_last_1_learn_hammer_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_eval',
         'index_fname': 'assets/data/pisa_last_1_learn_hammer_val_50k.index',
         'seq2seq': True},
        # {'dataset_name': 'first_step_eval',
        #  'index_fname': 'assets/data/eval_first_steps_valid.index',
        #  'seq2seq': True}
    ],
    'use_neptune': True
}

even_larger_lr_and_bs_state_only_learn_hammer_800m_config = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/800m_non_emb_50k_even_larger_lr_and_bs.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_800m/step_200000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/800m_even_larger_lr_and_bs_state_only_learn_hammer_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_state_only_learn_hammer_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_eval',
         'index_fname': 'assets/data/pisa_state_only_learn_hammer_val_50k.index',
         'seq2seq': True},
        # {'dataset_name': 'first_step_eval',
        #  'index_fname': 'assets/data/eval_first_steps_valid.index',
        #  'seq2seq': True}
    ],
    'use_neptune': True
}

large_lr_and_bs_trimmed_proof_and_state_config ={
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    # 'TrainTransformerJob.model_config_path': 'assets/transformer_configs/30m_non_emb_50k.json',
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/200m_non_emb_50k_large_lr_and_bs.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_200m/step_100000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/large_lr_and_bs_trimmed_proof_and_state_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_trimmed_proof_and_state_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_eval',
         'index_fname': 'assets/data/pisa_trimmed_proof_and_state_val_50k.index',
         'seq2seq': True},
    ],
    'use_neptune': True
}

proof_only_config = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/200m_non_emb_50k.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_200m/step_100000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/proof_only_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_proof_only_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_proof_only_eval',
         'index_fname': 'assets/data/pisa_proof_only_valid_50k.index',
         'seq2seq': True},
        {'dataset_name': 'first_step_proof_only_eval',
         'index_fname': 'assets/data/eval_first_step_proof_only_valid.index',
         'seq2seq': True}
    ],
    'use_neptune': True
}

expert_iteration_debug_config = {
    # run parameters:
    'run.job_class': '@jobs.TrainTransformerJob',
    'TrainTransformerJob.run_locally': False,
    'TrainTransformerJob.model_config_path': 'assets/transformer_configs/200m_non_emb_50k_large_lr_and_bs.json',
    'TrainTransformerJob.tune_model_path': 'gs://n2formal-public-data-europe/pretrained_checkpoints/pretrained_200m/step_100000/',
    'TrainTransformerJob.save_model_bucket': 'n2formal-public-data-europe',
    'TrainTransformerJob.save_model_dir': f'models/expert_iteration_debug_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}',
    'TrainTransformerJob.train_set': 'assets/data/pisa_last_1_learn_hammer_train_50k.index',
    'TrainTransformerJob.val_sets': [
        {'dataset_name': 'pisa_eval',
         'index_fname': 'assets/data/pisa_last_1_learn_hammer_val_50k.index',
         'seq2seq': True},
        {'dataset_name': 'first_step_eval',
         'index_fname': 'assets/data/eval_first_steps_valid.index',
         'seq2seq': True}
    ],
    'use_neptune': True
}

params_grid = {'idx': [0]}

experiments_list = create_experiments_helper(
    # experiment_name='transformer-finetune-800M-tiled-dropout-large-lr-and-bs',
    # experiment_name='transformer-finetune-800M-tiled-dropout-even-larger-lr-and-bs',
    # experiment_name='transformer-finetune-200M-tiled-dropout-large-lr-and-bs-learn-hammer',
    # experiment_name='transformer-finetune-800M-tiled-dropout-large-lr-and-bs-learn-hammer',
    experiment_name='transformer-finetune-800M-tiled-dropout-even-larger-lr-and-bs-state-only-learn-hammer',
    # experiment_name='expert_iteration_debug',
    project_name='pmtest/labelleisabelle',
    script='python3 -m runner --mrunner --config_file=configs/empty.gin',
    python_path='',
    tags=['train'],
    # base_config=proof_only_config,
    # base_config=config_800M,
    # base_config=config_800M_large_lr_and_bs,
    # base_config=expert_iteration_debug_config,
    # base_config=even_larger_lr_and_bs_last_1_learn_hammer_800m_config,
    # base_config=config_800M_even_larger_lr_and_bs_and_dropout,
    base_config=even_larger_lr_and_bs_state_only_learn_hammer_800m_config,
    params_grid=params_grid
)
