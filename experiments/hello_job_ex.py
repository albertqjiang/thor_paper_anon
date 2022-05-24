from mrunner.helpers.specification_helper import create_experiments_helper

# This job is used for debugging purposes. .
base_config = {
    #run parameters:
    'run.job_class': '@jobs.HelloJob',
    'use_neptune': True
}

params_grid = {'learning_rate': [0]}

experiments_list = create_experiments_helper(
    experiment_name='hello-job',
    project_name='pmtest/labelleisabelle',
    script='python3 -m runner --mrunner --config_file=configs/empty.gin',
    python_path='',
    tags=['solving'],
    base_config=base_config, params_grid=params_grid
)
