from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # run parameters:
    'run.job_class': '@jobs.OutlinesGenerationJob',
    'OutlinesGenerationJob.afp_path': '/home/tomek/afp-2021-10-22',
    'OutlinesGenerationJob.src_path': '/home/tomek/Isabelle2021',
    'OutlinesGenerationJob.out_path': '/home/tomek/Research/atp_data/temp/all_outlines.json',

    'use_neptune': True
}

params_grid = {'idx': [0]}

experiments_list = create_experiments_helper(
    experiment_name='generate-outlines',
    project_name='pmtest/labelleisabelle',
    script='python3 -m runner --mrunner --config_file=configs/empty.gin',
    python_path='',
    tags=['solving'],
    base_config=base_config, params_grid=params_grid
)
