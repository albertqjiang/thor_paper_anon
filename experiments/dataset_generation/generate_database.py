from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # run parameters:
    'run.job_class': '@jobs.TheoremsDatabaseGenerationJob',

    'TheoremsDatabaseGenerationJob.afp_path': '/home/tomek/Research/atp_data/afp-2021-10-22',
    'TheoremsDatabaseGenerationJob.src_path': '/home/tomek/Isabelle2021',
    'TheoremsDatabaseGenerationJob.out_path': '/home/tomek/Research/atp_data/temp/all_thms_bb.json',

    'use_neptune': True
}

params_grid = {'idx': [0]}

experiments_list = create_experiments_helper(
    experiment_name='generate_database',
    project_name='pmtest/labelleisabelle',
    script='python3 -m runner --mrunner --config_file=configs/empty.gin',
    python_path='',
    tags=['solving'],
    base_config=base_config, params_grid=params_grid
)
