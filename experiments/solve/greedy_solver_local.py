from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    # run parameters:
    'run.job_class': '@jobs.BaselineSolveJob',

    'BaselineSolveJob.port ': 9000,
    'BaselineSolveJob.isabelle_path': '/home/tomek/Isabelle2021',
    'BaselineSolveJob.afp_path': '/home/tomek/Research/atp_data/afp-2021-10-22',
    'BaselineSolveJob.universal_test_theorems_path': 'assets/isabelle/universal_test_theorems_all.json',
    'BaselineSolveJob.problems_to_solve': [2],
    'BaselineSolveJob.solver_class': '@solvers.GreedyBaselineSolver',
    'BaselineSolveJob.log_samples_p': 0.99,
    'BaselineSolveJob.reset_server_every_n_problems': 10,
    'BaselineSolveJob.compile_pisa': False,

    # 'GreedyBaselineSolver.checkpoint_dir': '/home/tomek/Research/atp_data/checkpoints/small_model',
    'GreedyBaselineSolver.checkpoint_dir': 'gs://n2formal-public-data-europe/pretrained_checkpoints/job_2021_12_15-06_54_09_PM',
    'GreedyBaselineSolver.batch_size': 4,
    'GreedyBaselineSolver.gen_length': 64,
    'GreedyBaselineSolver.top_p': 0.9,
    'GreedyBaselineSolver.temp': 0.8,
    'GreedyBaselineSolver.steps_limit': 10,
    'GreedyBaselineSolver.n_timeout_limit': 16,
    'GreedyBaselineSolver.initialise_env_timeout': 10,
    'GreedyBaselineSolver.proceed_to_line_timeout': 10,
    'GreedyBaselineSolver.step_timeout': 2,

    'use_neptune': True
}

params_grid = {'idx': [0]}

experiments_list = create_experiments_helper(
    experiment_name='solve-baseline-greedy-30M',
    project_name='pmtest/labelleisabelle',
    script='python3 -m runner --mrunner --config_file=configs/empty.gin',
    python_path='',
    tags=['solving'],
    base_config=base_config, params_grid=params_grid
)
