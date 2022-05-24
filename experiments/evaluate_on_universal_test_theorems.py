from mrunner.helpers.specification_helper import create_experiments_helper

exp_name = "state_only_large_lr_and_bs_step_45000_typical"

base_config = {
    "run.job_class": "@jobs.EvaluateOnUniversalTestTheoremsJob",
    "EvaluateOnUniversalTestTheoremsJob.checkpoint_dir": "gs://n2formal-public-data-europe/models/large_lr_and_bs_metric_state_only_2022_02_17-09_21_26_PM",
    "EvaluateOnUniversalTestTheoremsJob.isabelle_port": 8000,
    "EvaluateOnUniversalTestTheoremsJob.isabelle_path": "/home/mateusz/Isabelle2021",
    "EvaluateOnUniversalTestTheoremsJob.prompt_processing_method": "state_only",
    "EvaluateOnUniversalTestTheoremsJob.gcp_bucket": "n2formal-public-data-europe",
    "EvaluateOnUniversalTestTheoremsJob.save_path": "eval_results",
    "EvaluateOnUniversalTestTheoremsJob.experiment_name": exp_name,
    "EvaluateOnUniversalTestTheoremsJob.starting_index": 0,
    "EvaluateOnUniversalTestTheoremsJob.ending_index": 400,
    "BestFirstSearchSolver.sampling_method": "typical",
    "BestFirstSearchSolver.top_p": 0.8,
    "use_neptune": True,
}

params_grid = {"EvaluateOnUniversalTestTheoremsJob.checkpoint_step": [45_000]}

experiments_list = create_experiments_helper(
    experiment_name=exp_name,
    project_name="pmtest/labelleisabelle",
    script="python3 -m runner --mrunner --config_file=configs/empty.gin",
    python_path="",
    tags=["evaluation", "typical"],
    base_config=base_config,
    params_grid=params_grid,
    env={'TOKENIZERS_PARALLELISM': 'false'}
)
