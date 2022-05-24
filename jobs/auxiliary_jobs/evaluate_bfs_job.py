from jobs.core import Job
from solvers.bfs_solvers import evaluate_on_universal_test_theorems


class EvaluateOnUniversalTestTheoremsJob(Job):
    def __init__(
        self,
        checkpoint_dir,
        isabelle_port,
        isabelle_path,
        prompt_processing_method,
        gcp_bucket,
        save_path,
        experiment_name,
        checkpoint_step,
        starting_index=None,
        ending_index=None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.isabelle_port = isabelle_port
        self.isabelle_path = isabelle_path
        self.prompt_processing_method = prompt_processing_method
        self.gcp_bucket = gcp_bucket
        self.save_path = save_path
        self.experiment_name = experiment_name
        self.checkpoint_step = checkpoint_step

        # parameters with defaults
        self.other_kwargs = {}
        if starting_index is not None:
            self.other_kwargs["starting_index"] = starting_index
        if ending_index is not None:
            self.other_kwargs["ending_index"] = ending_index

    def execute(self):
        evaluate_on_universal_test_theorems(
            checkpoint_dir=self.checkpoint_dir,
            isabelle_port=self.isabelle_port,
            isabelle_path=self.isabelle_path,
            prompt_processing_method=self.prompt_processing_method,
            gcp_bucket=self.gcp_bucket,
            save_path=self.save_path,
            experiment_name=self.experiment_name,
            checkpoint_step=self.checkpoint_step,
            **self.other_kwargs,
        )
