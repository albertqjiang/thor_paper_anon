from distutils.command.config import config
import gin

from jobs import hello_job
from jobs.auxiliary_jobs.train_bpe_job import TrainBPEJob
from jobs.baseline_solve_job import BaselineSolveJob
from jobs.auxiliary_jobs.theorems_database_generation_job import TheoremsDatabaseGenerationJob, OutlinesGenerationJob
from jobs.dataset_generation_from_isabelle import DataGenerationFromIsabelleJob
from jobs.train_transformer_job import TrainTransformerJob
from jobs.infer_transformer_job import InferenceTransformerJob
from jobs.auxiliary_jobs.create_data_job import CreateDataJob
from jobs.auxiliary_jobs.evaluate_bfs_job import EvaluateOnUniversalTestTheoremsJob

def configure_job(goal_generator_class):
    return gin.external_configurable(
        goal_generator_class, module='jobs'
    )

JobHello = configure_job(hello_job.HelloJob)

# Data generation jobs:
JobDataGenerationFromIsabelle = configure_job(DataGenerationFromIsabelleJob)
JobTheoremsDatabaseGeneration = configure_job(TheoremsDatabaseGenerationJob)
JobOutlinesGeneration = configure_job(OutlinesGenerationJob)

JobBaselineSolve = configure_job(BaselineSolveJob)
JobInferenceTransformer = configure_job(InferenceTransformerJob)
JobTrainTransformer = configure_job(TrainTransformerJob)

# AuxiliaryJobs:
JobTrainBPE = configure_job(TrainBPEJob)
JobCreateData = configure_job(CreateDataJob)
JobEvaluateBFS = configure_job(EvaluateOnUniversalTestTheoremsJob)
