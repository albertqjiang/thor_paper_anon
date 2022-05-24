import gin
from solvers.baseline_solvers import GreedyBaselineSolver


def configure_solver(goal_generator_class):
    return gin.external_configurable(
        goal_generator_class, module='solvers'
    )

GreedyBaselineSolver = configure_solver(GreedyBaselineSolver)
