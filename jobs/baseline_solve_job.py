import numpy as np

import metric_logging
from assets.isabelle.problems_to_load import PROBLEMS_TO_LOAD
from isabelle_utils.evaluation_dataset_utils import load_test_theorem, get_universal_test_theorems
from isabelle_utils.isabelle_server_utils import IsabelleServerTmuxConnection
from jobs.core import Job

class BaselineSolveJob(Job):
    def __init__(self, port,
                 isabelle_path,
                 afp_path,
                 universal_test_theorems_path,
                 solver_class,
                 problems_to_solve=None,
                 log_samples_p=None,
                 reset_server_every_n_problems=10,
                 compile_pisa=True):
        self.port = port
        self.isabelle_path = isabelle_path
        self.afp_path = afp_path
        self.universal_test_theorems_path = universal_test_theorems_path
        self.solver = solver_class()
        self.problems_to_solve = problems_to_solve
        self.log_samples_p = log_samples_p

        self.reset_server_every_n_problems= reset_server_every_n_problems
        self.compile_pisa = compile_pisa

        self.test_theorems = get_universal_test_theorems()

    def execute(self):
        isa_server_connection = IsabelleServerTmuxConnection(compile_pisa=self.compile_pisa)
        isa_server_connection.start_isabelle_server(self.port)

        all_solved = 0
        all_tested = 0

        global_prediction_time = 0.01
        global_isabelle_time = 0.01
        global_loading_time = 0.01

        total_state_len = 0
        total_state_count = 0

        if self.problems_to_solve is None or self.problems_to_solve == 'all':
            self.problems_to_solve = get_universal_test_theorems().keys()
        elif self.problems_to_solve == 'standard':
            self.problems_to_solve = PROBLEMS_TO_LOAD
        for num, problem in enumerate(self.problems_to_solve):
            test_theory_file, test_lemma = load_test_theorem(self.afp_path, self.test_theorems, problem)
            solved, successful_steps, all_steps_taken, agent_info = \
                self.solver.solve_single_problem(self.port, self.isabelle_path, test_theory_file, test_lemma, self.log_samples_p)
            if agent_info['problem_loaded']:
                all_tested += 1
            if solved:
                all_solved += 1
                if 'trajectory_states' in agent_info:
                    metric_logging.log_scalar('proof_length', num, len(set(agent_info['trajectory_states'])))
            else:
                metric_logging.log_scalar('proof_length', num, -1)
            metric_logging.log_scalar('solved', num, int(solved))
            metric_logging.log_scalar('succesfull_steps', num, successful_steps)
            metric_logging.log_scalar('all_steps_taken', num, all_steps_taken)
            metric_logging.log_scalar('problems_tested', num, all_tested)
            if all_tested > 0:
                avg_solved_rate = all_solved/all_tested
            else:
                avg_solved_rate = 0
            metric_logging.log_scalar('avg_solved_rate', num, avg_solved_rate)
            metric_logging.log_scalar('problem_loaded', num, agent_info['problem_loaded'])

            if 'samples_to_log' in agent_info:
                for sample in agent_info['samples_to_log']:
                    metric_logging.log_text('action_samples', sample, True)
            print(f'solved = {solved}, succesfull_steps = {successful_steps} all_steps_taken = {all_steps_taken}')

            if 'trajectory_states' in agent_info:
                metric_logging.log_scalar('all_visited_states', num, len(agent_info['trajectory_states']))
                for state_num, trajectory_state in enumerate(agent_info['trajectory_states']):
                    metric_logging.log_text(f'problem_{num}_states', f'{state_num}: {trajectory_state}', False)

            if 'trajectory_actions' in agent_info:
                for action_num, trajectory_action in enumerate(agent_info['trajectory_actions']):
                    metric_logging.log_text(f'problem_{num}_actions', f'{action_num}: {trajectory_action}', False)

            if 'action_lens' in agent_info:
                if len(agent_info['action_lens']) > 0:
                    metric_logging.log_scalar('avg_action_len', num, np.mean(agent_info['action_lens']))
                    metric_logging.log_scalar('max_action_len', num, max(agent_info['action_lens']))

            if 'state_lens' in agent_info:
                if len(agent_info['state_lens']) > 0:
                    metric_logging.log_scalar('avg_state_len', num, np.mean(agent_info['state_lens']))
                    metric_logging.log_scalar('max_state_len', num, max(agent_info['state_lens']))
                    total_state_len += sum(agent_info['state_lens'])
                    total_state_count += len(agent_info['state_lens'])
                    metric_logging.log_scalar('avg_state_len_global', num, total_state_len/total_state_count)

            metric_logging.log_scalar('time:prediction', num, agent_info['prediction_time'])
            metric_logging.log_scalar('time:isabelle', num, agent_info['isabelle_time'])
            metric_logging.log_scalar('time:loading', num, agent_info['problem_loading_time'])

            global_prediction_time += agent_info['prediction_time']
            global_isabelle_time += agent_info['isabelle_time']
            global_loading_time += agent_info['problem_loading_time']

            metric_logging.log_scalar('time_global:prediction', num, global_prediction_time/(global_loading_time+global_prediction_time+global_isabelle_time))
            metric_logging.log_scalar('time_global:isabelle', num, global_isabelle_time/(global_loading_time+global_prediction_time+global_isabelle_time))
            metric_logging.log_scalar('time_global:loading', num, global_loading_time/(global_loading_time+global_prediction_time+global_isabelle_time))

            if num % self.reset_server_every_n_problems == 0 and num > 0:
                isa_server_connection.restart_isabelle_server(self.port)