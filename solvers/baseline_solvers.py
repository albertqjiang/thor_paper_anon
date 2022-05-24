import random
import time

import numpy as np

from func_timeout.exceptions import FunctionTimedOut
from isabelle_utils.evaluation_dataset_utils import get_universal_test_theorems
from mesh_transformer_utils.load_model_interactive import InteractiveTransformer
from third_party.pisa.src.main.python.PisaFlexibleClient import initialise_env


def get_sampler_options(top_p, temp, batch_size):
    return {
        "top_p": np.ones(batch_size) * top_p,
        "temp": np.ones(batch_size) * temp
    }

class MockModel:
    def __init__(self):
        self.prediction_num = 0
        self.mock_actions = (['wrong', 'unfolding top_formula_def'], ['bad action', 'by (intro Abs_formula_inverse fba_UNIV)'])

    def predict(self, obs, gen_len, sampler_options):
        out =  self.mock_actions[self.prediction_num]
        self.prediction_num += 1
        return out

class GreedyBaselineSolver:
    def __init__(self, checkpoint_dir, batch_size, gen_length, top_p, temp, steps_limit,
                 n_timeout_limit, initialise_env_timeout, proceed_to_line_timeout, step_timeout):
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.gen_length = gen_length
        self.top_p = top_p
        self.temp = temp
        self.steps_limit = steps_limit
        self.n_timeout_limit = n_timeout_limit
        self.initialise_env_timeout = initialise_env_timeout
        self.proceed_to_line_timeout = proceed_to_line_timeout
        self.step_timeout = step_timeout

        self.test_theorems = get_universal_test_theorems()
        self.model = InteractiveTransformer(self.checkpoint_dir, self.batch_size, ' <ISA_OBS> ', ' Cambridge', '<|endoftext|>')
        self.model.start()

    def solve_single_problem(self, port, isabelle_path, test_theory_file, test_lemma, log_samples_p, debugg_mode=True):
        agent_info = {}
        prediction_time = 0
        isabelle_time = 0
        problem_loading_time = 0

        agent_info['prediction_time'] = prediction_time
        agent_info['isabelle_time'] = isabelle_time
        agent_info['problem_loading_time'] = problem_loading_time

        if debugg_mode:
            print('Will initialize env')
        try:
            time_before_initialize = time.time()
            env = initialise_env(port, isabelle_path, test_theory_file, forceTimeout=self.initialise_env_timeout)
        except FunctionTimedOut:
            agent_info['problem_loaded'] = False
            return False, -1, -1, agent_info
        if debugg_mode:
            print('Env initialized')
            print('Will proceed to line')
        try:
            current_obs = env.proceed_to_line(test_lemma, 'after', forceTimeout=self.proceed_to_line_timeout)
        except FunctionTimedOut:
            agent_info['problem_loaded'] = False
            current_obs = None
        if debugg_mode:
            print('Line proceeding finised')

        problem_loading_time = time.time() - time_before_initialize
        trajectory_states = [current_obs]
        trajectory_actions = []
        time_out_counter = 0

        all_action_lens = []
        all_state_lens = []
        if current_obs is not None:
            all_state_lens.append(len(current_obs))

        agent_info['trajectory_states'] = trajectory_states
        agent_info['trajectory_actions'] = trajectory_actions

        if current_obs is None:
            agent_info['problem_loaded'] = False
            return False, -1, -1, agent_info

        agent_info['problem_loaded'] = True

        env.clone_top_level_state('root')
        current_state = 'root'
        succesfull_steps = 0
        all_steps_taken = 0
        solved = False
        done = False

        samples_to_log = []

        while True:
            if done:
                solved = True
                break
            elif all_steps_taken >= self.steps_limit:
                break

            if debugg_mode:
                print('Will predict action batch')
            time_before_prediction = time.time()
            action_batch = self.model.predict(
                current_obs, self.gen_length,
                get_sampler_options(self.top_p, self.temp, self.batch_size)
            )
            prediction_time += time.time() - time_before_prediction
            if debugg_mode:
                print(f'Predicted action batch = {action_batch}')

            for action in action_batch:
                all_action_lens.append(len(action))
                new_obs = None
                try:
                    time_before_step = time.time()
                    new_obs, _, done, _ = env.step_to_top_level_state(action, current_state, f'state_{succesfull_steps}', forceTimeout=self.step_timeout)
                    isabelle_time += time.time() - time_before_step
                except FunctionTimedOut:
                    time_out_counter += 1
                    print(f'Timeout')
                    if time_out_counter > self.n_timeout_limit:
                        agent_info['timeout'] = True
                        return False, succesfull_steps, all_steps_taken, agent_info

                if log_samples_p is not None:
                    if random.random() < log_samples_p:
                        samples_to_log.append(f'Action: {action} | Observation: {current_obs}')
                all_steps_taken += 1
                if new_obs is not None:
                    all_state_lens.append(len(new_obs))
                    current_obs = new_obs
                    current_state = f'state_{succesfull_steps}'
                    succesfull_steps += 1
                    trajectory_states.append(current_obs)
                    trajectory_actions.append(action)
                    break

        agent_info['samples_to_log'] = samples_to_log
        agent_info['trajectory_states'] = trajectory_states
        agent_info['trajectory_actions'] = trajectory_actions
        agent_info['proof_length'] = succesfull_steps - 1
        agent_info['action_lens'] = all_action_lens
        agent_info['state_lens'] = all_state_lens
        agent_info['prediction_time'] = prediction_time
        agent_info['isabelle_time'] = isabelle_time
        agent_info['problem_loading_time'] = problem_loading_time

        env.destroy_isabelle()

        return solved, succesfull_steps, all_steps_taken, agent_info

