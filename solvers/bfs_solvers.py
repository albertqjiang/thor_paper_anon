import heapq
import time
import logging
import argparse
import json
import os
import subprocess
import pathlib
import traceback

from smart_open import open
from typing import Callable
from dataclasses import dataclass, field
import gin
import numpy as np

from func_timeout.exceptions import FunctionTimedOut
from mesh_transformer_utils.load_model_interactive import InteractiveTransformer
from PisaFlexibleClient import initialise_env
import metric_logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    filename="output.txt",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

STATE_PREFIX = "<ISA_OBS>"
PROOF_PREFIX = "<ISA_PRF>"
TRIMMED_PROOF_PREFIX = "<ISA_TRIM_PRF>"
INPUT_EOS = "<INPUT_EOS>"
OUTPUT_EOS = "<|endoftext|>"
DEFAULT_MAX_LENGTH = 3000


def get_last_k_prefix(k=1):
    return f"<ISA_LAST_{k}>"


def get_state_only_prompt(_, proof_state_string):
    state_string = " ".join(proof_state_string.replace("\n", " ").split()).strip()
    return f"{STATE_PREFIX} {state_string} {INPUT_EOS}"


def get_last_k_proof_steps_prompt(proof_steps_string, proof_state_string, k=1):
    proof_last_k_string = "\\\\n".join(proof_steps_string.split("\n")[-k:])
    state_string = " ".join(proof_state_string.replace("\n", " ").split()).strip()
    return f"{get_last_k_prefix(k)} {proof_last_k_string} {STATE_PREFIX} {state_string} {INPUT_EOS}"


def get_trimmed_proof_and_state(proof_steps_string, proof_state_string):
    state_string = " ".join(proof_state_string.replace("\n", " ").split()).strip()
    state_length = len(state_string)
    proof_steps_string = proof_steps_string.replace("\n", "\\n")
    proof_lines = proof_steps_string.split("\\n")
    proof_lines = [line.strip() for line in proof_lines if len(line.strip()) > 0]
    trimmed_proof_string = ""
    for i in reversed(range(len(proof_lines))):
        if (
            len(trimmed_proof_string) + state_length + len(proof_lines[i].strip())
            > DEFAULT_MAX_LENGTH
        ):
            break
        trimmed_proof_string = f"{proof_lines[i].strip()} \\n {trimmed_proof_string}"
    return f"{TRIMMED_PROOF_PREFIX} {trimmed_proof_string.strip()} {STATE_PREFIX} {state_string} {INPUT_EOS}"


prompt_processing_methods = {
    "state_only": get_state_only_prompt,
    "last_k": get_last_k_proof_steps_prompt,
    "trimmed_proof_and_state": get_trimmed_proof_and_state,
}


@dataclass(order=True)
class BFSNode:
    # Note we use the negative of the accumulative log prob here to make the heap a min heap
    neg_acc_log_prob: float
    toplevel_state_name: str = field(compare=False)
    proof_state_string: str = field(compare=False)
    proof_steps_string: str = field(compare=False)
    proof_length: int = field(compare=False)
    proof_level: int = field(compare=False)


@gin.configurable
class BestFirstSearchSolver:
    def __init__(
        self,
        checkpoint_dir,
        gen_length=64,
        top_p=1.0,
        query_limit=300,
        batch_size=32,
        temp=1.2,
        queue_length=32,
        sampling_method="nucleus",
        initialise_env_timeout=200.0,
        proceed_to_line_timeout=50.0,
        step_timeout=10.0,
        solve_timeout=250.0,
        max_step_timeouts=60,
        return_logits=True,
        process_prompt_method: Callable[[str, str], str] = None,
        step_to_load=None,
        use_sledgehammer=False,
        vanilla=False
    ):
        """
        :param process_prompt_method: function that takes in the state string and the proof string to produce
                                      the language model prompt. If not provided, use the state string as the prompt.
        """
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.gen_length = gen_length
        self.top_p = top_p
        self.temp = temp
        self.queue_length = queue_length
        self.sampling_method = sampling_method
        self.query_limit = query_limit
        self.initialise_env_timeout = initialise_env_timeout
        self.proceed_to_line_timeout = proceed_to_line_timeout
        self.step_timeout = step_timeout
        self.solve_timeout = solve_timeout
        self.max_step_timeouts = max_step_timeouts
        self.process_prompt_method = process_prompt_method
        self.use_sledgehammer = use_sledgehammer
        self.vanilla = vanilla

        logging.info(
            f"bfs_solvers:BestFirstSearchSolver - Using sampling method {sampling_method}"
        )

        self.model = InteractiveTransformer(
            self.checkpoint_dir,
            self.batch_size,
            output_end_token=OUTPUT_EOS,
            return_logits=return_logits,
            step_to_load=step_to_load,
            sampler=self.sampling_method,
        )

        self.model.start()

    @staticmethod
    def get_sampler_options(top_p, temp, batch_size):
        return {
            "top_p": np.ones(batch_size) * top_p,
            "temp": np.ones(batch_size) * temp,
        }

    def initialise(self, port, isabelle_path, test_theory_file, agent_info):
        try:
            time_before_initialise = time.time()
            env = initialise_env(
                port,
                isabelle_path,
                test_theory_file,
                forceTimeout=self.initialise_env_timeout,
            )
            logging.info(
                f"Initialisation time: {(time.time() - time_before_initialise):.2f}"
            )
            logging.info(
                f"Initialised current env with:\n"
                f"port: {port}\nisabelle_path: {isabelle_path}\ntest_theory_file: {test_theory_file}"
            )
            return env
        except FunctionTimedOut:
            agent_info["problem_loaded"] = False
            logging.error("Initialise timeout")
            return None

    def proceed(self, env, test_lemma, before_after, agent_info):
        try:
            time_before_loading = time.time()
            current_obs = env.proceed_to_line(
                test_lemma, before_after, forceTimeout=self.proceed_to_line_timeout
            )
            problem_loading_time = time.time() - time_before_loading
            logging.info(f"Problem loading time: {problem_loading_time:.2f}")
            logging.info(f"Loaded problem with name: {test_lemma}")
            return current_obs, problem_loading_time
        except FunctionTimedOut:
            agent_info["problem_loaded"] = False
            return None

    def apply_new_action(self, env, action, old_name, new_name, custom_timeout=None):
        if custom_timeout is None:
            custom_timeout = self.step_timeout
        try:
            new_obs, _, done, _ = env.step_to_top_level_state(
                action, old_name, new_name, forceTimeout=custom_timeout
            )
            return new_obs, done
        except FunctionTimedOut:
            return False, False

    def solve_multiple_problems_with_timeout(
        self, port, isabelle_path, test_theory_file, test_lemmas
    ):
        env, agent_info = self.get_env_initialised_and_info(
            port=port, isabelle_path=isabelle_path, test_theory_file=test_theory_file
        )
        if env is None:
            raise AssertionError

    def solve_single_problem_with_timeout(
        self, port, isabelle_path, test_theory_file, test_lemma
    ):
        try:
            total_timeout = (
                self.solve_timeout
                + self.initialise_env_timeout
                + self.proceed_to_line_timeout
            )
            proved, proof_steps_string, reward, agent_info = self.solve_single_problem(
                port=port,
                isabelle_path=isabelle_path,
                test_theory_file=test_theory_file,
                test_lemma=test_lemma,
                timeout=total_timeout,
            )
        except FunctionTimedOut:
            proved, proof_steps_string, reward, agent_info = (
                False,
                "",
                0,
                {"result": "timeout"},
            )
        return proved, proof_steps_string, reward, agent_info

    def get_env_initialised_and_info(self, port, isabelle_path, test_theory_file):
        prediction_time = 0
        isabelle_time = 0
        problem_loading_time = 0
        agent_info = {
            "prediction_time": prediction_time,
            "isabelle_time": isabelle_time,
            "problem_loading_time": problem_loading_time,
            "proof_length": -1,
            "successful_proof": "",
        }
        env = self.initialise(port, isabelle_path, test_theory_file, agent_info)
        return env, agent_info

    def solve_single_problem(
        self, port, isabelle_path, test_theory_file, test_lemma, timeout=500.0
    ):
        prediction_time, isabelle_time = 0, 0
        # Initialise env
        env, agent_info = self.get_env_initialised_and_info(
            port=port, isabelle_path=isabelle_path, test_theory_file=test_theory_file
        )
        if env is None:
            return False, "", -1, agent_info

        # Proceed to line
        current_obs, problem_loading_time = self.proceed(
            env, test_lemma, "after", agent_info
        )

        if current_obs is None:
            return False, "", -1, agent_info

        problem_begin_time = time.time()
        env.initialise_toplevel_state_map()

        """
        Priority queue have elements of the type BFSNode.
        """
        default_name = "default"
        priority_queue = [
            BFSNode(
                neg_acc_log_prob=0.0,
                toplevel_state_name=default_name,
                proof_state_string=current_obs,
                proof_steps_string=test_lemma,
                proof_length=0,
                proof_level=env.get_proof_level(default_name),
            )
        ]

        # Two conditions to trigger break: queue empty, query limit reached
        proved = False
        queries_made = 0
        state_name_counter = 0
        number_of_timeouts = 0
        all_prompt_strings = set()

        while (
            priority_queue
            and queries_made < self.query_limit
            and number_of_timeouts < self.max_step_timeouts
        ):
            if time.time() - problem_begin_time > timeout:
                raise FunctionTimedOut

            if proved:
                break

            queries_made += 1

            # Get the first element of the queue
            current_node = heapq.heappop(priority_queue)
            logging.info(f"[NODE ACC LOGPROB]: {current_node.neg_acc_log_prob}")
            if self.process_prompt_method is None:
                prompt_string = current_node.proof_state_string
            else:
                prompt_string = self.process_prompt_method(
                    current_node.proof_steps_string, current_node.proof_state_string
                )
            if (
                self.process_prompt_method is not get_state_only_prompt
                and prompt_string in all_prompt_strings
            ):
                continue
            all_prompt_strings.add(prompt_string)

            time_before_prompt = time.time()
            logging.info(f"[PROMPT] {prompt_string}")
            action_batch = self.model.predict(
                prompt_string,
                self.gen_length,
                self.get_sampler_options(self.top_p, self.temp, self.batch_size),
            )
            if queries_made == 1 and (not self.vanilla) and self.use_sledgehammer:
                action_batch.append(("metis", 0.0))
            logging.info(f"[QUERY] used {queries_made}")
            prediction_time += time.time() - time_before_prompt

            hammered = False
            for action, log_prob in action_batch:
                action = action.strip()
                if "sorry" in action or action.strip().startswith("oops"):
                    continue

                # Apply the action to the current state
                time_before_step = time.time()
                hammer_keywords_exist = (
                    "metis" in action or "meson" in action or "smt" in action
                )
                if self.vanilla:
                    custom_timeout = self.step_timeout
                elif hammer_keywords_exist and (self.use_sledgehammer and not hammered):
                    action = "sledgehammer"
                    custom_timeout = 60.0
                    hammered = True
                elif action.strip() == "sledgehammer":
                    action = action.strip()
                    custom_timeout = 60.0
                    hammered = True
                elif hammer_keywords_exist and (
                    (self.use_sledgehammer and hammered)
                    or ((not self.use_sledgehammer) and hammer_keywords_exist)
                ):
                    continue
                else:
                    custom_timeout = self.step_timeout

                # Increment the state name counter and generate new name for the resulting state
                state_name_counter += 1
                new_name = str(state_name_counter)

                new_obs, done = self.apply_new_action(
                    env,
                    action,
                    current_node.toplevel_state_name,
                    new_name,
                    custom_timeout,
                )

                if isinstance(new_obs, str) and "<hammer>" in new_obs:
                    record_action, new_obs = new_obs.split("<hammer>")
                    record_action, new_obs = record_action.strip(), new_obs.strip()
                else:
                    record_action = action

                if action == "sledgehammer":
                    os.system("ps -ef | grep z3 | awk '{print $2}' | xargs kill -9")
                    os.system("ps -ef | grep veriT | awk '{print $2}' | xargs kill -9")
                    os.system("ps -ef | grep cvc4 | awk '{print $2}' | xargs kill -9")
                    os.system(
                        "ps -ef | grep eprover | awk '{print $2}' | xargs kill -9"
                    )
                    os.system("ps -ef | grep SPASS | awk '{print $2}' | xargs kill -9")
                    os.system("ps -ef | grep csdp | awk '{print $2}' | xargs kill -9")

                isabelle_time += time.time() - time_before_step

                if done:
                    # Theorem proven
                    logging.info(f"[SUCCESS] for action: {record_action}, logprob: {log_prob}")
                    proved = True
                    agent_info["result"] = "proved"
                    agent_info["proof_length"] = current_node.proof_length + 1
                    agent_info["successful_proof"] = (
                        current_node.proof_steps_string + "\n" + record_action
                    )
                    break

                if new_obs is False:
                    # Step timed out
                    number_of_timeouts += 1
                    logging.info(f"[TIMEOUT] for action: {record_action}, logprob: {log_prob}")
                    continue

                if new_obs is None or new_obs == "Step error":
                    # The action failed to advance the state
                    logging.info(f"[FAILED] for action: {record_action}, logprob: {log_prob}")
                    continue

                # If the current step didn't prove the theorem but advanced the state,
                # then add the new state to the priority queue.
                # Trim the queue if it becomes too long.
                logging.info(f"[PROCEED] for action: {record_action}, logprob: {log_prob}")
                heapq.heappush(
                    priority_queue,
                    BFSNode(
                        neg_acc_log_prob=current_node.neg_acc_log_prob - log_prob,
                        toplevel_state_name=new_name,
                        proof_state_string=new_obs,
                        proof_steps_string=current_node.proof_steps_string
                        + "\n"
                        + action,
                        proof_length=current_node.proof_length + 1,
                        proof_level=env.get_proof_level(new_name),
                    ),
                )
                if len(priority_queue) > self.queue_length:
                    index_max_neg_acc_log_prob = priority_queue.index(
                        max(priority_queue)
                    )
                    del priority_queue[index_max_neg_acc_log_prob]
                    heapq.heapify(priority_queue)
                    assert len(priority_queue) == self.queue_length

            env.delete_toplevel_state(current_node.toplevel_state_name)
            del current_node

        agent_info["prediction_time"] = prediction_time
        agent_info["isabelle_time"] = isabelle_time
        agent_info["problem_loading_time"] = problem_loading_time
        env.destroy_isabelle()

        if proved:
            return (
                True,
                agent_info["successful_proof"],
                agent_info["proof_length"],
                agent_info,
            )

        if not priority_queue:
            agent_info["result"] = "queue empty"
        elif queries_made >= self.query_limit:
            agent_info["result"] = "query limit reached"
        elif number_of_timeouts >= self.max_step_timeouts:
            agent_info["result"] = "step timeout limit reached"
        else:
            raise AssertionError("Unknown failure case")

        return False, "", -1, agent_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs the Isabelle proof search agent."
    )
    parser.add_argument(
        "--problem_name_directory",
        "-pnd",
        type=str,
        default="Portal-to-ISAbelle/universal_test_theorems"
    )
    parser.add_argument(
        "--checkpoint-dir",
        "-cd",
        type=str,
        default="gs://<MODEL_PATH>",
    )
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument(
        "--isabelle-path",
        "-ip",
        type=str,
        default=f"{pathlib.Path.home()}/Isabelle2021",
    )
    parser.add_argument(
        "--prompt-processing-method",
        "-ppm",
        choices=list(prompt_processing_methods.keys()),
        default=None,
    )
    parser.add_argument(
        "--gcp-bucket", "-gb", type=str, default="n2formal-public-data-europe"
    )
    parser.add_argument("--save-path", "-sp", type=str, default="eval_results")
    parser.add_argument("--experiment-name", "-en", type=str)
    parser.add_argument("--step", "-s", type=int, default=None)
    parser.add_argument("--starting-index", "-si", type=int, default=0)
    parser.add_argument("--ending-index", "-ei", type=int, default=50)
    parser.add_argument("--use-sledgehammer", "-us", action="store_true", default=False)
    parser.add_argument("--vanilla", "-v", action="store_true", default=False)
    parser.add_argument("--temperature", "-t", type=float, default=1.2)
    return parser.parse_args()


def kill_isabelle():
    os.system("ps -ef | grep scala | awk '{print $2}' | xargs kill -9")
    os.system("ps -ef | grep z3 | awk '{print $2}' | xargs kill -9")
    os.system("ps -ef | grep veriT | awk '{print $2}' | xargs kill -9")
    os.system("ps -ef | grep cvc4 | awk '{print $2}' | xargs kill -9")
    os.system("ps -ef | grep eprover | awk '{print $2}' | xargs kill -9")
    os.system("ps -ef | grep SPASS | awk '{print $2}' | xargs kill -9")
    os.system("ps -ef | grep poly | awk '{print $2}' | xargs kill -9")
    logging.info(f"Killing all scala processes to terminate isabelle servers")
    time.sleep(10)


def start_isabelle():
    isabelle_server_script = (
        f"(cd {pathlib.Path.home()}/Portal-to-ISAbelle; rm -rf target/bg-jobs; "
        'sbt "runMain pisa.server.PisaOneStageServer8000" > output.txt & echo "$!")'
    )
    sub = subprocess.check_output(isabelle_server_script, shell=True)
    subprocess_pid = int(sub.decode("utf-8").strip())
    logging.info(f"Started Isabelle server with PID: {subprocess_pid}")
    logging.info(f"Waiting for Isabelle server to start...")
    time.sleep(15)


def evaluate_on_universal_test_theorems(
    checkpoint_dir,
    isabelle_port,
    isabelle_path,
    prompt_processing_method,
    gcp_bucket,
    save_path,
    experiment_name,
    checkpoint_step,
    starting_index=0,
    ending_index=400,
    use_sledgehammer=False,
    problem_names_directory="Portal-to-ISAbelle/universal_test_theorems",
    vanilla=False,
    temperature=1.2,
):
    prompt_processing_method = (
        prompt_processing_methods[prompt_processing_method]
        if prompt_processing_method
        else None
    )
    bfs_solver = BestFirstSearchSolver(
        checkpoint_dir=checkpoint_dir,
        process_prompt_method=prompt_processing_method,
        step_to_load=checkpoint_step,
        use_sledgehammer=use_sledgehammer,
        vanilla=vanilla,
        temp=temperature
    )

    dir_path = os.path.join(f"{pathlib.Path.home()}", problem_names_directory)

    dump_path = "temp_results"
    if not os.path.isdir(dump_path):
        os.makedirs(dump_path)

    accumulator = metric_logging.MetricsAccumulator()
    for problem_index in range(starting_index, ending_index):
        file_path = os.path.join(dir_path, f"test_name_{problem_index}.json")
        logging.info(f"Starting problem [{problem_index}]")
        kill_isabelle()
        start_isabelle()
        try:
            problem_config = json.load(open(file_path))
            test_theory_file, test_lemma = problem_config[0]
            test_theory_file = test_theory_file.replace(
                "/home/ywu/afp-2021-02-11", f"{pathlib.Path.home()}/afp-2021-10-22"
            )
            (
                proved,
                proof_script,
                proof_length,
                agent_info,
            ) = bfs_solver.solve_single_problem_with_timeout(
                port=isabelle_port,
                isabelle_path=isabelle_path,
                test_theory_file=test_theory_file,
                test_lemma=test_lemma,
            )
            # Directly upload to gcp to save progress
            with open(
                f"gs://{gcp_bucket}/{save_path}/{experiment_name}/{problem_index}.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "proved": proved,
                        "proof_script": proof_script,
                        "proof_length": proof_length,
                        "agent_info": agent_info,
                    },
                    f,
                )
            accumulator.log_metric_to_accumulate(
                f"result:{agent_info.get('result')}", 1
            )
            accumulator.log_metric_to_accumulate("numProblems", 1)
            accumulator.log_metric_to_average("solvedRate", int(proved))
            metric_logging.log_dict_as_scalars(
                step=problem_index, scalars_dict=accumulator.return_scalars()
            )
        except Exception as e:
            logging.warning(f"bfs_solver:main - Suppressed exception {e}")
            traceback.print_exc()

    kill_isabelle()


if __name__ == "__main__":
    args = parse_args()
    evaluate_on_universal_test_theorems(
        checkpoint_dir=args.checkpoint_dir,
        isabelle_port=args.port,
        isabelle_path=args.isabelle_path,
        prompt_processing_method=args.prompt_processing_method,
        gcp_bucket=args.gcp_bucket,
        save_path=args.save_path,
        experiment_name=args.experiment_name,
        checkpoint_step=args.step,
        starting_index=args.starting_index,
        ending_index=args.ending_index,
        use_sledgehammer=args.use_sledgehammer,
        problem_names_directory=args.problem_name_directory,
        vanilla=args.vanilla,
        temperature=args.temperature
    )
