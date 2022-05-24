from __future__ import print_function

import os
import json
import grpc


from func_timeout import func_set_timeout
import server_pb2_grpc
import server_pb2


def create_stub(port=9000):
    channel = grpc.insecure_channel('localhost:{}'.format(port))
    return server_pb2_grpc.ServerStub(channel)


class IsaFlexEnv:
    def __init__(self, port=9000, isa_path="/Applications/Isabelle2020.app/Isabelle",
                 starter_string="theory Test imports Complex_Main begin",
                 working_directory="/Users/<USER_NAME>/Projects/afp-2021-02-11/thys/Functional-Automata"):
        self.port = port
        self.isa_path = isa_path
        self.starter_string = starter_string
        self.working_directory = working_directory

        self.stub = None
        self.obs_string = None
        self.reset()

    def observation(self):
        return self.obs_string

    def change_problem(self, new_theory_file_path):
        self.destroy_isabelle()
        working_directory = os.path.dirname(os.path.realpath(new_theory_file_path))
        print(self.stub.IsabelleWorkingDirectory(server_pb2.IsaPath(path=working_directory)))
        print(self.stub.IsabelleContext(server_pb2.IsaContext(context=new_theory_file_path)))

    @staticmethod
    def reward(done):
        return 1. if done else 0.

    def reset(self):
        self.stub = create_stub(port=self.port)
        try:
            print(self.stub.InitialiseIsabelle(server_pb2.IsaPath(path=self.isa_path)))
            print(self.stub.IsabelleWorkingDirectory(server_pb2.IsaPath(path=self.working_directory)))
            print(self.stub.IsabelleContext(server_pb2.IsaContext(context=self.starter_string)))
        except Exception as e:
            print("Failure at initialising Isabelle process. "
                  "Make sure the path your provide is where the Isabelle executable is.")
            print(e)
        return self.obs_string

    def destroy_isabelle(self):
        self.stub.IsabelleCommand(server_pb2.IsaCommand(command="exit"))

    @func_set_timeout(10, allowOverride=True)
    def initialise_toplevel_state_map(self):
        try:
            obs_string = self.stub.IsabelleCommand(server_pb2.IsaCommand(command="<initialise>")).state
            print(obs_string)
        except Exception as e:
            print("**Unsuccessful initialisation**")
            print(str(e))

    def delete_toplevel_state(self, tls_name):
        try:
            obs_string = self.stub.IsabelleCommand(server_pb2.IsaCommand(command=f"<delete> {tls_name}")).state
            print(obs_string)
        except Exception as e:
            print("**Unsuccessful deletion**")
            print(str(e))

    @func_set_timeout(10, allowOverride=True)
    def step_to_top_level_state(self, action, tls_name, new_name):
        try:
            command = f"<apply to top level state> {tls_name} <apply to top level state> " \
                      f"{action} <apply to top level state> {new_name}"
            obs_string = self.stub.IsabelleCommand(
                server_pb2.IsaCommand(command=command)).state.strip()
            if obs_string == "Step error":
                done = False
            else:
                # print(obs_string)
                done = self.get_proof_level(new_name) == 0
            return obs_string, self.reward(done), done, {}
        except Exception as e:
            print('Step to top level state exception')
            print(e)
            return None, 0, False, {'exception': 'other', 'details': e}

    def get_facts(self, tls_name):
        facts = self.stub.IsabelleCommand(
            server_pb2.IsaCommand(
                command=f"<get_facts> {tls_name}")).state
        return facts

    def get_imports(self):
        facts = self.stub.IsabelleCommand(
            server_pb2.IsaCommand(
                command=f"<get_imports>")).state
        return facts

    def get_proof_level(self, tls_name):
        proof_level = self.stub.IsabelleCommand(
            server_pb2.IsaCommand(
                command=f"<get_proof_level> {tls_name}")).state
        return int(proof_level)

    def clone_top_level_state(self, old_name, new_name):
        try:
            message = self.stub.IsabelleCommand(
                server_pb2.IsaCommand(command=f"<clone> {old_name} <clone> {new_name}")).state
            print(message)
            print(f"Cloned state called {new_name}")
        except Exception as e:
            print("**Clone unsuccessful**")
            print(e)

    @func_set_timeout(200, allowOverride=True)
    def proceed_to_line(self, line_string, before_after):
        assert before_after in ["before", "after"]
        try:
            obs_string = self.stub.IsabelleCommand(server_pb2.IsaCommand(command=f"<proceed {before_after}> {line_string}")).state
            return obs_string
        except Exception as e:
            print("Failure to proceed before line")
            print(e)

    def extract_theory_steps(self):
        all_steps_str = self.stub.IsabelleCommand(server_pb2.IsaCommand(command="PISA extract actions")).state
        list_of_steps = all_steps_str.split('<\\ISA_STEP>')
        list_of_steps = [step for step in list_of_steps if step[0:2] != '(*' and len(step) > 0]
        return list_of_steps

    def find_thm(self, tls_name, thm_name):
        return self.stub.IsabelleCommand(server_pb2.IsaCommand(command=f"<find_thm> {tls_name} <find_thm> {thm_name}")).state


def parsed_json_to_env_and_dict(path_to_json, afp_path, port=9000, isa_path="/Applications/Isabelle2020.app/Isabelle"):
    save_dict = json.load(open(path_to_json))
    project = save_dict["project"]
    wd = os.path.join(afp_path, "thys", project)
    segments = save_dict["segments"]
    # Find starter string
    starter_string = None
    for line in segments:
        if line.strip().startswith("theory"):
            starter_string = " ".join(line.strip().split("\n"))
            break
    assert starter_string
    return IsaFlexEnv(port=port, isa_path=isa_path,
                     starter_string=starter_string,
                     working_directory=wd), save_dict


@func_set_timeout(100, allowOverride=True)
def initialise_env(port, isa_path, theory_file_path=None, working_directory=None):
    print(f'theory_file_path = {theory_file_path}')
    if "miniF2F" in theory_file_path:
        working_directory = "/home/<USER_NAME>/afp-2021-10-22/thys/Symmetric_Polynomials"
    elif working_directory is None:
        print(f'real_path = {os.path.realpath(theory_file_path)}')
        layers = theory_file_path.split("/")
        while layers[-2] != "thys" and len(layers) > 2:
            layers = layers[:-1]
        assert layers[-2] == "thys"
        working_directory = os.path.join(*layers)
        if not working_directory.startswith("/"):
            working_directory = "/" + working_directory.strip()
        print(f'Automatically detected working directory: {working_directory}')
    return IsaFlexEnv(port=port, isa_path=isa_path, starter_string=theory_file_path,
                      working_directory=working_directory)


def initialise_problem(env, problem_name):
    env.proceed_to_line(problem_name, "after")
    return env
