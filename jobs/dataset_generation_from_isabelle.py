import glob
import json
import os
import re

from func_timeout import FunctionTimedOut
from joblib import Parallel, delayed

import metric_logging
from database_generation.data_generation_utils import get_all_theorems_from_scope, get_scope, isa_step_to_lemmas, \
    get_relative_path, find_all_theory_files
from isabelle_utils.isabelle_server_utils import IsabelleServerTmuxConnection
from jobs.core import Job
from third_party.pisa.src.main.python import server_pb2
from third_party.pisa.src.main.python.PisaFlexibleClient import initialise_env


def isa_extract_actions(stub, theory_file_path):
    stub.IsabelleContext(server_pb2.IsaContext(context=theory_file_path))
    return stub.IsabelleCommand(server_pb2.IsaCommand(command="PISA extract actions")).state


class DataGenerationFromIsabelleJob(Job):
    def __init__(self, isa_path, afp_path, import_database_path, theorems_database_path, out_path):


        self.isa_path = isa_path
        self.afp_path = afp_path

        with open(import_database_path, 'r') as f:
            self.import_database = json.load(f)

        with open(theorems_database_path, 'r') as f:
            self.theorems_database = json.load(f)

        self.out_path = out_path
        self.isa_tmux = IsabelleServerTmuxConnection(False)

        self.ports = self.isa_tmux.accessible_ports

        for port in self.ports:
            self.isa_tmux.start_isabelle_server(port)

    def single_file_to_data(self, port, prefix, theory_file_path):
        file_relative_path = get_relative_path(prefix, theory_file_path)
        scope = get_scope('afp', file_relative_path, self.import_database)
        names2theorems, _ = get_all_theorems_from_scope(scope, self.theorems_database)
        proofs_key = (file_relative_path, prefix)
        proofs_data = {

            'scope' : scope,
            'init_failed': False,
            'parsing_failed': False,
            'failed_on_step' : None,
            'proofs': [],
        }

        try:
            env = initialise_env(port, self.isa_path, theory_file_path)
            env.clone_top_level_state('root')
            all_steps = env.extract_theory_steps()
        except:
            proofs_data['parsing_failed'] = True
            return proofs_data

        proof_open = False
        proof_level = 0
        state = None

        current_proof = {'statement': None, 'transitions': []}

        for step in all_steps:
            if proof_open and  not step.startswith('text'):
                current_proof['transitions'].append(
                    {'state': state, 'step': step, 'premises': isa_step_to_lemmas(step, names2theorems),
                     'proof_level': proof_level})

            try:
                state, rew, done, _ = env.step_to_top_level_state(step, 'root', 'root')
            except:
                proofs_data['failed_on_step'] = step
                return proofs_data

            proof_level = env.get_proof_level('root')

            if not proof_open:
                if proof_level > 0:
                    proof_open = True
                    current_proof['statement'] = step
            else:
                if proof_level == 0:
                    proof_open = False
                    proofs_data['proofs'].append(current_proof)
                    current_proof = {'statement': None, 'transitions': []}

        env.destroy_isabelle()
        return {proofs_key: proofs_data}

    def many_files_to_data(self, port, prefix, file_list):
        proofs_data = {}
        for theory_file_path in file_list:
            new_proof_data = self.single_file_to_data(port, prefix, theory_file_path)
            proofs_data.update(new_proof_data)
        return proofs_data

    def execute(self):
        all_theory_files = list(find_all_theory_files(self.afp_path))[:23]
        gathered_data = {}

        files_per_channel = 5
        files_to_test = all_theory_files[0:len(self.ports)]

        data_episode = 0
        processed_files = 0
        files_to_process = len(all_theory_files)

        while files_to_process > 0:
            data_batch_start = data_episode * files_per_channel * len(self.ports)
            data_batch_len = min(files_per_channel * len(self.ports), len(all_theory_files))

            files_for_channel = {i : [] for i in range(len(self.ports))}
            for i in range(data_batch_start, data_batch_start + data_batch_len):
                files_for_channel[(i % len(self.ports))].append(all_theory_files[i])

            results = Parallel(n_jobs=len(self.ports))(
                delayed(self.many_files_to_data)(port, 'afp', theory_file_paths)
                for port, theory_file_paths in zip(self.ports, files_for_channel)
            )

            for result in results:
                gathered_data.update(result)

            data_episode += 1
            processed_files += data_batch_len
            files_to_process -= processed_files

            metric_logging.log_scalar('processed', data_batch_start, processed_files)
            metric_logging.log_scalar('to_process', data_batch_start, files_to_process)

        with open(self.out_path, 'w') as f:
            json.dump(gathered_data, f)


if __name__ == "__main__":
    dupa = DataGenerationFromIsabelleJob('/home/qj213/Isabelle2021',
                                         '/home/qj213/Research/atp_data/afp-2021-10-22',
                                         '/home/qj213/Research/atp/assets/isabelle/imports_database_all.json',
                                         '/home/qj213/Research/atp/assets/isabelle/all_theorems.json',
                                         '/home/qj213/Research/atp_data/isabelle_extraction')

    dupa.execute()

# z = dupa.many_files_to_data(9000, 'afp', ['/home/qj213/Research/atp_data/afp-2021-10-22/thys/Knuth_Morris_Pratt/KMP.thy',
#                                           '/home/qj213/Research/atp_data/afp-2021-10-22/thys/Free-Boolean-Algebra/Free_Boolean_Algebra.thy'])

# '/home/qj213/Research/atp_data/afp-2021-10-22/thys/Free-Boolean-Algebra/Free_Boolean_Algebra.thy'

# '/home/qj213/Research/atp_data/afp-2021-10-22/thys/Free-Boolean-Algebra/Free_Boolean_Algebra.thy

# print(z)

# x = glob.glob(f"/home/qj213/Research/atp_data/afp-2021-10-22/thys/**/*.thy")
# print(f"{self.afp_path}/thys/**/*.thy)
# print(x)