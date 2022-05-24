import os
import subprocess
from time import sleep

from isabelle_utils.evaluation_dataset_utils import get_universal_test_theorems
from isabelle_utils.isabelle_server_utils import IsabelleServerTmuxConnection
from third_party.pisa.src.main.python.PisaFlexibleClient import initialise_env


PORT = 9000
ISA_PATH = '/home/qj213/Isabelle2021'
THEORY_FILE_PATH = '/home/qj213/Isabelle2021/src/HOL/Examples/Drinker.thy'
# WORKING_DIRECTORY = '/home/qj213/Isabelle2021/src/HOL/Examples'
AFP_PATH = '/home/qj213/Research/atp_data/afp-2021-10-22'

# isa_server_connection = IsabelleServerTmuxConnection()
# isa_server_connection.start_isabelle_server(PORT)

def get_imports(problem_num):
    test_theorems = get_universal_test_theorems()
    test_theory_file = test_theorems[str(problem_num)]['relative_path']
    test_lemma = test_theorems[str(problem_num)]['lemma']

    print(test_theory_file)
    print(test_lemma)
    # action = 'by (induction "[] :: \'x list\" ys i rule: sublist_at.induct) auto'
    print(os.path.join(AFP_PATH, test_theory_file))
    env = initialise_env(PORT, ISA_PATH,
                         os.path.join(AFP_PATH, test_theory_file))
    # env.proceed_to_line(test_lemma, 'after')
    imports = env.get_imports()
    print(f'imports = {imports}')


def proof_example_test_theorem(problem_num, action_steps):

    test_theorems = get_universal_test_theorems()
    test_theory_file = test_theorems[str(problem_num)]['relative_path']
    test_lemma = test_theorems[str(problem_num)]['lemma']

    print(test_theory_file)
    print(test_lemma)
    # action = 'by (induction "[] :: \'x list\" ys i rule: sublist_at.induct) auto'
    print(os.path.join(AFP_PATH, test_theory_file))
    env = initialise_env(PORT, ISA_PATH,
                         os.path.join(AFP_PATH, test_theory_file))
    env.proceed_to_line(test_lemma, 'after')
    env.clone_top_level_state('root')

    imports = env.get_imports()
    print(f'imports = {imports}')

    curr_name = 'root'
    for num, action in enumerate(action_steps):
        new_name = f'state_{num}'
        obs, reward, done, info = env.step_to_top_level_state(action, curr_name, new_name)
        curr_name = new_name
        proof_level = env.get_proof_level(curr_name)
        print(f'proof_level = {proof_level}')
        print(f'obs2 = {obs}, rew = {reward} done = {done}')

    return env

def fill_missing_heap(name):
    s1 = f'34.147.124.15:/home/qj213/.isabelle/Isabelle2021/heaps/polyml-5.8.2_x86_64_32-linux/log/{name}.db'
    s2 = f'	34.147.124.15:/home/qj213/.isabelle/Isabelle2021/heaps/polyml-5.8.2_x86_64_32-linux/log/{name}.gz'
    s3 = f'	34.147.124.15:/home/qj213/.isabelle/Isabelle2021/heaps/polyml-5.8.2_x86_64_32-linux/{name}'
    l1 = f'/home/qj213/.isabelle/Isabelle2021/heaps/polyml-5.8.2_x86_64_32-linux/log/{name}.db'
    l2 = f'/home/qj213/.isabelle/Isabelle2021/heaps/polyml-5.8.2_x86_64_32-linux/log/{name}.gz'
    l3 = f'/home/qj213/.isabelle/Isabelle2021/heaps/polyml-5.8.2_x86_64_32-linux/{name}'

    subprocess.run(f'scp {s1} {l1}', shell=True)
    subprocess.run(f'scp {s2} {l2}', shell=True)
    subprocess.run(f'scp {s3} {l3}', shell=True)


def proof_1_plus_1():
    env = initialise_env(PORT, ISA_PATH, THEORY_FILE_PATH)

    env.change_problem(THEORY_FILE_PATH)
    proof_lines = ('theory Test imports Main begin',
                   'lemma abcd: "1+1=2"',
                   'proof -',
                   'show ?thesis',
                   'by auto',
                   'qed',
                   'thm_deps abcd'
                   )
    for step in proof_lines:
        print(f'Applying: {step}')
        obs, reward, done, _ = env.step(step)
        print(f'Obs = {obs}, reward = {reward}, done = {done}')

    env.clone_top_level_state('root')
    # # env.find_thm('root', 'abcd')
    u = env.get_facts('root')
    print(f' facts = {u}')


# fill_missing_heap('HOL-Examples')
# fill_missing_heap('Automatic_Refinement')
# fill_missing_heap('Refine_Monadic')
# fill_missing_heap('Collections')
# fill_missing_heap('Free-Boolean-Algebra')
# fill_missing_heap('HOL-Library')
# fill_missing_heap('HOL-Complex_Analysis')
# get_imports(3)


# proof_1_plus_1()

env = proof_example_test_theorem(0, [' unfolding top_formula_def ', 'by (intro Abs_formula_inverse fba_UNIV)'])
# proof_1_plus_1()


