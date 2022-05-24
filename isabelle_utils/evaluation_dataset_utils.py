import json
import os

AFP_PATH = '/afp-2021-12-14-non-working'

def read_evaluation_examples(path):
    universal_test_theorems_all = {}
    for i, eval_problem_file in enumerate(os.listdir(path)):
        with open(os.path.join(path, eval_problem_file), 'r') as f:
            eval_problem_data = json.load(f)[0]
        relative_path = eval_problem_data[0].replace('/home/ywu/afp-2021-02-11/', '')
        local_path = os.path.join(AFP_PATH, relative_path)
        lemma = eval_problem_data[1]
        eval_problem_data = {'relative_path': relative_path,
                             'lemma': lemma}
        universal_test_theorems_all[i] = eval_problem_data
    with open('universal_test_theorems_all.json', 'w') as fp:
        json.dump(universal_test_theorems_all, fp)

def get_universal_test_theorems(path='assets/isabelle/universal_test_theorems_all.json'):
    with open(path, 'r') as f:
        universal_test_theorems = json.load(f)
    return universal_test_theorems

def load_test_theorem(afp_path, universal_test_theorems, problem_num):
    test_theorem = universal_test_theorems[str(problem_num)]
    test_theory_file = test_theorem['relative_path']
    test_lemma = test_theorem['lemma']
    return os.path.join(afp_path, test_theory_file), test_lemma