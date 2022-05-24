import os
import re

from database_generation.generated_thy_parser import KEYWORDS

def get_relative_path(prefix, theory_file_path):
    absolute_path = theory_file_path.split("/")
    if prefix == 'afp':
        idx = absolute_path.index('thys')
    else:
        idx = absolute_path.index('src')
    return '/'.join(absolute_path[idx + 1:]).split('.')[0]

def find_all_theory_files(dir_path):
    all_theory_files = set()
    for location in os.walk(dir_path):
        dir_path, _, files = location
        for element in files:
            if isinstance(element, str):
                if element[-4:] == '.thy':
                    all_theory_files.add(f'{dir_path}/{element}')

    return all_theory_files

def get_scope(prefix, relative_path, imports_database):
    all_used_theories = {'src': set(), 'afp': set()}
    all_used_theories[prefix].add(relative_path)

    def record_imports(import_info):
        all_used_theories['src'].update(set(import_info['src']))
        all_used_theories['afp'].update(set(import_info['afp']))

    theories_queue = set()
    def add_to_queue(imports_dict):
        for theory in imports_dict['src']:
            theories_queue.add(('src', theory))
        for theory in imports_dict['afp']:
            theories_queue.add(('afp', theory))

    first_imports = imports_database[prefix][relative_path]
    record_imports(first_imports)
    add_to_queue(first_imports)

    while len(theories_queue) > 0:
        curr_source, curr_path = theories_queue.pop()
        if curr_path in imports_database[curr_source]:
            new_imports = imports_database[curr_source][curr_path]
            record_imports(new_imports)
            add_to_queue(new_imports)

    return all_used_theories

def get_all_theorems_from_scope(scope, theorems_database):
    names2theorems = {}
    theorems2names = {}
    for prefix in ['src', 'afp']:
        for theory_file in scope[prefix]:
            if theory_file in theorems_database[prefix]:
                for thm, info in theorems_database[prefix][theory_file].items():
                    if thm == 'unnamed':
                        names2theorems[thm] = info
                        for unnamed_statement, _ in info:
                            theorems2names[unnamed_statement] = "unnamed"
                    else:
                        statement, line = info
                        names2theorems[thm] = statement
                        theorems2names[statement] = thm

    return names2theorems, theorems2names

def isa_step_lemmas_candidates(step_str):
    pattern = r'[^A-Za-z0-9_]+'
    clean_step_str = re.sub(pattern, ' ', step_str)
    candidates = clean_step_str.split(' ')
    return [x for x in candidates if x not in KEYWORDS]

def isa_step_to_lemmas(step_str, names2thm):
    lemma_candidates = isa_step_lemmas_candidates(step_str)
    used_lemmas = set()
    for lemma in lemma_candidates:
        if lemma in names2thm:
            used_lemmas.add((lemma, names2thm[lemma]))
    return used_lemmas