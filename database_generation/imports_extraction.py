import os
import re
from copy import copy

from pathlib import Path

from database_generation.data_generation_utils import find_all_theory_files


def remove_old_comments(string):
    # goddamn do not leave emojis in the comments
    string = re.sub("\:\)", '', string)
    # remove comments, they are in parentheses
    if "(" not in string:
        return string
    ids = []
    st = 0
    for i, s in enumerate(string):
        if s == '(':
            st += 1
            if st == 1:
                ids.append(i)
        elif s == ')':
            st -= 1
            if not st:
                ids.append(i)
    assert (st == 0 and not len(ids) % 2)
    if len(ids) == 2:
        subs = string[:ids[0]] + string[ids[-1] + 1:]
    else:
        subs = [string[:ids[0]]] + [string[i + 1:j] for i, j in zip(ids[1::2], ids[2::2])] + [string[ids[-1] + 1:]]
        subs = [i.strip() for i in subs]
        subs = ' '.join(subs)
    return subs
def remove_new_comments(string):
    # goddamn do not leave emojis in the comments
    string = re.sub("\:\)", '', string)
    # remove comments, they are in parentheses
    if "<comment>" not in string:
        return string
    ids = []
    st = 0
    for i, s in enumerate(string):
        if string[i:i + 7] == '\<open>':
            st += 1
            if st == 1:
                ids.append(i)
        elif string[i:i + 8] == '\<close>':
            st -= 1
            if not st:
                ids.append(i + 8)
    assert (st == 0 and not len(ids) % 2)
    if len(ids) == 0:
        subs = string
    elif len(ids) == 2:
        subs = string[:ids[0]] + string[ids[-1] + 1:]
    else:
        subs = [string[:ids[0]]] + [string[i + 1:j] for i, j in zip(ids[1::2], ids[2::2])] + [string[ids[-1] + 1:]]
        subs = [i.strip() for i in subs]
        subs = ' '.join(subs)
    subs = subs.replace("\<comment>", '')
    return subs
def remove_keywords(subs):
    if "keywords" in subs:
        subs = subs.split("keywords")[0]
    if "abbrev" in subs:
        subs = subs.split("abbrevs")[0]
    return subs
def get_imports(outline_path):
    with open(outline_path, 'r') as file:
        file_str = file.read()
    try:
        found = re.search('imports((?s).*?)begin', file_str).group(1)
    except AttributeError:
        print("regex failed")
        print(outline_path)
        found = '<none>'  # apply your error handling
        return found
    try:
        found = remove_old_comments(found)
    except:
        print("old remove failed")
        print(outline_path)
        found = '<none>'
        return found
    try:
        found = remove_new_comments(found)
    except:
        print("new remove failed")
        print(outline_path)
        found = '<none>'
        return found
    found = remove_keywords(found)
    interesting_chunk = re.sub("\n", " ", found)
    interesting_chunk = re.sub(' +', ' ', interesting_chunk)
    interesting_chunk = interesting_chunk.strip().split()
    return interesting_chunk

def afp_import_dataset(afp_path):
    names = find_all_theory_files(afp_path)
    import_dict = {}
    for name in names:
        # import_dict[name.split("thys/")[1]] = (get_imports(name))
        import_dict[name] = (get_imports(name))
    return import_dict
def doubledot(path, num_of_doubledots=1):
    return str(Path(path).parents[num_of_doubledots - 1])
def refine(string):
    return "/" + string + ".thy"
def proper_find(file_path, filename):
    return find(filename, clean_to_thys(file_path))
def find(name, path):
    name += ".thy"
    try:
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)
    except:
        raise FileNotFoundError
def clean_to_thys(path):
    assert ("/thys/" in path)
    a, b = path.split("/thys/")
    suff = b.split("/")[0]
    return a + "/thys/" + suff
def afp_adress(name, afp_path):
    return afp_path.split("/thys")[0] + "/thys/" + name
# only works for AFP

def import_to_path(file_path, import_string, isabelle_src, afp_path):
    """
    I hope not one person has to modify this code in their life. Analysing this could be made equivalent to
    a medium length prison sentence.
    """
    afp_path = afp_path + '/thys'

    EMPTY = {}
    name_switch_dict = {"CAVA_Base": "CAVA_Automata", "SM": "CAVA_LTL_Modelchecker", "Isar_Ref": None,
                        "fuzzyrule": None,
                        "Spec_Check": None, "Lorenz_Approximation": "Ordinary_Differential_Equations",
                        "Collections_Examples": None, "LEM": "CakeML", "HOLCF": "HOL.HOLCF", "UTP-Toolkit": "UTP",
                        "\"HOL-ODE-Numerics.ODE_Numerics\"": None}
    key_beginnings = ["Benchmarks", "CCL", "CTT", "Cube",
                      "Doc", "FOL", "FOLP", "HOL", 'LCF', 'Provers', 'Pure', 'Sequents', 'Tools', 'ZF']
    key_delimiters = [".", "-"]
    key_prefixes = ['', '\"']
    keywords = tuple([p + i + j for i in key_beginnings for j in key_delimiters for p in key_prefixes])
    t = ["HOL-CSP","FOL-Fitting","HOLCF-Utils","HOLCF-Meet","HOLCF-Join-Classes","HOLCF_Prelude"]
    red_herrings = tuple([i + j for i in key_prefixes for j in t])
    if "ODE" in import_string in name_switch_dict.keys():
        import_string = name_switch_dict[import_string]
    if not import_string:
        return EMPTY
    for _ in range(2):
        if import_string.startswith(keywords) and not import_string.startswith(red_herrings):
            import_string = import_string.replace("\"", '').replace("-","/")
            return {"src":find(import_string.split(".")[-1], isabelle_src)}
        if import_string.startswith("HOLCF-Library."):
            return {"src":find(import_string.replace("\"", '').split(".")[-1], isabelle_src)}
        elif import_string in ["Main", "Complex_Main", "Pure", "HOLCF", "ZF"]:
            return {"src":find(import_string.replace("\"", '').split(".")[-1], isabelle_src)}
        elif import_string.startswith("\"../"):
            if import_string in ["\"../Libs/Refine_Monadic/Refine_Monadic\"","\"../JVM/JVMListExample\""]:
                return EMPTY
            return {"afp":proper_find(file_path, import_string.strip("\"").split("/")[-1])}
        elif import_string.startswith("\"~~"):
            return EMPTY
        if "\"" in import_string:
            if "/" in import_string:
                import_string = import_string.replace("\"", '')
                return {"afp":proper_find(file_path, import_string.split("/")[-1])}
            else:
                import_string = import_string.replace("\"", '')
                if import_string in name_switch_dict.keys():
                    import_string = name_switch_dict[import_string]
                if not import_string:
                    return EMPTY
    if "." not in import_string:
        return {"afp":proper_find(file_path, import_string)}
    else:
        theory, file = import_string.split(".")
        if theory in name_switch_dict.keys():
            theory = name_switch_dict[theory]
        if not theory:
            return EMPTY
        return {"afp":proper_find(afp_adress(theory, afp_path), file)}

def file_to_import_list(file_path, isabelle_src, afp_path):
    import_names = get_imports(file_path)
    d = {"afp":[],"src":[]}
    for name in import_names:
        dicct = import_to_path(file_path, name, isabelle_src, afp_path)
        for key,val in dicct.items():
            d[key].append(val)
    return d

