import json

import metric_logging
from database_generation.data_generation_utils import find_all_theory_files, get_relative_path
from database_generation.generate_outlines import get_theorems, get_outline
from database_generation.imports_extraction import file_to_import_list
from jobs.core import Job
from smart_open import open



class TheoremsDatabaseGenerationJob(Job):
    def __init__(self, afp_path, src_path, out_path):
        self.afp_path = afp_path
        self.src_path = src_path
        self.out_path = out_path

    def execute(self):
        afp_path = f"{self.afp_path}/thys"
        src_path = f"{self.src_path}/src"
        hol_theorems = self.get_all_theorems('src', src_path)
        afp_theorems = self.get_all_theorems('afp', afp_path)
        all_theorems = {'src' : hol_theorems, 'afp': afp_theorems}
        with open(self.out_path, 'w') as f:
            json.dump(all_theorems, f)

    def get_all_theorems(self, prefix, data_dir):
        assert prefix in ['src', 'afp']
        failed_theories = 0
        theorems_database = {}
        all_theory_files = find_all_theory_files(data_dir)
        for num, theory_file_path in enumerate(all_theory_files):
            relative_path = get_relative_path(prefix, theory_file_path)
            print(f'Processing {theory_file_path}')
            try:
                theorems = get_theorems(theory_file_path)
                theorems_database[relative_path] = theorems
            except:
                failed_theories += 1

            print(f'Done {num} of {len(all_theory_files)}')
            metric_logging.log_scalar(f'Processed {prefix}', num, num/len(all_theory_files))
            metric_logging.log_scalar(f'Failed {prefix}', num, failed_theories)

        return theorems_database


class OutlinesGenerationJob(Job):
    def __init__(self, afp_path, src_path, out_path):
        self.afp_path = afp_path
        self.src_path = src_path
        self.out_path = out_path

    def execute(self):
        afp_path = f"{self.afp_path}/thys"
        src_path = f"{self.src_path}/src"
        src_theorems = self.get_all_theorems('src', src_path)
        afp_theorems = self.get_all_theorems('afp', afp_path)
        all_theorems = {'src' : src_theorems, 'afp': afp_theorems}
        with open(self.out_path, 'w') as f:
            json.dump(all_theorems, f)

    def get_all_theorems(self, prefix, data_dir):
        assert prefix in ['src', 'afp']
        failed_theories = 0
        theorems_database = {}
        all_theory_files = find_all_theory_files(data_dir)
        for num, theory_file_path in enumerate(all_theory_files):
            relative_path = get_relative_path(prefix, theory_file_path)
            print(f'Processing {theory_file_path}')
            try:
                outline = get_outline(theory_file_path)
                theorems_database[relative_path] = outline
            except:
                failed_theories += 1

            print(f'Done {num} of {len(all_theory_files)}')
            metric_logging.log_scalar(f'Processed {prefix}', num, num/len(all_theory_files))
            metric_logging.log_scalar(f'Failed {prefix}', num, failed_theories)

        return theorems_database

class ImportsExtraction(Job):
    def __init__(self, afp_path, isa_path, out_path):
        self.afp_path = afp_path
        self.isa_path = isa_path
        self.out_path = out_path

    def execute(self):
        afp_path = f"{self.afp_path}/thys"
        src_path = f"{self.isa_path}/src"
        hol_imports = self.get_all_imports('src', src_path)
        afp_imports = self.get_all_imports('afp', afp_path)

        all_imports = {'src' : hol_imports, 'afp': afp_imports}
        with open(self.out_path, 'w') as f:
            json.dump(all_imports, f)

    def get_all_imports(self, prefix, data_dir):
        assert prefix in ['src', 'afp']
        failed_theories = 0
        imports_database = {}
        all_theory_files = find_all_theory_files(data_dir)
        for num, theory_file_path in enumerate(all_theory_files):
            relative_path = get_relative_path(prefix, theory_file_path)
            print(f'Processing {theory_file_path}')
            extracted_imports = {'afp': [], 'src': []}
            try:
                imports_absolute = file_to_import_list(theory_file_path, self.isa_path, self.afp_path)
                for import_prefix in ['afp', 'src']:
                    for absolute_file in imports_absolute[import_prefix]:
                        extracted_imports[import_prefix].append(get_relative_path(import_prefix, absolute_file))

            except:
                failed_theories += 1
            imports_database[relative_path] = extracted_imports
            print(f'Done {num} of {len(all_theory_files)}')
            metric_logging.log_scalar(f'Processed {prefix}', num, num/len(all_theory_files))
            metric_logging.log_scalar(f'Failed {prefix}', num, failed_theories)

        return imports_database
