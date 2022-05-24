import subprocess
import pathlib


def syntactic_filter(raw_autoformalisation_file_path, syntactically_correct_file_path):
    checker_script = f'(cd {pathlib.Path.home()}/Portal-to-ISAbelle;' \
                     f'"runMain pisa.agent.CheckSyntax {raw_autoformalisation_file_path} {syntactically_correct_file_path}")'
    subprocess.check_output(checker_script, shell=True)


def convert_syntactically_correct_statements_to_thy_file(syntactically_correct_file_path, thy_file_path):
    """
    This should add the necessary imports and the theory name to the file
    :param syntactically_correct_file_path: The path to the file containing syntactically correct statements
    :param thy_file_path: The path to the file where we will store the thy file
    :return:
    """
    pass


def merge_successful_proofs(pisa_proofs_path, past_successful_proofs_path, merged_proofs_path):
    """
    We expect each of these paths contains a directory train/last_1_train.txt.
    :param pisa_proofs_path: This should be the original pisa proofs directory
    :param past_successful_proofs_path: This should contain ONLY the file specified above
    :param merged_proofs_path: This should be the directory where the merged proofs will be stored
    :return: A directory containing a train/last_1_train.txt
    """
    pass
