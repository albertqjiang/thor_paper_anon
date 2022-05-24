import pathlib

def find_pisa_path():
    return pathlib.Path(__file__).parent.resolve()
