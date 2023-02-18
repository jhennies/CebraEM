
import os


def get_repo_path():
    return os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
