
from subprocess import run


def show_in_editor(filepath):

    run(
        f'notepad {filepath}', shell=True
    )
