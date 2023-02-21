
from subprocess import run


def show_in_editor(filepath):

    run([
        'bash --login -c '
        f'"vi {filepath}"'
    ], shell=True)
