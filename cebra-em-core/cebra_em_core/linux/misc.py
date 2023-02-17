
from subprocess import run


def show_in_editor(filepath):

    run([
        'bash --login -c '
        f'"gedit {filepath}"'
    ], shell=True)
