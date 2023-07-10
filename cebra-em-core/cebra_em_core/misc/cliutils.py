import click
import os

#
# Define some options that are used with multiple CLIs. Defining them once removes
# code duplication, makes it easier to refactor the code and makes help strings
# more consistent
#


def project_path_option(exists=True, **args):
    # If the project already exists, we use the current working directory
    # as the default project directory
    defaults = {}
    if exists:
        defaults["default"] = os.getcwd()

    return click.option(
        "--project-path",
        type=click.Path(file_okay=False, exists=exists, writable=True),
        required=True,
        help=f"Path to the {'(existing) ' if exists else ''}CebraEM project",
        **defaults,
        **args
    )

verbose_option = click.option(
    "--verbose",
    type=bool,
    is_flag=True,
    help="Additional console output (for debugging purposes)",
)

quiet_option = click.option(
    "--quiet",
    type=bool,
    is_flag=True,
    help="Do not print any default job information"
)


execution_options = lambda x: click.option(
        "--cores",
        type=click.IntRange(min=1),
        default=os.cpu_count(),
        help="Maximum number of CPU cores used, defaults to all available cores"
    )(
        click.option(
            "--gpus",
            type=click.IntRange(min=1),
            default=1,
            help="Maximum number of GPUs to be used"
        )
    (
        click.option(
            "--cluster",
            type=str,
            help="Enable running jobs on a cluster, currently only Slurm is supported"
        )
    (
        click.option(
            "--qos",
            type=click.Choice(['lowest', 'low', 'normal', 'high', 'highest']),
            default="normal",
            help="Quality of service for cluster jobs"
        )(x)
    )))


def default_parameter_file(name):
    """Locate a default parameter file"""
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_utils", "params", name))
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Default config file {filename} not found!")
    return filename


def str_to_integers(value, delimiter=",", slicechar=":"):
    """Parse a set of integers including slices """
    if delimiter in value:
        return sum(map(str_to_integers, value.split(delimiter)), ())
    if slicechar in value:
        slice_ = slice(*map(int, value.split(slicechar)))
        return tuple(range(slice_.start, slice_.stop))
    return (int(value),)


bdv_scaling = lambda x: click.option(
        "--bdv-scale-factors",
        type=str,
        default="2,2,4",
        callback=lambda c,p,val: str_to_integers(val),
        show_default=True,
        help="Scales computed for the big data viewer format"
    )(
        click.option(
            "--scale_mode",
            type=click.Choice(["interpolate", "mean", "nearest"]),
            default="mean",
            help="BDV scaling method, e.g. ['nearest', 'mean', 'interpolate']",
            show_default=True,
        )(x)
    )