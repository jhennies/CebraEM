from cebra_em.run import run
from cebra_em_core.dataset.preprocessing import normalize_instances, convert_to_bdv
from cebra_em_core.project_utils.gt import init_gt_cube
from cebra_em_core.cebra_em_project import init_project, init_segmentation
from cebra_em_core.project_utils.gt import link_gt_cubes, link_gt_cubes, log_gt

import click


@click.group("cebra")
def cebra():
    pass


cebra.add_command(run)
cebra.add_command(normalize_instances)
cebra.add_command(convert_to_bdv)
cebra.add_command(init_gt_cube)
cebra.add_command(init_project)
cebra.add_command(init_segmentation)
cebra.add_command(link_gt_cubes)
cebra.add_command(log_gt)


if __name__ == "__main__":
    cebra()
