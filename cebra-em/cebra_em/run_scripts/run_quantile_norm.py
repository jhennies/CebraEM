
import json
import numpy as np
from pybdv.util import open_file
from pybdv.metadata import get_data_path, get_key

from cebra_em_core.project_utils.project import get_current_project_path
from cebra_em_core.project_utils.config import get_config, absolute_path
from cebra_em_core.dataset.data import get_quantiles
from cebra_em.run_utils.run_specs import get_run_json
from cebra_em_core.dataset.bdv_utils import is_h5


if __name__ == '__main__':

    # _______________________________________________________________________________
    # Retrieving settings

    project_path = get_current_project_path(None)
    run_json = get_run_json(project_path)

    verbose = run_json['verbose']

    config_raw = get_config('raw', project_path=project_path)
    quantile_spacing = config_raw['quantile_spacing']
    quantile_samples = config_raw['quantile_samples']
    raw_resolution = config_raw['resolution']
    raw_xml_path = absolute_path(config_raw['xml_path'], project_path=project_path)
    raw_path = get_data_path(raw_xml_path, return_absolute_path=True)

    config_mask = get_config('mask', project_path=project_path)
    mask_ids = config_mask['args']['ids']
    mask_resolution = config_mask['resolution']
    mask_ds_level = config_mask['ds_level_for_init']
    mask_xml_path = absolute_path(config_mask['xml_path'], project_path=project_path)
    mask_path = get_data_path(mask_xml_path, return_absolute_path=True)

    # _______________________________________________________________________________
    # Compute the quantiles

    raw_handle = open_file(raw_path, mode='r')[get_key(is_h5(raw_xml_path), 0, 0, 0)]
    with open_file(mask_path, mode='r') as f:
        mask = f[get_key(is_h5(mask_xml_path), 0, 0, mask_ds_level)][:]
        downsampling_factors = np.array(f[get_key(is_h5(mask_xml_path), 0, 0, mask_ds_level)].attrs['downsamplingFactors'])

    mask_resolution = mask_resolution * downsampling_factors

    quantiles = get_quantiles(
        raw_handle,
        mask,
        raw_resolution,
        mask_resolution,
        seg_ids=mask_ids,
        quantile_spacing=quantile_spacing,
        method='sparse' if quantile_samples > 0 else 'dense',
        pixels_per_object=quantile_samples,
        verbose=verbose
    )

    # _______________________________________________________________________________
    # Save the results
    with open(snakemake.output[0], 'w') as f:
        json.dump(quantiles, f, indent=2)
