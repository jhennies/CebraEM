
import os
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

from pybdv.metadata import get_data_path, get_attributes, get_resolution
from cebra_em_core.project_utils.config import absolute_path, get_config, add_to_config_json, get_config_filepath
from cebra_em_core.dataset.bdv_utils import is_h5, get_shape, create_empty_dataset
from pybdv.util import get_key, open_file


def get_mobie_project_path(project_path=None, relpath=False):

    mobie_rel = get_config('main', project_path=project_path)['mobie_project_path']

    if relpath:
        return mobie_rel
    else:
        return absolute_path(mobie_rel, project_path=project_path)


def copy_bdv_xml(xml_in, xml_out):

    root = ET.parse(xml_in).getroot()

    # Change the location of the data
    seqdesc = root.find('SequenceDescription')
    imgload = seqdesc.find('ImageLoader')
    data_format = imgload.get('format').split('.')[-1]
    dataloc = imgload.find(data_format)

    data_path = dataloc.text

    out_data_path = os.path.relpath(
        os.path.join(os.path.dirname(xml_in), data_path),
        os.path.dirname(xml_out)
    )

    dataloc.set('type', 'relative')
    dataloc.text = out_data_path

    tree = ET.ElementTree(root)
    tree.write(xml_out)


# def _update_image_name(xml_path, image_name):
#     et = ET.parse(xml_path).getroot()
#     setups = et.find("SequenceDescription").find("ViewSetups").findall("ViewSetup")
#     for vs in setups:
#         if vs.find('id').text == '0':
#             nm = vs.find('name')
#             nm.text = image_name
#     tree = ET.ElementTree(et)
#     tree.write(xml_path)


def resolution_to_micrometer(xml_path):
    et = ET.parse(xml_path).getroot()
    # Change the unit in the view setups
    setups = et.find("SequenceDescription").find("ViewSetups").findall("ViewSetup")
    change_ratio = 1
    for vs in setups:
        vx = vs.find("voxelSize")
        unit = vx.find("unit")
        if unit.text == 'nanometer':
            print('Changing unit to micrometer ...')
            size = vx.find("size")
            size_arr = np.array([float(x) / 1000 for x in str.split(size.text, ' ')])
            size.text = str.join(' ', [str(x) for x in size_arr])
            unit.text = 'micrometer'
            change_ratio = 1000
        elif unit.text == 'micrometer':
            pass
        else:
            raise RuntimeError(f'Invalid unit: {unit.text}; Valid units: "micrometer", "nanometer"')

    # Also change it for the View registration
    if change_ratio != 1:
        affine = et.find("ViewRegistrations").find("ViewRegistration").find("ViewTransform").find("affine")
        affine_arr = np.array([float(x) / change_ratio for x in str.split(affine.text, ' ')])
        affine.text = str.join(' ', [str(x) for x in affine_arr])

    # Write the result
    tree = ET.ElementTree(et)
    tree.write(xml_path)


def get_mobie_table_path(project_path=None):
    from cebra_em_core.project_utils.project import get_current_project_path
    project_path = get_current_project_path(project_path)

    return os.path.join(project_path, 'mobie.csv')


def append_mobie_table(table_filepath, entry):

    import pandas as pd

    table_data = pd.DataFrame()
    if os.path.exists(table_filepath):
        table_data = pd.read_csv(table_filepath, sep='\t')

    new_table_data = pd.concat([table_data, pd.DataFrame(entry)], ignore_index=True, sort=False)
    new_table_data = new_table_data.fillna('')

    new_table_data.to_csv(table_filepath, index=False, sep='\t')


def init_with_raw(mobie_data_path, raw_xml_path, image_name, project_path=None, verbose=False):

    new_xml_path = os.path.join(mobie_data_path, f'{image_name}.xml')
    copy_bdv_xml(raw_xml_path, new_xml_path)
    resolution_to_micrometer(new_xml_path)

    mobie_table_path = get_mobie_table_path(project_path=project_path)

    # Create the mobie project table
    append_mobie_table(
        mobie_table_path,
        dict(
            uri=[new_xml_path],
            type=['intensities'],
            view=['em-raw'],
            group=['inputs']
        )
    )

    # Update the main config json
    add_to_config_json(
        get_config_filepath('main', project_path=project_path),
        {'mobie_table_filepath': mobie_table_path}
    )

    # Update the raw config json
    raw_attributes = get_attributes(new_xml_path, 0)
    raw_resolution = get_resolution(new_xml_path, 0)
    raw_shape = get_shape(new_xml_path, 0)
    if verbose:
        print('raw_attributes = {}'.format(raw_attributes))
        print('raw_resolution = {}'.format(raw_resolution))
        print('raw_shape = {}'.format(raw_shape))

    add_to_config_json(
        get_config_filepath('raw', project_path=project_path),
        {
            'resolution': raw_resolution if type(raw_resolution) != np.ndarray else raw_resolution.tolist(),
            'shape': raw_shape,
            'xml_path': raw_xml_path
        }
    )


def get_dataset_path(dataset_name, project_path=None, relpath=False):
    return os.path.join(
        get_mobie_project_path(project_path=project_path, relpath=relpath),
        dataset_name
    )


def _make_empty_dataset(
        image_name,
        shape,
        mobie_data_path,
        resolution,
        source_type='intensities',
        group='intermediates',
        project_path=None,
        verbose=False
):

    image_data_path = absolute_path(os.path.join(mobie_data_path, f'{image_name}.n5'), project_path=project_path)

    print('Making an empty dataset ...')

    xml_path = create_empty_dataset(
        image_data_path,
        0, 0,
        shape,
        data_dtype='uint64' if source_type == 'segmentation' else 'uint8',
        chunks=None,
        scale_factors=[[2, 2, 2], [2, 2, 2], [4, 4, 4]],
        resolution=resolution,
        unit='micrometer',
        setup_name=image_name
    )

    if verbose:
        print(f'xml_path = {xml_path}')

    xml_rel_path = os.path.relpath(xml_path, project_path)

    if source_type == 'labels':
        # Add max ID
        with open_file(image_data_path, 'a') as f:
            f[get_key(False, 0, 0, 0)].attrs['maxId'] = 0

    append_mobie_table(
        get_mobie_table_path(project_path=project_path),
        dict(
            uri=[xml_path],
            type=[source_type],
            view=[image_name],
            group=[group]
        )
    )

    return xml_rel_path


def init_membrane_prediction(
        mobie_data_path,
        project_path=None,
        verbose=False
):
    config_mem_fp = get_config_filepath('membrane_prediction', project_path=project_path)
    config_mem = get_config('membrane_prediction', project_path=project_path)
    mem_resolution = config_mem['resolution']
    config_raw = get_config('raw', project_path=project_path)
    raw_resolution = config_raw['resolution']
    raw_shape = config_raw['shape']

    # dataset_rel_path = get_dataset_path(dataset_name, project_path, relpath=True)

    if mem_resolution is not None and np.abs(mem_resolution).sum() == 0:
        mem_resolution = None
    if mem_resolution is None:
        mem_resolution = [np.product(raw_resolution) ** (1 / 3)] * 3

    # _______________________________________________________________________________
    # Make an empty dataset
    mem_name = 'membrane_prediction'
    mem_shape = (np.array(raw_shape) * np.array(raw_resolution) / np.array(mem_resolution)).astype(int).tolist()

    print('Making an empty membrane prediction ...')
    xml_rel_path = _make_empty_dataset(
        mem_name,
        mem_shape,
        mobie_data_path,
        mem_resolution,
        'intensities',
        project_path=project_path,
        verbose=verbose
    )

    # _______________________________________________________________________________
    # Update the membrane prediction config json
    if verbose:
        print(f'mem_resolution = {mem_resolution}')
    add_to_config_json(
        config_mem_fp,
        {
            'resolution': mem_resolution,
            'shape': mem_shape,
            'dep_datasets': ['raw'],
            'xml_path': xml_rel_path,
            "data_writing": {
                "type": "image",
                "stitch_method": "crop",
                "stitch_kwargs": {},
                "background_value": None,
                "downscale_mode": "mean",
                "unique_labels": False
            }
        },
        verbose=verbose
    )


def init_supervoxels(
        mobie_data_path,
        project_path=None,
        verbose=False
):

    config_sv_fp = get_config_filepath('supervoxels', project_path=project_path)
    config_sv = get_config('supervoxels', project_path=project_path)
    sv_resolution = config_sv['resolution']
    config_raw = get_config('raw', project_path=project_path)
    raw_resolution = config_raw['resolution']
    raw_shape = config_raw['shape']
    config_mem = get_config('membrane_prediction', project_path=project_path)
    mem_resolution = config_mem['resolution']

    # dataset_rel_path = get_dataset_path(dataset_name, project_path, relpath=True)

    if sv_resolution is not None and np.abs(sv_resolution).sum() == 0:
        sv_resolution = None
    if sv_resolution is None:
        sv_resolution = mem_resolution

    # _______________________________________________________________________________
    # Make an empty dataset
    sv_name = 'em-supervoxels'
    # images_rel_path = os.path.join(dataset_rel_path, 'images', 'bdv-n5')
    sv_shape = (np.array(raw_shape) * np.array(raw_resolution) / np.array(sv_resolution)).astype(int).tolist()

    print('Making an empty supervoxel dataset ...')
    xml_rel_path = _make_empty_dataset(
        sv_name,
        sv_shape,
        mobie_data_path,
        sv_resolution,
        'labels',
        project_path=project_path,
        verbose=verbose
    )

    # _______________________________________________________________________________
    # Update the supervoxel config json
    add_to_config_json(
        config_sv_fp,
        {
            'resolution': sv_resolution,
            'shape': sv_shape,
            'dep_datasets': ['membrane_prediction'],
            'xml_path': xml_rel_path,
            "data_writing": {
                "type": "segmentation",
                "stitch_method": "crop",
                "stitch_kwargs": {},
                "background_value": None,
                "downscale_mode": "nearest",
                "unique_labels": True,
                "dtype": "uint64"
            }
        },
        verbose=verbose
    )


def init_mask(mobie_data_path, mask_xml_path, image_name, project_path=None, verbose=False):

    config_mask = get_config('mask', project_path=project_path)
    method = config_mask['method']
    args = config_mask['args']

    # assert method == 'label_map', "Currently only implemented for method='label_map'"
    assert method in ['label_map', 'binary']

    def _make_table(table_path, ids):

        # the column names
        col_names = ['label_id',
                     'anchor_x', 'anchor_y', 'anchor_z',
                     'bb_min_x', 'bb_min_y', 'bb_min_z',
                     'bb_max_x', 'bb_max_y', 'bb_max_z',
                     'n_pixels']
        data = [[float(idx)] + [0.0] * (len(col_names) - 1) for idx in ids]

        table_dirpath = os.path.split(table_path)[0]
        os.makedirs(table_dirpath, exist_ok=True)
        df = pd.DataFrame(data, columns=col_names)
        df.to_csv(table_path, sep='\t', index=False)

    new_xml_path = os.path.join(mobie_data_path, f'{image_name}.xml')
    copy_bdv_xml(mask_xml_path, new_xml_path)
    resolution_to_micrometer(new_xml_path)

    mobie_table_path = get_mobie_table_path(project_path=project_path)

    append_mobie_table(
        mobie_table_path,
        dict(
            uri=[new_xml_path],
            type=['labels'],
            view=['em-mask'],
            group=['inputs']
        )
    )

    # Add the default table
    table_filepath = os.path.join(mobie_data_path, 'default.csv')
    _make_table(table_filepath, args['ids'])

    # Update the mask config
    mask_attributes = get_attributes(new_xml_path, 0)
    mask_resolution = get_resolution(new_xml_path, 0)
    mask_shape = get_shape(new_xml_path, 0)
    if verbose:
        print('mask_attributes = {}'.format(mask_attributes))
        print('mask_resolution = {}'.format(mask_resolution))
        print('mask_shape = {}'.format(mask_shape))

    add_to_config_json(
        get_config_filepath('mask', project_path=project_path),
        {
            'resolution': mask_resolution if type(mask_resolution) != np.ndarray else mask_resolution.tolist(),
            'shape': mask_shape,
            'xml_path': mask_xml_path
        }
    )

    return mask_resolution, mask_shape


def init_segmentation_map(
        seg_name,
        base_name,
        dataset_name,
        beta,
        project_path=None,
        stitched=False,
        verbose=False
):

    config_seg_fp = get_config_filepath(base_name, project_path=project_path)
    config_seg = get_config(base_name, project_path=project_path)
    seg_resolution = config_seg['resolution']
    config_raw = get_config('raw', project_path=project_path)
    raw_resolution = config_raw['resolution']
    raw_shape = config_raw['shape']
    config_sv = get_config('supervoxels', project_path=project_path)
    sv_resolution = config_sv['resolution']

    if seg_resolution is not None and np.abs(seg_resolution).sum() == 0:
        seg_resolution = None
    if seg_resolution is None:
        seg_resolution = sv_resolution

    # _______________________________________________________________________________
    # Make an empty dataset
    seg_name_hyph = seg_name.replace('_', '-', 1)
    # stitched_name_hyph = f'{seg_name_hyph}_stitch'
    seg_shape = (
            np.array(raw_shape) * np.array(raw_resolution) / np.array(seg_resolution).astype(float)
    ).astype(int).tolist()

    # The non-stitched dataset

    xml_rel_path = _make_empty_dataset(
        seg_name_hyph,
        seg_shape,
        dataset_name,
        seg_resolution,
        project_path=project_path,
        verbose=verbose
    )

    # Not adding the default table until it is actually used!
    # # Add the default table
    # make_table(os.path.join(data_structure_folder, 'tables', seg_name, 'default.csv'))

    # _______________________________________________________________________________
    # Update the segmentation config json

    if not stitched:
        add_to_config_json(
            config_seg_fp,
            {
                'segmentations': {
                    seg_name: {
                        'beta': beta,
                        'resolution': seg_resolution,
                        'shape': seg_shape,
                        'dep_datasets': [  # The order here is super critical, only the last one is used for Snakemake!
                            'raw',
                            'membrane_prediction',
                            'supervoxels'
                        ],
                        'xml_path': xml_rel_path,
                        "data_writing": {
                            "type": "segmentation",
                            "stitch_method": "crop",
                            "stitch_kwargs": {},
                            "background_value": 0,
                            "downscale_mode": "nearest",
                            "unique_labels": True,
                            "dtype": "uint64"
                        },
                        'add_dependencies': [],
                        # 'prepare': 'segmentation'
                    }
                }
            },
            verbose=verbose
        )
    else:
        add_to_config_json(
            config_seg_fp,
            {
                'segmentations': {
                    seg_name[:-9]: {
                        'xml_path_stitched': xml_rel_path,
                    }
                }
            },
            verbose=verbose
        )


