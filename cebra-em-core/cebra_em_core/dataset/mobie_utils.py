
import os
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

from pybdv.metadata import get_data_path, get_attributes, get_resolution
from mobie.utils import require_dataset_and_view
from mobie.xml_utils import copy_xml_with_newpath
from mobie.metadata.source_metadata import add_source_to_dataset
from cebra_em_core.project_utils.config import absolute_path, get_config, add_to_config_json, get_config_filepath
from cebra_em_core.dataset.bdv_utils import is_h5, get_shape, create_empty_dataset
from pybdv.util import get_key, open_file


def get_mobie_project_path(project_path=None, relpath=False):

    mobie_rel = get_config('main', project_path=project_path)['mobie_project_path']

    if relpath:
        return mobie_rel
    else:
        return absolute_path(mobie_rel, project_path=project_path)


def _update_image_name(xml_path, image_name):
    et = ET.parse(xml_path).getroot()
    setups = et.find("SequenceDescription").find("ViewSetups").findall("ViewSetup")
    for vs in setups:
        if vs.find('id').text == '0':
            nm = vs.find('name')
            nm.text = image_name
    tree = ET.ElementTree(et)
    tree.write(xml_path)


def _resolution_to_micrometer(xml_path):
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


def init_with_raw(mobie_project_path, dataset_name, raw_xml_path, image_name, project_path=None, verbose=False):

    if is_h5(raw_xml_path):
        data_format = 'bdv.hdf5'
    else:
        data_format = 'bdv.n5'

    # Generate the folder structure for the Mobie project
    # dataset_folder = create_dataset_structure(mobie_project_path, dataset_name, [data_format, 'bdv.n5'])
    dataset_folder = os.path.join(mobie_project_path, dataset_name)
    assert dataset_folder is not None
    if verbose:
        print(f'dataset_folder = {dataset_folder}')

    # Get the location of the raw data from the xml file
    raw_data_path = get_data_path(raw_xml_path, return_absolute_path=True)

    view = require_dataset_and_view(
        root=mobie_project_path,
        dataset_name=dataset_name,
        file_format=data_format,
        source_type='image',
        source_name=image_name,
        menu_name=None,
        view=None,
        is_default_dataset=True,
        contrast_limits=[0, 255]
    )

    # The target xml file in the new Mobie project
    xml_path = os.path.join(dataset_folder, 'images', '{}', f'{image_name}.xml')
    xml_path = xml_path.format(data_format.replace('.', '-'))
    copy_xml_with_newpath(raw_xml_path, xml_path, raw_data_path, path_type='absolute', data_format=data_format)
    _update_image_name(xml_path, image_name)
    _resolution_to_micrometer(xml_path)

    # Add the metadata
    # add_source_metadata(
    #     dataset_folder, 'image', image_name, xml_path,
    #     overwrite=True, view=view
    # )
    add_source_to_dataset(
        dataset_folder, 'image', image_name, xml_path,
        overwrite=True, view=view
    )

    raw_attributes = get_attributes(xml_path, 0)
    raw_resolution = get_resolution(xml_path, 0)
    raw_shape = get_shape(xml_path, 0)
    if verbose:
        print('raw_attributes = {}'.format(raw_attributes))
        print('raw_resolution = {}'.format(raw_resolution))
        print('raw_shape = {}'.format(raw_shape))

    dataset_rel = os.path.join(
        get_config('main', project_path=project_path)['mobie_project_path'],
        dataset_name
    )

    # Update the main config json
    add_to_config_json(
        get_config_filepath('main', project_path=project_path),
        {'dataset_folder': dataset_rel},
        verbose=verbose
    )

    # Update the raw config json
    add_to_config_json(
        get_config_filepath('raw', project_path=project_path),
        {
            'resolution': raw_resolution if type(raw_resolution) != np.ndarray else raw_resolution.tolist(),
            'shape': raw_shape,
            'xml_path': raw_xml_path
        }
    )

    return dataset_folder, raw_resolution, raw_shape


def get_dataset_path(dataset_name, project_path=None, relpath=False):
    return os.path.join(
        get_mobie_project_path(project_path=project_path, relpath=relpath),
        dataset_name
    )


def _make_empty_dataset(
        image_name,
        shape,
        dataset_name,
        resolution,
        source_type='segmentation',
        contrast_limits=None,
        project_path=None,
        verbose=False
):

    dataset_path = get_dataset_path(dataset_name, project_path=project_path, relpath=True)
    if verbose:
        print(f'dataset_path = {dataset_path}')

    images_path = os.path.join(dataset_path, 'images', 'bdv-n5')
    # Add image name and replace the project location
    image_data_path = absolute_path(os.path.join(images_path, f'{image_name}.n5'), project_path=project_path)

    print('Making an empty dataset ...')

    create_empty_dataset(
        image_data_path,
        0, 0,
        shape,
        data_dtype='uint64' if source_type == 'segmentation' else 'uint8',
        chunks=None,
        scale_factors=[[2, 2, 2], [2, 2, 2], [4, 4, 4]],
        resolution=resolution,
        unit='micrometer'
    )

    # The target xml file in the new Mobie project
    xml_rel_path = os.path.join(images_path, f'{image_name}.xml')
    xml_path = absolute_path(xml_rel_path, project_path=project_path)
    _update_image_name(xml_path, image_name)

    if source_type == 'segmentation':
        # Add max ID
        with open_file(image_data_path, 'a') as f:
            f[get_key(False, 0, 0, 0)].attrs['maxId'] = 0

    view = require_dataset_and_view(
        root=get_mobie_project_path(project_path, relpath=False),
        dataset_name=dataset_name,
        file_format='bdv.n5',
        source_type=source_type,
        source_name=image_name,
        menu_name=None,
        view=None,
        is_default_dataset=False,
        contrast_limits=contrast_limits
    )

    # add_source_metadata(
    #     absolute_path(dataset_path, project_path=project_path), 'image', image_name, xml_path,
    #     overwrite=True, view=view
    # )
    add_source_to_dataset(
        absolute_path(dataset_path, project_path=project_path), 'image', image_name, xml_path,
        overwrite=True, view=view
    )

    return xml_rel_path


def init_membrane_prediction(
        dataset_name,
        project_path=None,
        verbose=False
):
    config_mem_fp = get_config_filepath('membrane_prediction', project_path=project_path)
    config_mem = get_config('membrane_prediction', project_path=project_path)
    mem_resolution = config_mem['resolution']
    config_raw = get_config('raw', project_path=project_path)
    raw_resolution = config_raw['resolution']
    raw_shape = config_raw['shape']

    dataset_rel_path = get_dataset_path(dataset_name, project_path, relpath=True)

    if mem_resolution is not None and np.abs(mem_resolution).sum() == 0:
        mem_resolution = None
    if mem_resolution is None:
        mem_resolution = [np.product(raw_resolution) ** (1 / 3)] * 3

    # _______________________________________________________________________________
    # Make an empty dataset
    mem_name = 'em-membrane_prediction'
    images_rel_path = os.path.join(dataset_rel_path, 'images', 'bdv-n5')
    mem_shape = (np.array(raw_shape) * np.array(raw_resolution) / np.array(mem_resolution)).astype(int).tolist()

    print('Making an empty membrane prediction ...')
    xml_rel_path = _make_empty_dataset(
        mem_name,
        mem_shape,
        dataset_name,
        mem_resolution,
        'image',
        contrast_limits=[0, 255],
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
        dataset_name,
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

    dataset_rel_path = get_dataset_path(dataset_name, project_path, relpath=True)

    if sv_resolution is not None and np.abs(sv_resolution).sum() == 0:
        sv_resolution = None
    if sv_resolution is None:
        sv_resolution = mem_resolution

    # _______________________________________________________________________________
    # Make an empty dataset
    sv_name = 'em-supervoxels'
    images_rel_path = os.path.join(dataset_rel_path, 'images', 'bdv-n5')
    sv_shape = (np.array(raw_shape) * np.array(raw_resolution) / np.array(sv_resolution)).astype(int).tolist()

    print('Making an empty supervoxel dataset ...')
    xml_rel_path = _make_empty_dataset(
        sv_name,
        sv_shape,
        dataset_name,
        sv_resolution,
        'segmentation',
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


def init_mask(dataset_folder, mask_xml_path, image_name, project_path=None, verbose=False):

    if is_h5(mask_xml_path):
        data_format = 'bdv.hdf5'
    else:
        data_format = 'bdv.n5'

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

        table_folder = os.path.split(table_path)[0]
        os.makedirs(table_folder, exist_ok=True)
        df = pd.DataFrame(data, columns=col_names)
        df.to_csv(table_path, sep='\t', index=False)

    # Get the location of the mask data from the xml file
    mask_data_path = get_data_path(mask_xml_path, return_absolute_path=True)

    view = require_dataset_and_view(
        root=os.path.split(dataset_folder)[0],
        dataset_name=os.path.split(dataset_folder)[1],
        file_format=data_format,
        source_type='segmentation',
        source_name=image_name,
        menu_name=None,
        view=None,
        is_default_dataset=True
    )
    view['sourceDisplays'][0]['segmentationDisplay']['tables'] = ['default.tsv']

    # The target xml file in the Mobie project
    xml_path = os.path.join(dataset_folder, 'images', '{}', f'{image_name}.xml')
    xml_path = xml_path.format(data_format.replace('.', '-'))
    copy_xml_with_newpath(mask_xml_path, xml_path, mask_data_path, path_type='absolute', data_format=data_format)
    _update_image_name(xml_path, image_name)
    _resolution_to_micrometer(xml_path)

    mask_attributes = get_attributes(xml_path, 0)
    mask_resolution = get_resolution(xml_path, 0)
    mask_shape = get_shape(xml_path, 0)
    if verbose:
        print('mask_attributes = {}'.format(mask_attributes))
        print('mask_resolution = {}'.format(mask_resolution))
        print('mask_shape = {}'.format(mask_shape))

    # Add the default table
    table_folder = os.path.join(dataset_folder, 'tables', 'em-mask')
    _make_table(os.path.join(table_folder, 'default.tsv'), args['ids'])

    # Add the metadata
    # add_source_metadata(
    #     dataset_folder, 'segmentation', image_name, xml_path,
    #     overwrite=True, view=view, table_folder=table_folder
    # )
    add_source_to_dataset(
        dataset_folder, 'segmentation', image_name, xml_path,
        overwrite=True, table_folder=table_folder, view=view)

    # Update config
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


