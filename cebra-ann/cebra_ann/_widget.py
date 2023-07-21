import vigra.analysis
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QWidget,
    QLineEdit,
    QGroupBox,
    QSlider,
    QMessageBox,
    QInputDialog,
    QFrame
)
from PyQt5.QtCore import Qt
from napari.utils.notifications import show_info
from ._project import validate_project, AnnProject
from ._data import crop_to_same_shape, segmentation_cleanup
from ._multicut import supervoxel_merging
from ._funcs import get_disk_positions, assert_3d
from ._dialogs import QModifyLayerDialog, QCebraNetDialog, QSupervoxelsDialog
from ._widgets import QSliderLabelEdit
from h5py import File
from copy import deepcopy
import numpy as np
from cebra_em_core.cebra_net import default_model_path
from cebra_em_core.bioimageio.cebra_net import run_cebra_net
from cebra_em_core import pre_processing, watershed_dt_with_probs


ORGANELLES = dict(
    Cytoplasm='cyto',
    ER='er',
    Golgi='gol',
    Mitochondria='mito',
    Nucleus='nuc',
    NuclearEnvelope='ne',
    Resin='rsn',
    Background='bg'
)
MAX_BRUSH_SIZE = 40


# TODO Next steps:

# TODO Good to have: CTRL+E and CTRL+T (transposing axes) should rotate around mouse pointer!
# FIXME layer.editable=False does not work. Would be good to avoid the user altering the labels layers.

class CebraAnnWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        layout = QFormLayout()
        self.setLayout(layout)

        # Project
        self.btn_load_project = QPushButton('Load Project')
        layout.addWidget(self.btn_load_project)
        self.btn_load_project.clicked.connect(self._btn_load_project_onclicked)
        self.btn_save_project = QPushButton('Save Project')
        layout.addWidget(self.btn_save_project)
        self.btn_save_project.clicked.connect(self._btn_save_project_onclicked)
        self.btn_close_project = QPushButton('Close Project')
        layout.addWidget(self.btn_close_project)
        self.btn_close_project.clicked.connect(self._btn_close_project_onclicked)

        # Data inputs (raw, mem, sv) including load and compute buttons
        layout_inputs = QFormLayout()
        self.grp_inputs = QGroupBox('Inputs')
        self.lne_raw = QLineEdit()
        self.lne_mem = QLineEdit()
        self.lne_sv = QLineEdit()
        self.lne_raw.setEnabled(False)
        self.lne_mem.setEnabled(False)
        self.lne_sv.setEnabled(False)
        self.btn_raw_load = QPushButton('Load')
        self.btn_mem_load = QPushButton('Load')
        self.btn_sv_load = QPushButton('Load')
        self.btn_mem_compute = QPushButton('Compute')
        self.btn_sv_compute = QPushButton('Compute')
        self.btn_raw_x = QPushButton('X')
        self.btn_mem_x = QPushButton('X')
        self.btn_sv_x = QPushButton('X')
        self.btn_raw_load.clicked.connect(self._btn_raw_load_onclicked)
        self.btn_mem_load.clicked.connect(self._btn_mem_load_onclicked)
        self.btn_sv_load.clicked.connect(self._btn_sv_load_onclicked)
        self.btn_mem_compute.clicked.connect(self._btn_mem_compute_onclicked)
        self.btn_sv_compute.clicked.connect(self._btn_sv_compute_onclick)
        self.btn_raw_x.clicked.connect(self._btn_raw_x_onclick)
        self.btn_mem_x.clicked.connect(self._btn_mem_x_onclick)
        self.btn_sv_x.clicked.connect(self._btn_sv_x_onclick)
        self.raw_in_layout = QHBoxLayout()
        self.mem_in_layout = QHBoxLayout()
        self.sv_in_layout = QHBoxLayout()
        self.raw_in_layout.addWidget(self.lne_raw)
        self.raw_in_layout.addWidget(self.btn_raw_load)
        self.raw_in_layout.addWidget(self.btn_raw_x)
        self.mem_in_layout.addWidget(self.lne_mem)
        self.mem_in_layout.addWidget(self.btn_mem_compute)
        self.mem_in_layout.addWidget(self.btn_mem_load)
        self.mem_in_layout.addWidget(self.btn_mem_x)
        self.sv_in_layout.addWidget(self.lne_sv)
        self.sv_in_layout.addWidget(self.btn_sv_compute)
        self.sv_in_layout.addWidget(self.btn_sv_load)
        self.sv_in_layout.addWidget(self.btn_sv_x)
        layout_inputs.addRow(QLabel('Raw: '), self.raw_in_layout)
        layout_inputs.addRow(QLabel('Mem: '), self.mem_in_layout)
        layout_inputs.addRow(QLabel('SV: '), self.sv_in_layout)
        self.grp_inputs.setLayout(layout_inputs)
        layout.addWidget(self.grp_inputs)

        # Pre-merging
        layout_pre_merging = QFormLayout()
        self.grp_pre_merging = QGroupBox('Pre-merging')
        self.cmb_weight_method = QComboBox()
        self.cmb_weight_method = QComboBox()
        self.cmb_weight_method.addItems(['mean', 'min'])
        self.sld_beta = QSliderLabelEdit(
            0.5, (0, 1), 0.1,
            decimals=2,
            maximum_line_edit_width=75
        )
        # FIXME sld_beta should have it's own valueChanged event but it works like this for now
        self.sld_beta.sld.valueChanged.connect(self._sld_beta_onvaluechanged)
        self.btn_pre_merging = QPushButton('Compute')
        self.btn_pre_merging.clicked.connect(self._btn_pre_merging_onclicked)
        layout_weight_method = QHBoxLayout()
        layout_weight_method.addWidget(self.cmb_weight_method)
        layout_beta = QHBoxLayout()
        layout_beta.addWidget(self.sld_beta)
        layout_beta.addWidget(self.btn_pre_merging)
        layout_pre_merging.addRow(QLabel('Weight method: '), layout_weight_method)
        layout_pre_merging.addRow(QLabel('Beta: '), layout_beta)
        self.grp_pre_merging.setLayout(layout_pre_merging)
        layout.addWidget(self.grp_pre_merging)

        # - INSTANCE SEGMENTATION -
        layout_instances = QFormLayout()
        #   Line brush size
        self.sld_brush_size = QSliderLabelEdit(
            5, (1, MAX_BRUSH_SIZE), 1,
            decimals=None,
            maximum_line_edit_width=75
        )
        layout_brush_size_row = QHBoxLayout()
        layout_brush_size_row.addWidget(self.sld_brush_size)
        layout_instances.addRow(QLabel('Brush size: '), layout_brush_size_row)
        #   Line Clean-up
        self.chk_merge_small = QCheckBox('Merge Small')
        self.chk_merge_small.setChecked(True)
        self.lne_merge_small = QLineEdit('48')
        self.chk_connected_components = QCheckBox('Conn-comp')
        self.chk_connected_components.setChecked(True)
        layout_clean_up = QHBoxLayout()
        layout_clean_up.addWidget(self.chk_merge_small)
        layout_clean_up.addWidget(self.lne_merge_small)
        layout_clean_up.addWidget(self.chk_connected_components)
        layout_instances.addRow(QLabel('Clean-up: '), layout_clean_up)
        #   Line Actions
        self.btn_clean_up = QPushButton('Run clean-up')
        self.btn_instances = QPushButton('Start')
        layout_actions_instances = QHBoxLayout()
        layout_actions_instances.addWidget(self.btn_clean_up)
        layout_actions_instances.addWidget(self.btn_instances)
        layout_instances.addRow(QLabel('Actions: '), layout_actions_instances)
        #   Add the group
        self.grp_instances = QGroupBox('Instance segmentation')
        self.grp_instances.setLayout(layout_instances)
        layout.addWidget(self.grp_instances)
        #   Callbacks
        self.sld_brush_size.sld.valueChanged.connect(self._sld_brush_size_onvaluechanged)
        self.btn_instances.clicked.connect(self._btn_instances_onclicked)
        self.btn_clean_up.clicked.connect(self._btn_clean_up_onclicked)

        # TODO Remove the single/multi logic?
        #   Instead add a "merge all instances" button
        # Semantic segmentation
        self.cmb_organelle = QComboBox()
        self.cmb_organelle.addItems(['< other >'] + [org for org in ORGANELLES.keys()])
        self.cmb_type = QComboBox()
        self.cmb_type.addItems(['multi', 'single'])
        self.btn_add_semantic = QPushButton('Add')
        self.btn_add_semantic.setMaximumWidth(70)
        self.btn_remove_semantic = QPushButton('Remove')
        self.btn_modify_semantic = QPushButton('Modify')
        self.cmb_sem_layers = QComboBox()
        #
        self.btn_remove_semantic.clicked.connect(self._btn_remove_semantic_onclicked)
        self.btn_modify_semantic.clicked.connect(self._btn_modify_semantic_onclicked)
        self.btn_add_semantic.clicked.connect(self._btn_add_semantic_onclicked)
        #
        layout_organelle_row = QHBoxLayout()
        layout_organelle_row.addWidget(self.cmb_organelle)
        layout_type_row = QHBoxLayout()
        layout_type_row.addWidget(self.cmb_type)
        layout_type_row.addWidget(self.btn_add_semantic)
        layout_sem_layers = QHBoxLayout()
        layout_sem_layers.addWidget(self.cmb_sem_layers)
        layout_btns_row = QHBoxLayout()
        layout_btns_row.addWidget(self.btn_remove_semantic)
        layout_btns_row.addWidget(self.btn_modify_semantic)
        #
        layout_semantics = QFormLayout()
        layout_organelle_row.setContentsMargins(0, 10, 0, 0)
        lbl_organelle = QLabel('Organelle: ')
        lbl_organelle.setContentsMargins(0, 10, 0, 0)
        layout_semantics.addRow(lbl_organelle, layout_organelle_row)
        layout_type_row.setContentsMargins(0, 0, 0, 10)
        lbl_type = QLabel('Type: ')
        lbl_type.setContentsMargins(0, 0, 0, 10)
        layout_semantics.addRow(lbl_type, layout_type_row)
        layout_semantics.addRow(QLabel('Layers: '), layout_sem_layers)
        layout_semantics.addRow(QLabel('Actions:'), layout_btns_row)
        self.grp_semantics = QGroupBox('Semantic segmentation')
        self.grp_semantics.setLayout(layout_semantics)
        layout.addWidget(self.grp_semantics)

        # Property inits
        self._project = None

        # Disable buttons that should not be usable in the beginning
        self._set_project()

        # Key bindings
        self._add_key_bindings()

    def _add_key_bindings(self):

        @self.viewer.bind_key('Control')
        def ctrl_pressed(viewer):

            # It turns out it's super annoying if a semantics layer is selected when you want to start painting.
            # Solution: Switch to the instances layer as soon as CTRL is pressed and a semantic layer is selected
            #   as the CTRL key is currently always used when painting
            if 'instances' in self.viewer.layers:
                if len(self._project.semantics) > 0:
                    print([lyr.name[10:] for lyr in viewer.layers.selection])
                    if 'semantics_' in [lyr.name[:10] for lyr in viewer.layers.selection]:
                        viewer.layers.selection = {viewer.layers['instances']}

        @self.viewer.bind_key('Control-E')
        def ctrl_e_pressed(viewer):

            # (0, 1, 2) -> (2, 0, 1)
            # (2, 0, 1) -> (1, 2, 0)
            # (1, 2, 0) -> (0, 1, 2)

            # Get current order and current pointer location
            current_pos = np.array(viewer.cursor.position)
            current_zoom = viewer.camera.zoom
            current_disp_dim = list(viewer.dims.displayed)
            print(np.array(viewer.camera.center))
            offset_vector = current_pos[current_disp_dim] - np.array(viewer.camera.center[1:])

            current_order = viewer.dims.order

            if current_order == (0, 1, 2):
                viewer.dims.order = (2, 0, 1)
            elif current_order == (2, 0, 1):
                viewer.dims.order = (1, 2, 0)
            elif current_order == (1, 2, 0):
                viewer.dims.order = (0, 1, 2)

            displayed = list(viewer.dims.displayed)

            viewer.camera.center = current_pos[displayed] - offset_vector
            viewer.camera.zoom = current_zoom

            viewer.dims.set_current_step(current_order[-1], current_pos[current_order[-1]])

    def _btn_load_project_onclicked(self, value: bool):

        self._close_project()

        folder = QFileDialog.getExistingDirectory(
            self, 'Select project folder ...'
        )
        if folder is not None and folder != '':
            if validate_project(folder, create=True):
                self._project = AnnProject(folder)
                show_info(f'Project folder set to {folder}')
                self._set_project()
                self._save_project()
            else:
                show_info('No valid project location!')

    def _save_layer(self, name, data=None, attrs=None):
        fp = self._project.get_absolute_path(name=name)
        data = data if data is not None else self.viewer.layers[name].data
        with File(fp, mode='w') as f:
            d = f.create_dataset('data', data=data, compression='gzip')
            if attrs is not None:
                for k, v in attrs.items():
                    d.attrs[k] = v

    def _save_project(self):

        print(f'Saving project to {self._project.path}:')

        # Saves the project.json
        print('Saving project info ...')
        self._project.save()

        # Save the data
        if self._project.raw_touched:
            print(f'Saving raw data ...')
            self._save_layer('raw')
            self._project.raw_touched = False
        if self._project.mem_touched:
            print('Saving membrane prediction ...')
            self._save_layer('mem')
            self._project.mem_touched = False
        if self._project.sv_touched:
            print('Saving supervoxels ...')
            self._save_layer('sv')
            self._project.sv_touched = False
        if self._project.pre_merge_touched:
            print('Saving pre-merging ...')
            self._save_layer('pre_merge')
            self._project.pre_merge_touched = False
        if self._project.instances_touched:
            print('Saving instances ...')
            self._save_layer('instances')
            self._project.instances_touched = False
        if len(self._project.semantics_touched) > 0:
            for sem, touched in self._project.semantics_touched.items():
                if touched:
                    print(f'Saving {sem} ...')
                    self._save_layer(sem)
                    self._project.semantics_touched[sem] = False

        print('... done!')

    def _btn_save_project_onclicked(self, value: bool):

        if self._project is None:
            folder = QFileDialog.getExistingDirectory(
                self, 'Select an empty folder ...'
            )
            if folder is not None and folder != '':
                if validate_project(folder, create=True, empty=True):
                    self._project = AnnProject(folder)
                    show_info(f'Project folder set to {folder}')
                    # Now get the loaded layers into the project (any layers other than raw, mem and sv are ignored)
                    if 'raw' in self.viewer.layers:
                        self._project.set_raw()
                    if 'mem' in self.viewer.layers:
                        self._project.set_mem()
                    if 'sv' in self.viewer.layers:
                        self._project.set_sv()
                    self._set_project(do_not_load_data=True)
                else:
                    show_info('No valid project location! Select an empty folder!')
                    return
            else:
                return

        self._save_project()

    def _close_project(self):

        if self._project is not None:

            # Ask if the current project should really be closed and if the current state should be saved
            answer = QMessageBox.warning(
                self, 'Close Project?',
                'You are closing the current project, if not saved all recent progress will be lost.\n\n'
                'Do you want to save before closing?',
                buttons=QMessageBox.StandardButtons(QMessageBox.Save|QMessageBox.Close|QMessageBox.Cancel)
            )

            if answer != QMessageBox.Cancel:

                if answer == QMessageBox.Save:
                    self._save_project()

                # Remove all layers that belong to the project
                for lyr in self._project.get_all_active_layers():
                    self.viewer.layers.remove(lyr)

                # Close the project
                self._project = None

                # Update the GUI
                self._set_project(do_not_load_data=True)

    def _btn_close_project_onclicked(self, value: bool):

        self._close_project()

    def update_layer(self, name, data, layer_type, visible=True, translate=None):
        layers = self.viewer.layers
        try:
            name in layers
            layers[name].data = data
        except KeyError:
            if layer_type == 'image':
                self.viewer.add_image(data, name=name, visible=visible)
            elif layer_type == 'labels':
                self.viewer.add_labels(data, name=name, visible=visible)
        if translate is not None:
            self.viewer.layers[name].translate = translate

    def _set_layer_props(self):

        def _sv():
            self.viewer.layers['sv'].contour = 1
            self.viewer.layers['raw'].visible = True
            if 'mem' in self.viewer.layers:
                self.viewer.layers['mem'].visible = False
            self.viewer.layers['sv'].visible = True
            # FIXME this doesn't work as intended: As soon as one scrolls through the dataset, it is editable again
            self.viewer.layers['sv'].editable = False

        def _pre_merge():
            self.viewer.layers['pre_merge'].visible = True
            _sv()
            self.viewer.layers['sv'].visible = False

        def _instances():
            self.viewer.layers['instances'].visible = True
            _pre_merge()
            self.viewer.layers['pre_merge'].visible = False

        def _semantics():
            _instances()

        if self._project.semantic_names:
            _semantics()
        elif 'instances' in self.viewer.layers:
            _instances()
        elif 'pre_merge' in self.viewer.layers:
            _pre_merge()
        elif 'sv' in self.viewer.layers:
            _sv()
        elif 'raw' in self.viewer.layers:
            self.viewer.layers['raw'].visible = True
        else:
            pass

    def _load_supervoxels(self):

        # Load supervoxels
        with File(self._project.get_absolute_path(self._project.sv), mode='r') as f:
            sv = f['data'][:]
            attrs = f['data'].attrs

            is_unique = False if 'is_unique' not in attrs else attrs['is_unique']

        if not is_unique:
            dtype = sv.dtype

            # Compute connected components
            tsv = vigra.analysis.labelMultiArray(sv if sv.dtype == 'uint8' else sv.astype('uint32'))
            tsv = tsv if dtype == 'uint8' else tsv.astype(dtype)

            # Save and return
            if len(np.unique(tsv)) != len(np.unique(sv)):
                print('Supervoxels did not have unique IDs!')
                # Make sure the new supervoxel map is not overwriting a file that is not managed by the project.
                if self._project.sv[:9] != '{project}':
                    print('Setting sv path to default to avoid overwriting external data!')
                    self._project.set_sv()
                # Save the supervoxels
                self._save_layer('sv', data=tsv, attrs={'is_unique': True})
                self._project.sv_touched = False
                return tsv
            else:
                return sv
        else:
            return sv

    def _load_data(self):

        data = dict()

        if self._project.raw is not None:
            data['raw'] = File(self._project.get_absolute_path(self._project.raw), mode='r')['data'][:]
        if self._project.mem is not None:
            data['mem'] = File(self._project.get_absolute_path(self._project.mem), mode='r')['data'][:]
        if self._project.sv is not None:
            data['sv'] = self._load_supervoxels()
        if self._project.pre_merge is not None:
            data['pre_merge'] = File(self._project.get_absolute_path(self._project.pre_merge), mode='r')['data'][:]
        if self._project.instances is not None:
            data['instances'] = File(self._project.get_absolute_path(self._project.instances), mode='r')['data'][:]
        if len(self._project.semantics) > 0:
            for sem_name, sem in self._project.semantics.items():
                data[sem_name] = File(self._project.get_absolute_path(sem), mode='r')['data'][:]

        if len(data) > 0:
            translations = self._project.get_translations({k: v.shape for k, v in data.items()})

            for k, v in data.items():
                self.update_layer(
                    k, v,
                    'image' if (k == 'raw' or k == 'mem') else 'labels',
                    visible=True,
                    translate=translations[k]
                )

    def _set_project(self, do_not_load_data=False):
        """
        This function makes sure that all the GUI components are properly set and enabled/disabled
        as well as the project data is loaded.

        :param do_not_load_data: Data loading is omitted
        """

        if self._project is not None:

            # Sets the project information and status to the GUI
            self.viewer.title = f'CebraANN - {self._project.path}'
            self.lne_raw.setText(self._project.raw)
            self.lne_mem.setText(self._project.mem)
            self.lne_sv.setText(self._project.sv)
            self.sld_beta.setValue(self._project.beta)
            self.sld_brush_size.setValue(self._project.brush_size)

            if not do_not_load_data:
                # Load the data and create the respective Napari layers
                self._load_data()
            self._set_layer_props()

        else:

            self.viewer.title = f'CebraANN'
            self.lne_raw.setText('')
            self.lne_mem.setText('')
            self.lne_sv.setText('')
            self.cmb_sem_layers.clear()

        # Figure out which widgets need to be enabled or disabled (visible or hidden)
        if self._project is None:
            self.btn_save_project.setEnabled(True)
            self.grp_inputs.setEnabled(False)
            self.grp_pre_merging.setEnabled(False)
            self.grp_instances.setEnabled(False)
            self.grp_semantics.setEnabled(False)
        else:
            self.btn_save_project.setEnabled(True)
            self.grp_inputs.setEnabled(True)
            self.btn_mem_compute.setEnabled(self._project.raw is not None)
            self.btn_sv_compute.setEnabled(self._project.mem is not None)
            self.grp_pre_merging.setEnabled(
                self._project.raw is not None and self._project.mem is not None and self._project.sv is not None)
            self.grp_instances.setEnabled(
                self._project.raw is not None and self._project.sv is not None)
            self.btn_clean_up.setEnabled(self._project.instance_seg_running)
            self.grp_semantics.setEnabled(self._project.instances is not None)

        # Key bindings etc.
        if self._project is not None:
            if self._project.instance_seg_running:
                self._cb_instance_segmentation()
            if len(self._project.semantics) > 0:
                for sem_name in self._project.semantic_names:
                    self._cb_semantic_segmentation(sem_name)

        # Miscellaneous
        if self._project is not None:
            if self._project.instance_seg_running:
                self.btn_instances.setText('Restart')
            else:
                self.btn_instances.setText('Start')
            if len(self._project.semantics) > 0:
                self.cmb_sem_layers.clear()
                self.cmb_sem_layers.addItems(self._project.get_semantic_names_and_types())

    def _btn_raw_load_onclicked(self, value: bool):

        fp, ext = QFileDialog.getOpenFileName(
            self, 'Select raw data file ...', directory=self._project.path,
            filter='*.h5'
        )
        if fp is not None and fp != '':
            print(f'Opening raw file: {fp}')
            self._project.set_raw(fp)
            # self._set_layer_props()
            self._set_project()

    def _btn_mem_load_onclicked(self, value: bool):
        fp, ext = QFileDialog.getOpenFileName(
            self, 'Select membrane prediction data file ...', directory=self._project.path,
            filter='*.h5'
        )
        if fp is not None and fp != '':
            print(f'Opening mem file: {fp}')
            self._project.set_mem(fp)
            # self._set_layer_props()
            self._set_project()

    def _btn_sv_load_onclicked(self, value: bool):
        fp, ext = QFileDialog.getOpenFileName(
            self, 'Select supervoxel data file ...', directory=self._project.path,
            filter='*.h5'
        )
        if fp is not None and fp != '':
            print(f'Opening sv file: {fp}')
            self._project.set_sv(fp)
            # self._set_layer_props()
            self._set_project()

    def _btn_mem_compute_onclicked(self, value: bool):

        ok, res = QCebraNetDialog().get_results(self._project.mem_params)

        self._project.mem_params = res

        if ok:

            shape = assert_3d(res['shape']) if res['shape'] is not None else self.viewer.layers['raw'].data.shape
            halo = np.array(assert_3d(res['halo']))
            padding = res['padding']
            batch_size = assert_3d(res['batch_size'])
            sigma = res['sigma']
            qnorm_low = res['qnorm_low']
            qnorm_high = res['qnorm_high']

            full_shape = np.array(self.viewer.layers['raw'].data.shape)
            shape = np.array(shape)
            if padding == 'data':
                starts = ((full_shape - shape - halo) / 2).astype(int)
                stops = ((full_shape + shape + halo) / 2).astype(int)
            elif padding == 'zeros':
                starts = ((full_shape - shape) / 2).astype(int)
                stops = ((full_shape + shape) / 2).astype(int)
            print(f'starts = {starts}')
            print(f'stops = {stops}')

            raw = self.viewer.layers['raw'].data[
                  starts[0]: stops[0],
                  starts[1]: stops[1],
                  starts[2]: stops[2]
            ]

            self._project.set_mem()

            raw = pre_processing(raw, sigma, qnorm_low, qnorm_high)

            # run_cebra_net(
            #     raw_channels=[[raw]],
            #     model_filepath=default_model_path,
            #     target_filepath=self._project.get_absolute_path(self._project.mem),
            #     target_size=batch_size,
            #     overlap=(np.array(halo) / 2).astype(int),
            #     squeeze_result=True
            # )

            mem = run_cebra_net(raw)

            with File(self._project.get_absolute_path(self._project.mem), mode='w') as f:
                f.create_dataset('data', data=mem, compression='gzip')
            #     mem = f['data'][:]

            if padding == 'data':
                mem = mem[
                      halo[0]: -halo[0],
                      halo[1]: -halo[1],
                      halo[2]: -halo[2],
                ]
                with File(self._project.get_absolute_path(self._project.mem), mode='w') as f:
                    f.create_dataset('data', data=mem, compression='gzip')

            self.update_layer('mem', mem, 'image', visible=True, translate=None)

            self._set_layer_props()
            self._set_project()

    def _btn_sv_compute_onclick(self, value: bool):

        ok, res = QSupervoxelsDialog().get_results(self._project.sv_params)

        self._project.sv_params = res

        if ok:

            shape = assert_3d(res['shape']) if res['shape'] is not None else self.viewer.layers['mem'].data.shape
            halo = np.array(assert_3d(res['halo']))
            padding = res['padding']
            threshold = res['threshold']
            min_membrane_size = res['min_membrane_size']
            sigma_dt = res['sigma_dt']
            min_segment_size = res['min_segment_size']

            full_shape = np.array(self.viewer.layers['mem'].data.shape)
            shape = np.array(shape)
            if padding == 'data':
                starts = ((full_shape - shape - halo) / 2).astype(int)
                stops = ((full_shape + shape + halo) / 2).astype(int)
            elif padding == 'zeros':
                starts = ((full_shape - shape) / 2).astype(int)
                stops = ((full_shape + shape) / 2).astype(int)
            print(f'starts = {starts}')
            print(f'stops = {stops}')

            mem = self.viewer.layers['mem'].data[
                  starts[0]: stops[0],
                  starts[1]: stops[1],
                  starts[2]: stops[2]
            ]

            self._project.set_sv()

            sv = watershed_dt_with_probs(
                mem,
                threshold=threshold,
                min_membrane_size=min_membrane_size,
                anisotropy=(1, 1, 1),
                sigma_dt=sigma_dt,
                min_segment_size=min_segment_size,
                clean_close_seeds=True
            )

            if padding == 'data':
                sv = sv[
                      halo[0]: -halo[0],
                      halo[1]: -halo[1],
                      halo[2]: -halo[2],
                ]
            with File(self._project.get_absolute_path(self._project.sv), mode='w') as f:
                f.create_dataset('data', data=sv, compression='gzip')

            self.update_layer('sv', sv, 'labels', visible=True, translate=None)

            self._set_layer_props()
            self._set_project()

    def _remove_unused_layers(self):

        to_remove = []
        for lyr in self.viewer.layers:
            if lyr.name == 'raw' and self._project.raw is None:
                to_remove.append('raw')
            if lyr.name == 'mem' and self._project.mem is None:
                to_remove.append('mem')
            if lyr.name == 'sv' and self._project.sv is None:
                to_remove.append('sv')
            if lyr.name == 'pre_merge' and self._project.pre_merge is None:
                to_remove.append('pre_merge')
            if lyr.name == 'instances' and self._project.instances is None:
                to_remove.append('instances')
            if lyr.name[:10] == 'semantics_' and lyr.name not in self._project.semantic_names:
                to_remove.append(lyr.name)

        print(f'to_remove: {to_remove}')

        for lyr_name in to_remove:
            self.viewer.layers.remove(lyr_name)

    def _remove_downstream(self, step, tpe='remove'):
        """
        Sets the projects progress back to the specified step
        :param step: one of ['raw', 'mem', 'sv', 'pre_merge', 'instances', semantics']
        :param tpe: one of ['recompute', 'remove', 'restart']
        :return:
        """
        if tpe == 'remove':
            msg = f'Are you sure you want to remove layer {step}?'
        elif tpe == 'recompute':
            msg = f'Are you sure you want to re-compute step {step}?'
        elif tpe == 'restart':
            msg = f'Are you sure you want to restart step {step}?'
        else:
            raise NotImplementedError()
        answer = QMessageBox.warning(
            self, f'Removing progress!',
            f'{msg}\n\n'
            'Note that all downstream progress will be removed!',
            buttons=QMessageBox.StandardButtons(QMessageBox.Yes|QMessageBox.No)
        )

        if answer == QMessageBox.Yes:

            self._project.set_back(step)
            self._remove_unused_layers()
            self._set_project(do_not_load_data=True)
            return True

        else:
            return False

    def _btn_raw_x_onclick(self, value: bool):

        if self._project.raw is not None:
            self._remove_downstream('raw', tpe='remove')

    def _btn_mem_x_onclick(self, value: bool):

        if self._project.mem is not None:
            self._remove_downstream('mem', tpe='remove')

    def _btn_sv_x_onclick(self, value: bool):

        if self._project.sv is not None:
            self._remove_downstream('sv', tpe='remove')

    def _sld_beta_onvaluechanged(self, value: int):
        # FIXME I didn't yet figure out how to add a custom event to QSliderLabelEdit, so I still have to divide by 100
        self._project.beta = float(value) / 100

    def _btn_pre_merging_onclicked(self, value: bool):

        ok = self._remove_downstream('pre_merge', tpe='recompute') if 'instances' in self.viewer.layers else True

        if ok:
            self._pre_merge()
            self._set_layer_props()

    def _sld_brush_size_onvaluechanged(self, value: int):
        self._project.brush_size = int(value)

    def _btn_instances_onclicked(self, value: bool):

        self._start_instance_segmentation()

    def _btn_clean_up_onclicked(self, value: bool):

        # Get the parameters
        merge_small = self.chk_merge_small.isChecked()
        merge_small_size = int(self.lne_merge_small.text())
        conn_comp = self.chk_connected_components.isChecked()

        # Extract the relevant maps
        input_layer_data = {
            layer.name: layer.data
            for layer in self.viewer.layers if layer.name[:9] == 'semantics' or layer.name == 'instances'
        }
        sv = self.viewer.layers['sv'].data

        # Do the cleanup
        result_layer_data, sv = segmentation_cleanup(
            input_layer_data, sv,
            merge_small=merge_small, fill_holes=False, conn_comp=conn_comp,
            merge_small_size=merge_small_size,
            verbose=False
        )
        # Clearing undo list
        self._reset_history()

        # Write back
        for k, v in result_layer_data.items():
            self.viewer.layers[k].data = v
            self.viewer.layers[k].refresh()
        self.viewer.layers['sv'].data = sv

    def _cb_instance_segmentation(self):
        # The instance segmentation uses layers.data_setitem() to include the napari history for proper undos.

        # FIXME: This doesn't seem like a good way to do it
        # Empty the list to avoid adding the callback multiple times. I'm not expecting any other callbacks in there!
        self.viewer.layers['instances'].mouse_drag_callbacks = []

        @self.viewer.layers['instances'].mouse_drag_callbacks.append
        def lyr_instances_onmousedrag(layer, event):

            yield

            # ----- This is performed BEFORE the mouse move event -----
            # Set the instances layer as being touched
            if (
                (event.button == 1 and 'Control' in event.modifiers and len(event.modifiers) == 1)  # Merging objects
                or (event.button == 2 and 'Control' in event.modifiers and len(event.modifiers) == 1)  # Case drawing supervoxels
                or (event.button == 1 and 'Control' in event.modifiers and 'Shift' in event.modifiers and len(event.modifiers) == 2)  # Case creating new object
            ):
                self._project.instances_touched = True

            # Initialize the drawing event
            if event.button == 1 and 'Control' in event.modifiers and len(event.modifiers) == 1:
                start_label = layer.data[
                    tuple((np.array(event.position) - self.viewer.layers['instances'].translate).astype(int))
                ]
                start_sv_id = self.viewer.layers['sv'].data[
                    tuple((np.array(event.position) - self.viewer.layers['instances'].translate).astype(int))
                ]
                change_list = []
            elif (
                (event.button == 2 and 'Control' in event.modifiers and len(event.modifiers) == 1)  # Case drawing supervoxels
                or (event.button == 1 and 'Control' in event.modifiers and 'Shift' in event.modifiers and len(event.modifiers) == 2)  # Case creating new object
            ):
                self.viewer.layers['sv'].visible = True
                sv_opacity = self.viewer.layers['sv'].opacity
                self.viewer.layers['sv'].opacity = 0.9

                if 'Shift' not in event.modifiers:
                    print(f'Creating')
                    start_label = np.max(layer.data) + 1
                    start_sv_id = None
                else:
                    print(f'Drawing')
                    start_label = layer.data[tuple((np.array(event.position) - self.viewer.layers['instances'].translate).astype(int))]
                    start_sv_id = self.viewer.layers['sv'].data[tuple((np.array(event.position) - self.viewer.layers['instances'].translate).astype(int))]
                change_list = []
            else:
                print(f'Doing nothing')
                start_label = None
                start_sv_id = None
                change_list = None
                sv_opacity = None

            # ----- Stuff inside the loop happens WHILE the mouse move event -----
            while event.type == 'mouse_move':

                if start_label is not None and start_label != 0:
                    brush_size = self._project.brush_size
                    layer.cursor = 'circle'
                    layer.cursor_size = brush_size
                    if event.button == 1 and 'Control' in event.modifiers and len(event.modifiers) == 1:

                        if brush_size == 1:
                            pos = np.array(
                                (np.array(event.position) - self.viewer.layers['instances'].translate).astype(int))
                            # Only determine labels if the position is within the data
                            if (pos < layer.data.shape).all() and (pos >= 0).all():
                                this_labels = [layer.data[tuple(pos)]]
                            else:
                                this_labels = []
                        else:
                            positions = get_disk_positions(
                                brush_size,
                                tuple((np.array(event.position) - self.viewer.layers['instances'].translate).astype(int)),
                                layer.data.shape,
                                self.viewer.dims.order
                            )
                            this_labels = np.unique([layer.data[tuple(pos)] for pos in positions])

                        for this_label in this_labels:

                            if (
                                    this_label != 0  # Reserved for labels already moved to semantic segmentation
                                    and this_label != start_label  # Nothing would change anyways
                                    and this_label not in change_list  # Already made this merge
                            ):
                                print(f'{this_label} -> {start_label}')
                                # Apply the change
                                change_list.append(this_label)
                                indices = np.where(self.viewer.layers['instances'].data == this_label)
                                self.viewer.layers['instances'].data_setitem(
                                    indices, start_label, refresh=True
                                )

                    elif (
                            (event.button == 2 and 'Control' in event.modifiers and len(
                                event.modifiers) == 1)  # Case drawing supervoxels
                            or (event.button == 1 and 'Control' in event.modifiers and 'Shift' in event.modifiers and len(
                                event.modifiers) == 2)  # Case creating new object
                    ):

                        sv_layer = self.viewer.layers['sv']

                        if brush_size == 1:
                            pos = np.array(
                                (np.array(event.position) - self.viewer.layers['instances'].translate).astype(int))
                            if (pos < layer.data.shape).all() and (pos >= 0).all():
                                this_labels = [sv_layer.data[tuple(pos)]]
                                pm_labels = [layer.data[tuple(pos)]]
                            else:
                                this_labels = []
                                pm_labels = []

                        else:
                            positions = get_disk_positions(
                                brush_size,
                                tuple((np.array(event.position) - self.viewer.layers['instances'].translate).astype(int)),
                                layer.data.shape,
                                self.viewer.dims.order
                            )
                            this_labels = [sv_layer.data[tuple(pos)] for pos in positions]
                            pm_labels = [layer.data[tuple(pos)] for pos in positions]

                        for idx, this_label in enumerate(this_labels):

                            pm_label = pm_labels[idx]

                            if (

                                    this_label != start_sv_id
                                    and this_label not in change_list
                                    and pm_label != start_label
                                    and pm_label != 0
                            ):
                                # Apply the change
                                change_list.append(this_label)
                                indices = np.where(self.viewer.layers['sv'].data == this_label)
                                self.viewer.layers['instances'].data_setitem(
                                    indices, start_label, refresh=True
                                )

                yield

            # ----- The following happens AFTER the mouse move event -----
            if (
                (event.button == 2 and 'Control' in event.modifiers and len(event.modifiers) == 1)  # Case drawing supervoxels
                or (event.button == 1 and 'Control' in event.modifiers and 'Shift' in event.modifiers and len(event.modifiers) == 2)  # Case creating new object
            ):
                print(f'Drawing event done!')
                self.viewer.layers['sv'].visible = False
                self.viewer.layers['sv'].opacity = sv_opacity

            layer.cursor = 'standard'

    def _start_instance_segmentation(self, instance_data=None):

        # TODO Add field to choose the source: sv or pre-merge (by default it should be pre-merge if it exists)

        ok = self._remove_downstream('instances', tpe='restart') if self._project.instance_seg_running else True

        if ok:
            self._project.instance_seg_running = True
            if instance_data is None:
                instance_data = deepcopy(self.viewer.layers['pre_merge'].data)
            translate = self.viewer.layers['pre_merge'].translate
            self.update_layer('instances', instance_data, 'labels', visible=True, translate=translate)
            self._project.set_instances(self._project.brush_size)

            # With _project.instance_seg_running set to True this will now do all the key bindings etc.
            self._set_project(do_not_load_data=True)
            self._set_layer_props()

    def _pre_merge(self):
        """Run pre-merging of supervoxels using multicut"""

        beta = self._project.beta

        mem = self.viewer.layers['mem'].data.astype('float32')
        mem /= 255
        sv = self.viewer.layers['sv'].data
        mem, sv = crop_to_same_shape(mem, sv)

        pm = supervoxel_merging(mem, sv, beta=beta, weight_method=self.cmb_weight_method.currentText()) + 1

        translate = self.viewer.layers['mem'].translate

        self.update_layer('pre_merge', pm, 'labels', visible=True, translate=translate)
        self._project.set_pre_merge(beta)

    def _btn_remove_semantic_onclicked(self, value: bool):

        # Get the name of the currently selected layer
        sem_name, _ = self._project.get_semantic_names_and_types(self.cmb_sem_layers.currentText())

        # Make sure that the user knows what (s)he's doing!
        answer = QMessageBox.question(
            self, 'Delete layer?',
            f'Are your sure you want to delete {sem_name}?'
        )

        if answer == QMessageBox.Yes:

            # Move all objects in this layer back to the instances layer
            inst_layer = self.viewer.layers['instances']
            sem_layer = self.viewer.layers[sem_name]
            inst_layer.data[sem_layer.data > 0] = sem_layer.data[sem_layer.data > 0] + inst_layer.data.max() + 1
            inst_layer.refresh()

            # Remove the layer
            self.viewer.layers.remove(sem_name)

            # Remove all the project info on this layer
            del self._project.semantics[sem_name]
            del self._project.semantics_touched[sem_name]
            self._project.semantic_names.remove(sem_name)

            # Update the GUI
            self._set_project(do_not_load_data=True)

    def _btn_modify_semantic_onclicked(self, value: bool):

        # Get the name of the currently selected layer
        sem_name, sem_type = self._project.get_semantic_names_and_types(self.cmb_sem_layers.currentText())

        new_name, new_type = QModifyLayerDialog().get_results(
            sem_name,
            ['multi', 'single'],
            sem_type
        )

        set_project = False

        if new_name is not None and new_name != sem_name:
            print('Updating name')
            # Update the layer name
            self.viewer.layers[sem_name].name = new_name
            # Update the project info
            self._project.semantics[new_name] = self._project.default_semantic(new_name)
            del self._project.semantics[sem_name]
            self._project.semantics_touched[new_name] = True
            del self._project.semantics_touched[sem_name]
            self._project.semantic_types[new_name] = self._project.semantic_types[sem_name]
            del self._project.semantic_types[sem_name]
            idx = self._project.semantic_names.index(sem_name)
            self._project.semantic_names[idx] = new_name
            sem_name = new_name
            set_project = True

        if new_type is not None and new_type != sem_type:
            print('Updating type')
            self._project.semantic_types[sem_name] = new_type
            set_project = True

        if set_project:
            self._set_project(do_not_load_data=True)

    def _btn_add_semantic_onclicked(self, value: bool):
        organelle = self.cmb_organelle.currentText()
        if organelle == '< other >':
            organelle, ok = QInputDialog().getText(
                self, 'Organelle name',
                'Specify a name for the organelle'
            )
            if ' ' in organelle:
                ok = QMessageBox.information(
                    self, 'No spaces!',
                    'Do not include spaces in your organelle names! Use "_" instead.'
                )
        else:
            if organelle in ORGANELLES:
                organelle = ORGANELLES[organelle]
        if organelle:
            name = f'semantics_{organelle}'
            type = self.cmb_type.currentText()
            self._add_semantic_segmentation(name, type)

    def _add_semantic_segmentation(self, name, type, semantic_data=None):
        if semantic_data is None:
            semantic_data = np.zeros(self.viewer.layers['instances'].data.shape, dtype=self.viewer.layers['instances'].data.dtype)
        translate = self.viewer.layers['instances'].translate
        self.update_layer(name, semantic_data, 'labels', visible=True, translate=translate)
        self._project.set_semantic(name, type)

        # With _project.instance_seg_running set to True this will now do all the key bindings etc.
        self._set_project(do_not_load_data=True)

    def _cb_semantic_segmentation(self, name):
        # FIXME: This doesn't seem like a good way to do it
        # Empty the list to avoid adding the callback multiple times. I'm not expecting any other callbacks in there!
        self.viewer.layers[name].mouse_drag_callbacks = []

        @self.viewer.layers[name].mouse_drag_callbacks.append
        def lyr_instances_mdc(layer, event):

            yield

            # This is performed BEFORE the mouse move event
            self._project.instances_touched = True
            self._project.semantics_touched[layer.name] = True

            # TODO: Add Undo for this functionality!
            mouse_pos = tuple((np.array(event.position) - self.viewer.layers['instances'].translate).astype(int))
            tgt_label = layer.data[mouse_pos]
            inst_layer = self.viewer.layers['instances']

            if tgt_label == 0:

                # Move the object to the semantic layer

                src_label = inst_layer.data[mouse_pos]

                if src_label != 0:
                    layer.data[
                        inst_layer.data == src_label] = layer.data.max() + 1
                    inst_layer.data[inst_layer.data == src_label] = 0
                    layer.refresh()
                    inst_layer.refresh()

            else:

                # Move the object back to the instance layer
                inst_layer.data[layer.data == tgt_label] = inst_layer.data.max() + 1
                layer.data[layer.data == tgt_label] = 0
                layer.refresh()
                inst_layer.refresh()

            self._reset_history()

    def _reset_history(self):

        # FIXME: This is supposed to be an intermediate solution until I figure out a better way
        # Reset the history of the instances layer to make sure no CTRL+Z surpasses this event
        for layer in self.viewer.layers:
            try:
                layer._reset_history()
            except:
                pass
