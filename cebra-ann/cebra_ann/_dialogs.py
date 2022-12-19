
from PyQt5.QtCore import *
from qtpy.QtWidgets import *

from ._widgets import QSliderLabelEdit


class QParameterDialog(QDialog):

    def __init__(self, title='Dialog'):

        super(QParameterDialog, self).__init__()
        self.layout = QFormLayout()
        self.title = title
        
    def setup_ui(self, *args, standard_rows=None, **kwargs):

        # Add the parameter rows as defined in standard_rows
        if standard_rows is not None:

            self.std_rows = []

            for row_spec in standard_rows:

                idx = row_spec['id']
                row_type = row_spec['type']
                default = row_spec['default']
                label = row_spec['label']
                row_kwargs = row_spec['kwargs'] if 'kwargs' in row_spec else None

                lyo_this = QHBoxLayout()

                if row_type == 'line_edit':
                    self.std_rows.append(QLineEdit(default))

                elif row_type == 'combo_box':
                    cmb = QComboBox()
                    cmb.addItems(default[0])
                    cmb.setCurrentText(default[1])
                    self.std_rows.append(cmb)

                elif row_type == 'slider':
                    sld = QSliderLabelEdit(
                        default, **row_kwargs
                    )
                    self.std_rows.append(sld)

                lyo_this.addWidget(self.std_rows[-1])

                self.layout.addRow(QLabel(label), lyo_this)

        # Add the OK and cancel buttons
        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)

        self.setWindowTitle(self.title)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def get_results(self, *args, standard_rows=None, **kwargs):
        """
        Generate a dialog using some standard rows as described below
        :param args:
        :param standard_rows:
            list like so:
            [
                {
                    "id": "id1",
                    "type": "line_edit",
                    "default": def_val1,
                    "label": "Line edit"
                },
                {
                    "id": "id2",
                    "type": "combo_box",
                    "default": [["current_item", "other_item"], "current_item"],
                    "label": "Combo box"
                },
                {
                    "id": "id3",
                    "type": "slider",
                    "default": [["current_item", "other_item"], "current_item"],
                    "label": "Slider",
                    "kwargs": dict with fields:
                        mandatory: "range", "single_step"
                        optional: "decimals", "maximum_line_edit_width"
                },
            ]

            possible types: "line_edit", "combo_box", "slider"
        :param kwargs:
        :return:
        """
        
        self.setup_ui(standard_rows=standard_rows)


class QModifyLayerDialog(QParameterDialog):

    def __init__(self):
        super(QModifyLayerDialog, self).__init__()
        self.title = 'Modify Layer'

    def get_results(self, txt, cmb_list, cmb_item):

        # Add an additional custom explanation
        layout = self.layout
        layout_caption = QHBoxLayout()
        layout_caption.addWidget(QLabel(f'Change name and type of {txt}:'))
        layout_caption.setContentsMargins(0, 0, 0, 10)
        layout.addRow(layout_caption)

        std_rows = [
            dict(
                id='name',
                type='line_edit',
                default=txt[10:],
                label='Name:   semantics_'
            ),
            dict(
                id='type',
                type='combo_box',
                default=[cmb_list, cmb_item],
                label='Type: '
            )
        ]

        self.setup_ui(standard_rows=std_rows)

        if self.exec_() == QDialog.Accepted:
            return f'semantics_{self.std_rows[0].text()}', self.std_rows[1].currentText()
        else:
            return None, None


class QCebraNetDialog(QParameterDialog):

    def __init__(self):
        super(QCebraNetDialog, self).__init__()
        self.title = 'CebraNET parameters'

    def get_results(self, mem_params):

        def _to_str(val):
            if val is None:
                return ''
            elif type(val) is tuple or type(val) is list:
                return str.join(', ', [str(v) for v in val])
            else:
                return f'{val}'

        def _to_val(txt, dtype=float):
            if txt == '':
                return None
            elif ',' in txt:
                values = str.split(txt, ',')
                assert len(values) == 3, 'Supply either only one value or an individual value for all dimensions!'
                return [dtype(v) for v in values]
            else:
                return dtype(txt)

        std_rows = [
            dict(
                id='shape',
                type='line_edit',
                default=_to_str(mem_params['shape']),
                label='Shape: '
            ),
            dict(
                id='halo',
                type='line_edit',
                default=_to_str(mem_params['halo']),
                label='Halo: '
            ),
            dict(
                id='padding',
                type='combo_box',
                default=[['zeros', 'data'], mem_params['padding']],
                label='Padding: '
            ),
            dict(
                id='batch_size',
                type='line_edit',
                default=_to_str(mem_params['batch_size']),
                label='Batch size: '
            ),
            dict(
                id='sigma',
                type='slider',
                default=mem_params['sigma'],
                label='Sigma: ',
                kwargs=dict(
                    range=(0, 10),
                    single_step=0.1,
                    decimals=1,
                    maximum_line_edit_width=70
                )
            ),
            dict(
                id='qnorm_low',
                type='slider',
                default=mem_params['qnorm_low'],
                label='QNorm low: ',
                kwargs=dict(
                    range=(0, 0.5),
                    single_step=0.01,
                    decimals=2,
                    maximum_line_edit_width=70
                )
            ),
            dict(
                id='qnorm_high',
                type='slider',
                default=mem_params['qnorm_high'],
                label='QNorm high: ',
                kwargs=dict(
                    range=(0.5, 1),
                    single_step=0.01,
                    decimals=2,
                    maximum_line_edit_width=70
                )
            )
        ]

        self.setup_ui(standard_rows=std_rows)

        def get_return_dict():
            return dict(
                shape=_to_val(self.std_rows[0].text(), dtype=int),
                halo=_to_val(self.std_rows[1].text(), dtype=int),
                padding=self.std_rows[2].currentText(),
                batch_size=_to_val(self.std_rows[3].text(), dtype=int),
                sigma=self.std_rows[4].value,
                qnorm_low=self.std_rows[5].value,
                qnorm_high=self.std_rows[6].value
            )

        if self.exec_() == QDialog.Accepted:
            return True, get_return_dict()
        else:
            return False, get_return_dict()


class QSupervoxelsDialog(QParameterDialog):

    def __init__(self):
        super(QSupervoxelsDialog, self).__init__()
        self.title = 'CebraNET parameters'

    def get_results(self, sv_params):

        def _to_str(val):
            if val is None:
                return ''
            elif type(val) is tuple or type(val) is list:
                return str.join(', ', [str(v) for v in val])
            else:
                return f'{val}'

        def _to_val(txt, dtype=float):
            if txt == '':
                return None
            elif ',' in txt:
                values = str.split(txt, ',')
                assert len(values) == 3, 'Supply either only one value or an individual value for all dimensions!'
                return [dtype(v) for v in values]
            else:
                return dtype(txt)

        std_rows = [
            dict(
                id='shape',
                type='line_edit',
                default=_to_str(sv_params['shape']),
                label='Shape: '
            ),
            dict(
                id='halo',
                type='line_edit',
                default=_to_str(sv_params['halo']),
                label='Halo: '
            ),
            dict(
                id='padding',
                type='combo_box',
                default=[['zeros', 'data'], sv_params['padding']],
                label='Padding: '
            ),
            dict(
                id='threshold',
                type='slider',
                default=sv_params['threshold'],
                label='Threshold: ',
                kwargs=dict(
                    range=(0, 1),
                    single_step=0.001,
                    decimals=3,
                    maximum_line_edit_width=70
                )
            ),
            dict(
                id='min_membrane_size',
                type='slider',
                default=sv_params['min_membrane_size'],
                label='Min membrane size: ',
                kwargs=dict(
                    range=(0, 100),
                    single_step=1,
                    maximum_line_edit_width=70
                )
            ),
            dict(
                id='sigma_dt',
                type='slider',
                default=sv_params['sigma_dt'],
                label='Sigma (DT): ',
                kwargs=dict(
                    range=(0, 10),
                    single_step=0.1,
                    decimals=1,
                    maximum_line_edit_width=70
                )
            ),
            dict(
                id='min_segment_size',
                type='slider',
                default=sv_params['min_segment_size'],
                label='Min segment size: ',
                kwargs=dict(
                    range=(0, 256),
                    single_step=1,
                    maximum_line_edit_width=70
                )
            )
        ]

        self.setup_ui(standard_rows=std_rows)

        def get_return_dict():
            return dict(
                shape=_to_val(self.std_rows[0].text(), dtype=int),
                halo=_to_val(self.std_rows[1].text(), dtype=int),
                padding=self.std_rows[2].currentText(),
                threshold=self.std_rows[3].value,
                min_membrane_size=self.std_rows[4].value,
                sigma_dt=self.std_rows[5].value,
                min_segment_size=self.std_rows[6].value
            )

        if self.exec_() == QDialog.Accepted:
            return True, get_return_dict()
        else:
            return False, get_return_dict()
