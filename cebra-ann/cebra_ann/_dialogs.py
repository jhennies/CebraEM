
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
                type = row_spec['type']
                default = row_spec['default']
                label = row_spec['label']

                lyo_this = QHBoxLayout()

                if type == 'line_edit':
                    self.std_rows.append(QLineEdit(default))

                elif type == 'combo_box':
                    cmb = QComboBox()
                    cmb.addItems(default[0])
                    cmb.setCurrentText(default[1])
                    self.std_rows.append(cmb)

                elif type == 'slider':
                    sld = QSliderLabelEdit(default, (0, 10), 1)
                    self.std_rows.append(sld)

                lyo_this.addWidget(self.std_rows[-1])

                self.layout.addRow(QLabel(f'{label}: '), lyo_this)

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
            ]

            possible types: "line_edit", "combo_box", "slider"
        :param kwargs:
        :return:
        """
        
        self.setup_ui(standard_rows=standard_rows, )


class QModifyLayerDialog(QParameterDialog):

    def __init__(self):
        super(QModifyLayerDialog, self).__init__()
        self.title = 'Modify Layer'

    def setup_ui(self, txt, cmb_list, cmb_item):

        layout = self.layout

        edit_text = txt[10:]
        self.txt = QLineEdit(edit_text)
        self.cmb = QComboBox()
        self.cmb.addItems(cmb_list)
        self.cmb.setCurrentText(cmb_item)

        layout_caption = QHBoxLayout()
        layout_caption.addWidget(QLabel(f'Change name and type of {txt}:'))
        layout_caption.setContentsMargins(0, 0, 0, 10)
        layout.addRow(layout_caption)
        layout_txt = QHBoxLayout()
        layout_txt.addWidget(QLabel('semantics_'))
        layout_txt.addWidget(self.txt)
        layout.addRow(layout_txt)
        layout_cmb = QHBoxLayout()
        layout_cmb.addWidget(self.cmb)
        layout_cmb.setContentsMargins(0, 0, 0, 10)
        layout.addRow(layout_cmb)
        
        super(QModifyLayerDialog, self).setup_ui()

    def get_results(self, txt, cmb_list, cmb_item):

        self.setup_ui(txt, cmb_list, cmb_item)

        if self.exec_() == QDialog.Accepted:
            return f'semantics_{self.txt.text()}', self.cmb.currentText()
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
                return str.join(', ', val)
            else:
                return f'{val}'

        std_rows = [
            dict(
                id='shape',
                type='line_edit',
                default=_to_str(mem_params['shape']),
                label='Shape'
            ),
            dict(
                id='halo',
                type='line_edit',
                default=_to_str(mem_params['halo']),
                label='Halo'
            ),
            dict(
                id='batch_size',
                type='line_edit',
                default=_to_str(mem_params['batch_size']),
                label='Batch size'
            ),
            dict(
                id='sigma',
                type='slider',
                default=mem_params['sigma'],
                label='Sigma'
            ),
            dict(
                id='qnorm_low',
                type='slider',
                default=mem_params['qnorm_low'],
                label='QNorm low'
            ),
            dict(
                id='qnorm_high',
                type='slider',
                default=mem_params['qnorm_high'],
                label='QNorm high'
            )
        ]

        self.setup_ui(standard_rows=std_rows)

        if self.exec_() == QDialog.Accepted:
            return 1
        else:
            return None


class QCebraNetDialogOld(QParameterDialog):

    def __init__(self):
        super(QCebraNetDialog, self).__init__()
        self.title = 'CebraNET parameters'

    def setup_ui(self, mem_params):

        def _to_str(val):
            if val is None:
                return ''
            elif type(val) is tuple or type(val) is list:
                return str.join(', ', val)
            else:
                return f'{val}'

        layout = self.layout

        shape = mem_params['shape']
        halo = mem_params['halo']
        batch_size = mem_params['batch_size']
        sigma = mem_params['sigma']
        qnorm_low = mem_params['qnorm_low']
        qnorm_high = mem_params['qnorm_high']

        # Layouts for each row
        lyo_shape = QHBoxLayout()
        lyo_halo = QHBoxLayout()
        lyo_batch_size = QHBoxLayout()
        lyo_sigma = QHBoxLayout()
        lyo_qnorm_low = QHBoxLayout()
        lyo_qnorm_high = QHBoxLayout()

        # Row shape
        self.lne_shape = QLineEdit(_to_str(shape))
        lyo_shape.addWidget(self.lne_shape)
        # Row halo
        self.lne_halo = QLineEdit(_to_str(halo))
        lyo_halo.addWidget(self.lne_halo)
        # Row batch size
        self.lne_batch_size = QLineEdit(_to_str(batch_size))
        lyo_batch_size.addWidget(self.lne_batch_size)
        # Row sigma
        self.sld_sigma = QSlider(Qt.Horizontal)
        self.sld_sigma.setRange(0, 100)
        self.sld_sigma.setSingleStep(1)
        self.sld_sigma.setValue(0)
        self.lne_sigma = QLineEdit(_to_str(sigma))
        self.lne_sigma.setMaximumWidth(70)
        self.lne_sigma.setAlignment(Qt.AlignRight)
        lyo_sigma.addWidget(self.sld_sigma)
        lyo_sigma.addWidget(self.lne_sigma)
        # Row qnorm low
        self.sld_qnorm_low = QSlider(Qt.Horizontal)
        self.sld_qnorm_low.setRange(0, 50)
        self.sld_qnorm_low.setSingleStep(1)
        self.sld_qnorm_low.setValue(0)
        self.lne_qnorm_low = QLineEdit(_to_str(qnorm_low))
        self.lne_qnorm_low.setMaximumWidth(70)
        self.lne_qnorm_low.setAlignment(Qt.AlignRight)
        lyo_qnorm_low.addWidget(self.sld_qnorm_low)
        lyo_qnorm_low.addWidget(self.lne_qnorm_low)
        # Row qnorm high
        self.sld_qnorm_high = QSlider(Qt.Horizontal)
        self.sld_qnorm_high.setRange(50, 100)
        self.sld_qnorm_high.setSingleStep(1)
        self.sld_qnorm_high.setValue(0)
        self.lne_qnorm_high = QLineEdit(_to_str(qnorm_high))
        self.lne_qnorm_high.setMaximumWidth(70)
        self.lne_qnorm_high.setAlignment(Qt.AlignRight)
        lyo_qnorm_high.addWidget(self.sld_qnorm_high)
        lyo_qnorm_high.addWidget(self.lne_qnorm_high)

        # Add the rows
        layout.addRow(QLabel('Shape: '), lyo_shape)
        layout.addRow(QLabel('Halo: '), lyo_halo)
        layout.addRow(QLabel('Batch size: '), lyo_batch_size)
        layout.addRow(QLabel('Sigma: '), lyo_sigma)
        layout.addRow(QLabel('QNorm low: '), lyo_qnorm_low)
        layout.addRow(QLabel('QNorm high: '), lyo_qnorm_high)
        
        super(QCebraNetDialog, self).setup_ui()

    def get_results(self, mem_params):

        self.setup_ui(mem_params)

        if self.exec_() == QDialog.Accepted:
            return 1
        else:
            return None
