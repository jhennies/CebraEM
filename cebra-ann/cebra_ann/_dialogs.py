
from PyQt5.QtCore import *
from qtpy.QtWidgets import *


class QParameterDialog(QDialog):

    def __init__(self, title='Dialog'):

        super(QParameterDialog, self).__init__()
        self.layout = QFormLayout()
        self.title = title
        
    def setup_ui(self, *args, **kwargs):
        
        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)

        self.setWindowTitle(self.title)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
    def get_results(self, *args, **kwargs):
        
        self.setup_ui()


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

    def setup_ui(self):

        layout = self.layout

        # Layouts for each row
        lyo_shape = QHBoxLayout()
        lyo_halo = QHBoxLayout()
        lyo_batch_size = QHBoxLayout()
        lyo_sigma = QHBoxLayout()
        lyo_qnorm_low = QHBoxLayout()
        lyo_qnorm_high = QHBoxLayout()

        # Row shape
        self.lne_shape = QLineEdit()
        lyo_shape.addWidget(self.lne_shape)
        # Row halo
        self.lne_halo = QLineEdit()
        lyo_halo.addWidget(self.lne_halo)
        # Row batch size
        self.lne_batch_size = QLineEdit()
        lyo_batch_size.addWidget(self.lne_batch_size)
        # Row sigma
        self.sld_sigma = QSlider(Qt.Horizontal)
        self.sld_sigma.setRange(0, 100)
        self.sld_sigma.setSingleStep(1)
        self.sld_sigma.setValue(0)
        self.lne_sigma = QLineEdit()
        lyo_sigma.addWidget(self.sld_sigma)
        lyo_sigma.addWidget(self.lne_sigma)
        # Row qnorm low
        self.sld_qnorm_low = QSlider(Qt.Horizontal)
        self.sld_qnorm_low.setRange(0, 50)
        self.sld_qnorm_low.setSingleStep(1)
        self.sld_qnorm_low.setValue(0)
        self.lne_qnorm_low = QLineEdit()
        lyo_qnorm_low.addWidget(self.sld_qnorm_low)
        lyo_qnorm_low.addWidget(self.lne_qnorm_low)
        # Row qnorm high
        self.sld_qnorm_high = QSlider(Qt.Horizontal)
        self.sld_qnorm_high.setRange(50, 100)
        self.sld_qnorm_high.setSingleStep(1)
        self.sld_qnorm_high.setValue(0)
        self.lne_qnorm_high = QLineEdit()
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

    def get_results(self):

        self.setup_ui()

        if self.exec_() == QDialog.Accepted:
            return 1
        else:
            return None
