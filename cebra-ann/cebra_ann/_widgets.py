
from PyQt5.QtCore import *
from qtpy.QtWidgets import *


class QSliderLabelEdit(QWidget):
    
    def __init__(
            self, value, range, single_step,
            decimals=None,
            maximum_line_edit_width=None
    ):

        super(QSliderLabelEdit, self).__init__()

        # self.value = value
        self.value = value
        self.range = range
        self.single_step = single_step
        self.decimals = decimals

        layout = QFormLayout()
        self.setLayout(layout)

        self.sld = QSlider(Qt.Horizontal)
        self.sld.setRange(self._dec2int(range[0]), self._dec2int(range[1]))
        self.sld.setSingleStep(self._dec2int(single_step))
        self.sld.setValue(self._dec2int(value))
        self.sld.sliderMoved.connect(self._sld_slider_moved)

        layout.addWidget(self.sld)

        self.lne = QLineEdit()
        if maximum_line_edit_width is not None:
            self.lne.setMaximumWidth(maximum_line_edit_width)
        self._set_text(value)
        self.lne.setAlignment(Qt.AlignRight)
        self.lne.editingFinished.connect(self._lne_editing_finished)

        hlyo = QHBoxLayout()
        hlyo.addWidget(self.sld)
        hlyo.addWidget(self.lne)

        layout.addRow(hlyo)
        # layout.addLayout(hlyo)

        layout.setContentsMargins(0, 0, 0, 0)

    def _int2dec(self, value):
        if self.decimals is not None and self.decimals > 0:
            return float(value) / (10 ** self.decimals)
        else:
            return int(value)

    def _dec2int(self, value):
        if self.decimals is not None and self.decimals > 0:
            return value * 10 ** self.decimals
        else:
            return int(value)

    def _set_text(self, value):
        format_pattern = '{}' if (self.decimals is None or self.decimals == 0) else '{' + f':.{self.decimals}f' + '}'
        self.lne.setText(format_pattern.format(value))

    def _sld_slider_moved(self, value: int):
        self.value = self._int2dec(value)
        self._set_text(self.value)

    def _lne_editing_finished(self):
        if self.decimals is not None and self.decimals > 0:
            value = float(self.lne.text())
        else:
            value = int(self.lne.text())
        if value > self.range[1]:
            value = self.range[1]
        if value < self.range[0]:
            value = self.range[0]
        self._set_text(value)  # To make sure the format is right
        self.sld.setValue(self._dec2int(value))
        self.value = value

    def setValue(self, value):
        self.sld.setValue(self._dec2int(value))
        self._set_text(value)



