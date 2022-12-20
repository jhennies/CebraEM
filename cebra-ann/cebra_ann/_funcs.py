
import numpy as np
from skimage.morphology import disk


def assert_3d(val):
    if type(val) is not tuple and type(val) is not list:
        val = [val] * 3
    return val


def get_disk_positions(brush_size, center, shape, dims_order):

    shape = np.array(shape)
    z_axis = np.where(np.array(dims_order) == 0)[0][0]
    strel = np.transpose(disk(brush_size / 2)[None, :], axes=dims_order)

    if dims_order == (2, 0, 1) or dims_order == (1, 2, 0):

        # I don't get it why I need to do the swapaxes in these cases ...
        strel = strel.swapaxes(1, 2)

        # ... and also swap z_axis 2 and 1 here! It just works like this -.-
        z, y, x = np.mgrid[
            np.s_[int(center[0] - brush_size / 2):int(center[0] + brush_size / 2 + 1)] if z_axis != 0 else np.s_[center[0]: center[0] + 1],
            np.s_[int(center[1] - brush_size / 2):int(center[1] + brush_size / 2 + 1)] if z_axis != 2 else np.s_[center[1]: center[1] + 1],
            np.s_[int(center[2] - brush_size / 2):int(center[2] + brush_size / 2 + 1)] if z_axis != 1 else np.s_[center[2]: center[2] + 1]
        ]

    else:
        z, y, x = np.mgrid[
            np.s_[int(center[0] - brush_size / 2):int(center[0] + brush_size / 2 + 1)] if z_axis != 0 else np.s_[center[0]: center[0] + 1],
            np.s_[int(center[1] - brush_size / 2):int(center[1] + brush_size / 2 + 1)] if z_axis != 1 else np.s_[center[1]: center[1] + 1],
            np.s_[int(center[2] - brush_size / 2):int(center[2] + brush_size / 2 + 1)] if z_axis != 2 else np.s_[center[2]: center[2] + 1]
        ]

    # Mask with the structuring element to get the list of positions
    pos = np.array([z[strel > 0], y[strel > 0], x[strel > 0]]).swapaxes(0, 1)

    # Remove positions that are out of upper bounds
    pos = pos[(pos < shape).all(axis=1), :]
    # Remove positions that are out of lower bounds
    pos = pos[(pos >= 0).all(axis=1), :]

    return pos
