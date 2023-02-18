
import numpy as np

def quantile_norm(volume, qlow, qhigh):

    dtype = volume.dtype
    assert dtype == 'uint8', 'Only unsigned 8bit is implemented'
    volume = volume.astype('float64')

    # Get quantiles of full volume
    # Could potentially also be a reference slice, multiple reference slices, ...
    q_lower_ref = np.quantile(volume, qlow)
    q_upper_ref = np.quantile(volume, qhigh)

    volume -= q_lower_ref
    volume /= q_upper_ref - q_lower_ref
    volume *= 255

    # Clip everything that went out of range
    # FIXME this assumes dtype==uint8
    volume[volume < 0] = 0
    volume[volume > 255] = 255

    # Convert back to the original dtype
    return volume.astype(dtype)

