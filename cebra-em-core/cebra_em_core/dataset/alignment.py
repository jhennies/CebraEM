
import numpy as np
from scipy.ndimage import shift
from vigra.filters import gaussianSmoothing
from skimage import filters
from skimage.registration import phase_cross_correlation
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d


def xcorr(offset_image, image):
    image = gaussianSmoothing(image, 1)
    offset_image = gaussianSmoothing(offset_image, 1)
    image = filters.sobel(image)
    offset_image = filters.sobel(offset_image)
    shift, error, diffphase = phase_cross_correlation(image, offset_image, upsample_factor=100)
    return shift[1], shift[0]


def xcorr_on_pair(fixed, moving, verbose=False):
    h, w = fixed.shape
    h2, w2 = moving.shape
    if w != w2:
        print('Images with different widths!!')
        return 0., 0.
    offset = xcorr(fixed, moving)
    if verbose:
        print('offset = {}'.format(offset))
    return offset


def displace_slice(image, offset):

    image = shift(image, -np.round([offset[1], offset[0]]))
    return image


def xcorr_on_volume(vol, median_radius=3):

    # from h5py import File
    #
    # with File('/scratch/hennies/tmp/vol0.h5', mode='w') as f:
    #     f.create_dataset('data', data=vol)

    # Compute offset
    offsets = [np.array([0., 0.])]  # Offset for the initial slice
    for idx in range(len(vol) - 1):
        offsets.append(
            offsets[-1] + np.array(xcorr_on_pair(vol[idx, :], vol[idx + 1, :]))
        )
    offsets = np.array(offsets)

    # Smooth the offsets to get the running average
    avg_x = offsets[:, 0]
    avg_y = offsets[:, 1]
    avg_x = medfilt(avg_x, median_radius * 2 + 1)
    avg_x = gaussian_filter1d(avg_x, median_radius * 2 + 1)
    avg_y = medfilt(avg_y, median_radius * 2 + 1)
    avg_y = gaussian_filter1d(avg_y, median_radius * 2 + 1)
    avg = np.concatenate((avg_x[:, None], avg_y[:, None]), axis=1)
    assert avg.shape == offsets.shape
    # print(f'avg.shape = {avg.shape}, offsets.shape = {offsets.shape}')

    # Apply the running average
    offsets = offsets - avg
    assert len(offsets) == len(vol)

    # print(f'offsets = {offsets}')

    # with File('/scratch/hennies/tmp/vol1.h5', mode='w') as f:
    #     f.create_dataset('data', data=vol)

    # Apply the offsets to the volume
    aligned = np.zeros(vol.shape, dtype=vol.dtype)
    for idx, im in enumerate(vol):
        aligned[idx, :] = displace_slice(im.copy(), offsets[idx])

    # with File('/scratch/hennies/tmp/vol2.h5', mode='w') as f:
    #     f.create_dataset('data', data=vol)
    # with File('/scratch/hennies/tmp/aligned.h5', mode='w') as f:
    #     f.create_dataset('data', data=aligned)

    return aligned
