
import numpy as np
from vigra import filters, analysis


def _cdist(xy1, xy2):
    # influenced by: http://stackoverflow.com/a/1871630
    d = np.zeros((xy1.shape[1], xy1.shape[0], xy1.shape[0]))
    for i in np.arange(xy1.shape[1]):
        d[i, :, :] = np.square(np.subtract.outer(xy1[:, i], xy2[:, i]))
    d = np.sum(d, axis=0)
    return np.sqrt(d)


def _findBestSeedCloserThanMembrane(seeds, distances, distanceTrafo, membraneDistance):
    """ finds the best seed of the given seeds, that is the seed with the highest value distance transformation."""
    closeSeeds = distances <= membraneDistance
    # np.zeros_like(closeSeeds)
    # iterate over all close seeds
    maximumDistance = -np.inf
    mostCentralSeed = None
    for seed in seeds[closeSeeds]:
        if distanceTrafo[seed[0], seed[1], seed[2]] > maximumDistance:
            maximumDistance = distanceTrafo[seed[0], seed[1], seed[2]]
            mostCentralSeed = seed
    return mostCentralSeed


def _nonMaximumSuppressionSeeds(seeds, distanceTrafo):
    """ removes all seeds that have a neigbour that is closer than the the next membrane

    seeds is a list of all seeds, distanceTrafo is array-like
    return is a list of all seeds that are relevant.

    works only for 3d
    """
    seedsCleaned = set()

    # calculate the distances from each seed to the next seeds.
    distances = _cdist(seeds, seeds)
    for i in np.arange(len(seeds)):
        membraneDistance = distanceTrafo[seeds[i, 0], seeds[i, 1], seeds[i, 2]]
        bestAlternative = _findBestSeedCloserThanMembrane(seeds, distances[i, :], distanceTrafo, membraneDistance)
        seedsCleaned.add(tuple(bestAlternative))
    return np.array(list(seedsCleaned))


def _volumeToListOfPoints(seedsVolume, threshold=0.):
    return np.array(np.where(seedsVolume > threshold)).transpose()


def _placePointsInVolumen(points, shape):
    volumen = np.zeros(shape)
    points = np.maximum(points, np.array((0, 0, 0)))
    points = np.minimum(points, np.array(shape) - 1)
    for point in (np.floor(points)).astype(int):
        volumen[point[0], point[1], point[2]] = 1
    return volumen


def watershed_dt_with_probs(
        pmap,
        threshold=0.001,
        min_membrane_size=1,
        anisotropy=(1, 1, 1),
        sigma_dt=1.0,
        min_segment_size=48,
        clean_close_seeds=True,
        return_intermediates=False,
        verbose=False
):

    pmap = pmap.astype('float32')
    pmap /= pmap.max()

    if verbose:
        print('pmin = {}'.format(threshold))

    mask = pmap >= threshold

    if verbose:
        print('delete small CCs')
    # delete small CCs
    mask = analysis.labelVolumeWithBackground(mask.astype('uint32'))
    analysis.sizeFilterSegInplace(mask, int(np.max(mask)), int(min_membrane_size), checkAtBorder=True)

    # print('use cleaned binary image as mask')
    # use cleaned binary image as mask
    mask = mask > 0

    if verbose:
        print('Distance transform')
    # Distance transform
    dt = filters.distanceTransform(mask.astype('uint32'), pixel_pitch=anisotropy).astype('float16')
    dt_signed = filters.distanceTransform(mask.astype('uint32'), pixel_pitch=anisotropy, background=False).astype('float16')
    dt_signed[dt_signed > 0] -= 1
    dt_signed = dt.max() - dt + dt_signed

    if verbose:
        print('Do the smoothings')
    # Do the smoothings
    anisotropy = np.array(anisotropy)
    sigma_dt = sigma_dt / anisotropy * anisotropy.min()
    dt_signed = filters.gaussianSmoothing(dt_signed.astype('float32'), sigma_dt)

    dt_signed[mask] = 0
    dt_signed /= dt_signed.max()
    dt_signed[mask] = pmap[mask] + 1
    del pmap

    seeds = analysis.localMinima3D(dt_signed, neighborhood=26, allowAtBorder=True)

    if np.max(seeds) == 0:
        print('No seeds found ...')
        if return_intermediates:
            return np.zeros(seeds.shape, dtype=seeds.dtype), np.zeros(seeds.shape, dtype=seeds.dtype)
        else:
            return np.zeros(seeds.shape, dtype=seeds.dtype)

    if verbose:
        print('clean_close_seeds')
    if clean_close_seeds:
        seeds = _nonMaximumSuppressionSeeds(_volumeToListOfPoints(seeds), dt.astype('float32'))
        seeds = _placePointsInVolumen(seeds, dt.shape).astype(np.uint32)
    del dt

    if verbose:
        print('labelVolumeWithBackground')
    segmentation = seeds
    segmentation = analysis.labelVolumeWithBackground(segmentation)
    segmentation = analysis.watershedsNew(dt_signed, seeds=segmentation, neighborhood=26)[0]

    if verbose:
        print('sizeFilterSegInplace')
    segmentation = analysis.sizeFilterSegInplace(segmentation, int(np.max(segmentation)), int(min_segment_size),
                                                 checkAtBorder=True)

    if verbose:
        print('watershedsNew')
    segmentation = analysis.watershedsNew(dt_signed, seeds=segmentation, neighborhood=26)[0]

    if return_intermediates:
        return segmentation, [mask.astype('uint8'), dt_signed]
    else:
        return segmentation
