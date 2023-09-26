
if __name__ == '__main__':

    # ___________________________________________________
    # Command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description='Converts a dataset to the Big Data Viewer format\n'
                    '  - Optional run connected component analysis on a segmentation map.\n'
                    '  - Optional size filtering of detected objects\n'
                    '  - The result is relabeled consecutively\n'
                    '  - The output is a bdv dataset (h5 or n5)'
                    '  !! Note that the full dataset is loaded into memory !!',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('source_path', type=str,
                        help="Path of the dataset (tiff slices, h5, n5 or MIB's model)")
    parser.add_argument('target_path', type=str,
                        help="Path where the result is saved. Include file name.")
    parser.add_argument('-k', '--key', type=str, default=None,
                        help="Dataset key within h5 or n5 file; "
                             "Default=None")
    parser.add_argument('-r', '--resolution', type=float, nargs=3, metavar=('z', 'y', 'x'), default=(0.01, 0.01, 0.01),
                        help="Used for the big data viewer format to specify the resolution; "
                             "Default=(0.01, 0.01, 0.01)")
    parser.add_argument('-u', '--unit', type=str, default='micrometer',
                        help="Unit of resolution;"
                             "Default='micrometer'")
    parser.add_argument('-cv', '--clip_values', type=int, nargs=2, default=None,
                        help='Clips grey values, specify lower and upper bound; '
                             'Default=None')
    parser.add_argument('-i', '--invert', action='store_true',
                        help='Inverts the grey values')
    parser.add_argument('--roi', type=int, nargs=6, metavar=('x', 'y', 'z', 'w', 'h', 'd'), default=None,
                        help="Define the ROI according to Fiji's coordinate system; "
                             "Default=None")
    parser.add_argument('-scf', '--bdv_scale_factors', type=int, nargs='+', default=(2, 2, 4),
                        help="Scales computed for the big data viewer format; "
                             "Default=(2, 2, 4)")
    parser.add_argument('-scm', '--scale_mode', type=str, default='mean',
                        help="BDV scaling method, e.g. ['nearest', 'mean', 'interpolate']; "
                             "Default='mean'")
    parser.add_argument('-cc', '--connected_components', action='store_true',
                        help='Performs a connected component operation on a label map before converting to BDV format')
    parser.add_argument('-sf', '--size_filter', type=int, default=0,
                        help='Applies a size filter after running the connected component analysis; '
                             'int pixels; Default=0 (no size filter)')
    parser.add_argument('-ax', '--axes_order', type=str, default='zyx',
                        help="Order of the axes when reading an h5 or n5 container. Default='zyx'")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Additional console output (for debugging purposes)")

    args = parser.parse_args()
    source_path = args.source_path
    target_path = args.target_path
    key = args.key
    resolution = args.resolution
    unit = args.unit
    clip_values = args.clip_values
    invert = args.invert
    roi = args.roi
    bdv_scale_factors = args.bdv_scale_factors
    scale_mode = args.scale_mode
    connected_components = args.connected_components
    size_filter = args.size_filter
    axes_order = args.axes_order
    verbose = args.verbose
    # ___________________________________________________

    from cebra_em_core.dataset.preprocessing import convert_to_bdv

    convert_to_bdv(
        source_path,
        target_path,
        key=key,
        resolution=resolution,
        unit=unit,
        clip_values=clip_values,
        invert=invert,
        roi=roi,
        bdv_scale_factors=bdv_scale_factors,
        scale_mode=scale_mode,
        connected_components=connected_components,
        size_filter=size_filter,
        axes_order=axes_order,
        verbose=verbose
    )
