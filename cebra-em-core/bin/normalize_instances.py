
if __name__ == '__main__':

    # ___________________________________________________
    # Command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description='Quantile normalization of the raw data based on a instance segmentation',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('raw_source', type=str,
                        help="Path of the raw dataset (h5 or n5)")
    parser.add_argument('raw_key', type=str,
                        help="Key within raw dataset")
    parser.add_argument('seg_source', type=str,
                        help="Path of the instance segmentation (h5 or n5); \n"
                             "Note that the segmentation is loaded entirely into memory! "
                             "It can (and should) have a lower resolution compared to the raw data.")
    parser.add_argument('seg_key', type=str,
                        help="Key within segmentation dataset")
    parser.add_argument('target_path', type=str,
                        help="Path where the result is saved. Include file name (n5 or h5).")
    parser.add_argument('-rr', '--raw_resolution', type=float, nargs=3, metavar=('z', 'y', 'x'), default=(0.01, 0.01, 0.01),
                        help="Resolution of the raw dataset; "
                             "Default=(0.01, 0.01, 0.01)")
    parser.add_argument('-sr', '--seg_resolution', type=float, nargs=3, metavar=('z', 'y', 'x'), default=(0.01, 0.01, 0.01),
                        help="Resolution of the segmentation dataset; "
                             "Default=(0.01, 0.01, 0.01)")
    parser.add_argument('-u', '--unit', type=str, default='micrometer',
                        help="Unit of resolutions, note that raw and seg resolutions have to be given in the same unit;"
                             "Default='micrometer'")
    parser.add_argument('-ids', '--instance_ids', type=int, nargs='+', default=None,
                        help="Ids of the instances in the segmentation that will be used, "
                             "where 0 is reserved for background; \n"
                             "Default=None (all present instances are used)")
    parser.add_argument('-q', '--quantiles', type=float, nargs=2, default=(0.1, 0.9),
                        help="Quantiles of the greyscale histogram used to anchor the distribution; "
                             "Default=(0.1, 0.9)")
    parser.add_argument('-ap', '--anchor_values', type=float, nargs=2, default=(0.05, 0.95),
                        help="Which grey values to anchor the quantiles to, specified in percent of the data range; "
                             "Default=(0.05, 0.95) yields (13, 242) for uint8 output")
    parser.add_argument('-tpe', '--dtype', type=str, default='uint8',
                        help="Data type of the result dataset. Currently only supporting 'uint8'")
    parser.add_argument('-scf', '--bdv_scale_factors', type=int, nargs='+', default=(2, 2, 4),
                        help="Scales computed for the big data viewer format; "
                             "Default=(2, 2, 4)")
    parser.add_argument('-scm', '--scale_mode', type=str, default='mean',
                        help="BDV scaling method, e.g. ['nearest', 'mean', 'interpolate']; "
                             "Default='mean'")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Additional console output (for debugging purposes)")

    args = parser.parse_args()
    raw_source = args.raw_source
    raw_key = args.raw_key
    seg_source = args.seg_source
    seg_key = args.seg_key
    target_path = args.target_path
    raw_resolution = args.raw_resolution
    seg_resolution = args.seg_resolution
    unit = args.unit
    instance_ids = args.instance_ids
    quantiles = args.quantiles
    anchor_values = args.anchor_values
    dtype = args.dtype
    bdv_scale_factors = args.bdv_scale_factors
    scale_mode = args.scale_mode
    verbose = args.verbose
    # ___________________________________________________

    from cebra_em_core.dataset.preprocessing import normalize_instances
    normalize_instances(
        raw_source,
        raw_key,
        seg_source,
        seg_key,
        target_path,
        raw_resolution=raw_resolution,
        seg_resolution=seg_resolution,
        unit=unit,
        instance_ids=instance_ids,
        quantiles=quantiles,
        anchor_values=anchor_values,
        dtype=dtype,
        bdv_scale_factors=bdv_scale_factors,
        scale_mode=scale_mode,
        verbose=verbose
    )
