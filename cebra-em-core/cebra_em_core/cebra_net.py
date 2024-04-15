
# TODO This is deprecated, using the bioimageio.core functionality now
#   Currently this is still used in CebraANN though

try:
    import torch as t
    from cebra_em_core.deep_models.run_models2 import predict_model_from_h5
    from cebra_em_core.deep_models.piled_unets import PiledUnet
except ModuleNotFoundError:
    t = None

import os


default_model_path = os.path.join(os.path.dirname(__file__), '..', 'models.bkp', 'default_weights.pyt')


def run_cebra_net(
        raw_channels,
        model_filepath,
        target_filepath,
        target_size=(128, 128, 128),
        overlap=(32, 32, 32),
        squeeze_result=False,
        verbose=False
):

    model = PiledUnet(
        n_nets=3,
        in_channels=1,
        out_channels=[1, 1, 1],
        filter_sizes_down=(
            ((8, 16), (16, 32), (32, 64)),
            ((8, 16), (16, 32), (32, 64)),
            ((8, 16), (16, 32), (32, 64))
        ),
        filter_sizes_bottleneck=(
            (64, 128),
            (64, 128),
            (64, 128)
        ),
        filter_sizes_up=(
            ((64, 64), (32, 32), (16, 16)),
            ((64, 64), (32, 32), (16, 16)),
            ((64, 64), (32, 32), (16, 16))
        ),
        batch_norm=True,
        output_activation='sigmoid',
        predict=True
    )

    if t.cuda.is_available():
        model.cuda()
        model.load_state_dict(t.load(model_filepath))
    else:
        model.load_state_dict(t.load(model_filepath, map_location=t.device('cpu')))

    mask = None

    with t.no_grad():
        model.eval()

        predict_model_from_h5(
            model=model,
            results_filepath=target_filepath,
            raw_channels=raw_channels,
            num_result_channels=1,
            # target_size=(64, 64, 64),
            target_size=target_size,
            overlap=overlap,
            scale=1,
            compute_empty_volumes=False,
            thresh=(16, 239),
            use_compute_map=False,
            mask=mask,
            squeeze_result=squeeze_result,
            verbose=verbose
        )
