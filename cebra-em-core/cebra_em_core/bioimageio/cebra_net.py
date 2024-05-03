
import bioimageio.core
import xarray as xr
from cebra_em_core.misc.repo import get_repo_path
import os


def run_cebra_net(
        input_data
):

    # Set up the model
    try:
        rdf_path = os.path.join(
            get_repo_path(), 'models', 'cebranet-cellular-membranes-in-volume-sem_pytorch_state_dict.zip'
            # get_repo_path(), 'models', 'cebra_net.zip'
        )
        print(f'rdf_path = {rdf_path}')
        print(f'trying locally')
        model_resource = bioimageio.core.load_resource_description(rdf_path)
        # model_resource = bioimageio.core.load_description(rdf_path)
    except TypeError:
        print('using this')
        rdf_doi = "10.5281/zenodo.7274275"
        model_resource = bioimageio.core.load_resource_description(rdf_doi)

    # Prepare the data
    input_array = xr.DataArray(input_data[None, None, :], dims=tuple(model_resource.inputs[0].axes))

    # Set up the prediction pipeline
    devices = None
    weight_format = None
    prediction_pipeline = bioimageio.core.create_prediction_pipeline(
        model_resource, devices=devices, weight_format=weight_format
    )

    # Make sure the input and output shapes match
    prediction_pipeline.input_specs[0].shape = prediction_pipeline.output_specs[0].shape
    tiling = True

    # sample = bioimageio.core.Sample(members={0: input_array}, stat=None, id=0)

    # Predict with tiling
    result = bioimageio.core.predict_with_tiling(prediction_pipeline, input_array, tiling=tiling, verbose=True)[0]
    # result = prediction_pipeline.predict_sample_with_blocking(sample)
    return (result[0, 0, :] * 255).astype('uint8')


if __name__ == '__main__':

    from h5py import File
    with File('/media/julian/Data/tmp/gt000/raw.h5', mode='r') as f:
        data = f['data'][:]

    run_cebra_net(data)

