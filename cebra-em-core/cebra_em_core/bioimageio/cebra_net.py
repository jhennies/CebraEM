
import bioimageio.core
import xarray as xr


def run_cebra_net(
        input_data
):

    # Set up the model
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

    # Predict with tiling
    result = bioimageio.core.predict_with_tiling(prediction_pipeline, input_array, tiling=tiling, verbose=True)[0]

    return (result[0, 0, :] * 255).astype('uint8')

