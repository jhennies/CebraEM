
from cebra_em_core.segmentation.stitching_utils import merge_mappings, solve_mapping
import json


if __name__ == '__main__':

    image = snakemake.params['image_name']

    print(f">>> STARTING: Solve mapping for {image}")

    inputs = snakemake.input
    output = snakemake.output[0]
    print(inputs)
    if type(inputs) == str:
        mapps = [json.load(open(inputs, mode='r'))]
    else:
        mapps = [json.load(open(inp, mode='r')) for inp in inputs]

    print('Merging mappings ...')
    m = merge_mappings(mapps, convert_items='int')
    print('Solving mappings ...')
    m = solve_mapping(m, verbose=True)

    with open(output, mode='w') as f:
        json.dump(m, f)

    print(f"<<< DONE: Solve mapping for {image}")
