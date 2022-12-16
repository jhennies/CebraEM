
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats


def supervoxel_merging(mem, sv, beta=0.5):

    rag = feats.compute_rag(sv)
    costs = feats.compute_boundary_features(rag, mem)[:, 0]

    edge_sizes = feats.compute_boundary_mean_and_length(rag, mem)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=beta)

    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)

    return segmentation

