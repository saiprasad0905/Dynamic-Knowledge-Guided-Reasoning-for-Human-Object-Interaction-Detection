
from pathlib import Path

import torch

GRAPH_FILE = (
    Path("training_hybrid_graph_dataset")
    / "training_graph_chunk_0000.pt"
)

data = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False,
)

graph = data["graphs"][0]

# Build stable node-index mappings for this graph.
human_ids = list(graph["humans"].keys())
object_ids = list(graph["objects"].keys())

human_index = {
    node_id: index
    for index, node_id in enumerate(human_ids)
}
object_index = {
    node_id: index
    for index, node_id in enumerate(object_ids)
}


def encode_nodes(nodes, width, height):
    features = []

    for node in nodes.values():
        x1, y1, x2, y2 = node["bbox"]

        features.append([
            float(node["category"]),
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        ])

    return torch.tensor(features, dtype=torch.float32)


human_features = encode_nodes(
    graph["humans"],
    graph["image_width"],
    graph["image_height"],
)

object_features = encode_nodes(
    graph["objects"],
    graph["image_width"],
    graph["image_height"],
)

# Each spatial edge has a source and target node.
def encode_spatial_edges(edges, index_map):
    edge_indices = []
    edge_features = []

    for edge in edges:
        edge_indices.append([
            index_map[edge["source"]],
            index_map[edge["target"]],
        ])

        features = edge["features"]

        edge_features.append([
            features["dx"],
            features["dy"],
            features["distance"],
            features["iou"],
        ])

    if edge_indices:
        indices = torch.tensor(
            edge_indices, dtype=torch.long
        ).T.contiguous()

        features = torch.tensor(
            edge_features, dtype=torch.float32
        )
    else:
        indices = torch.empty((2, 0), dtype=torch.long)
        features = torch.empty((0, 4), dtype=torch.float32)

    return indices, features


human_edge_index, human_edge_features = encode_spatial_edges(
    graph["human_edges"], human_index
)

object_edge_index, object_edge_features = encode_spatial_edges(
    graph["object_edges"], object_index
)

# Keep every verb candidate as a separate supervised example.
candidates = graph["labeled_candidates"]

verb_scores = torch.tensor(
    [
        [
            float(candidate["score"]),
            float(candidate["verb_id"]) / 117.0,
        ]
        for candidate in candidates
    ],
    dtype=torch.float32,
)

labels = torch.tensor(
    [candidate["label"] for candidate in candidates],
    dtype=torch.float32,
)

# Report shapes and basic validity checks.
print("Image ID:", graph["image_id"])
print("Human feature shape:", tuple(human_features.shape))
print("Object feature shape:", tuple(object_features.shape))
print("Human edge index shape:", tuple(human_edge_index.shape))
print("Human edge feature shape:", tuple(human_edge_features.shape))
print("Object edge index shape:", tuple(object_edge_index.shape))
print("Object edge feature shape:", tuple(object_edge_features.shape))
print("Candidate score/verb shape:", tuple(verb_scores.shape))
print("Label shape:", tuple(labels.shape))

print("Positive labels:", int((labels == 1).sum()))
print("Negative labels:", int((labels == 0).sum()))

assert human_features.shape == (len(human_ids), 5)
assert object_features.shape == (len(object_ids), 5)
assert human_edge_index.shape[0] == 2
assert object_edge_index.shape[0] == 2
assert human_edge_features.shape[1] == 4
assert object_edge_features.shape[1] == 4
assert verb_scores.shape == (len(candidates), 2)
assert labels.shape == (len(candidates),)
assert torch.isfinite(human_features).all()
assert torch.isfinite(object_features).all()
assert torch.isfinite(human_edge_features).all()
assert torch.isfinite(object_edge_features).all()
assert torch.isfinite(verb_scores).all()

print("\nPASS: Tensor conversion smoke test.")

