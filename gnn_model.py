
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialMessagePassing(nn.Module):
    """Aggregate information from spatial neighbors."""

    def __init__(self, hidden_dim=64, edge_dim=4):
        super().__init__()

        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, nodes, edge_index, edge_features):
        # No spatial neighbors: preserve node representations.
        if edge_index.shape[1] == 0:
            return nodes

        source = edge_index[0]
        target = edge_index[1]

        messages = self.message_net(
            torch.cat([nodes[source], edge_features], dim=1)
        )

        aggregated = torch.zeros_like(nodes)
        aggregated.index_add_(0, target, messages)

        counts = torch.zeros(
            nodes.shape[0], 1, device=nodes.device
        )
        counts.index_add_(
            0,
            target,
            torch.ones(
                target.shape[0], 1, device=nodes.device
            ),
        )

        aggregated = aggregated / counts.clamp(min=1)

        updated = self.update_net(
            torch.cat([nodes, aggregated], dim=1)
        )

        # Residual connection helps retain original node information.
        return nodes + updated


class HybridHOIGNN(nn.Module):
    """Candidate-level HOI classifier using hybrid graph context."""

    def __init__(
        self,
        num_object_categories=80,
        num_verbs=118,
        hidden_dim=64,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Category IDs are categorical, not continuous measurements.
        self.human_embedding = nn.Embedding(1, 16)
        self.object_embedding = nn.Embedding(
            num_object_categories, 16
        )
        self.verb_embedding = nn.Embedding(num_verbs, 16)

        # Four normalized bounding-box coordinates.
        self.human_encoder = nn.Sequential(
            nn.Linear(16 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.object_encoder = nn.Sequential(
            nn.Linear(16 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.human_message_passing = SpatialMessagePassing(
            hidden_dim
        )
        self.object_message_passing = SpatialMessagePassing(
            hidden_dim
        )

        # Human + object + verb embeddings + QPIC score + 4 spatial features.
        candidate_dim = hidden_dim * 2 + 16 + 1 + 4

        self.classifier = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graph):
        # Encode human nodes.
        human_features = torch.cat(
            [
                self.human_embedding(graph["human_categories"]),
                graph["human_boxes"],
            ],
            dim=1,
        )
        humans = self.human_encoder(human_features)

        # Encode object nodes.
        object_features = torch.cat(
            [
                self.object_embedding(graph["object_categories"]),
                graph["object_boxes"],
            ],
            dim=1,
        )
        objects = self.object_encoder(object_features)

        # Learn context within each node type.
        humans = self.human_message_passing(
            humans,
            graph["human_edge_index"],
            graph["human_edge_features"],
        )

        objects = self.object_message_passing(
            objects,
            graph["object_edge_index"],
            graph["object_edge_features"],
        )

        # Gather contextual features for every HOI candidate.
        candidate_humans = humans[graph["candidate_subject_index"]]
        candidate_objects = objects[graph["candidate_object_index"]]
        candidate_verbs = self.verb_embedding(
            graph["candidate_verb_ids"]
        )

        candidate_features = torch.cat(
            [
                candidate_humans,
                candidate_objects,
                candidate_verbs,
                graph["candidate_scores"].unsqueeze(1),
                graph["candidate_spatial_features"],
            ],
            dim=1,
        )

        # Return raw logits; use BCEWithLogitsLoss for training.
        return self.classifier(candidate_features).squeeze(1)


def graph_to_tensors(graph):
    """Convert one saved training graph to model-ready tensors."""

    width = float(graph["image_width"])
    height = float(graph["image_height"])

    human_ids = list(graph["humans"].keys())
    object_ids = list(graph["objects"].keys())

    human_index = {
        node_id: i for i, node_id in enumerate(human_ids)
    }
    object_index = {
        node_id: i for i, node_id in enumerate(object_ids)
    }

    def box_features(node, image_width, image_height):
        x1, y1, x2, y2 = node["bbox"]
        return [
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
        ]

    def edge_tensors(edges, index_map):
        if not edges:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 4), dtype=torch.float32),
            )

        edge_index = torch.tensor(
            [
                [index_map[e["source"]] for e in edges],
                [index_map[e["target"]] for e in edges],
            ],
            dtype=torch.long,
        )

        edge_features = torch.tensor(
            [
                [
                    e["features"]["dx"],
                    e["features"]["dy"],
                    e["features"]["distance"],
                    e["features"]["iou"],
                ]
                for e in edges
            ],
            dtype=torch.float32,
        )

        return edge_index, edge_features

    human_edge_index, human_edge_features = edge_tensors(
        graph["human_edges"], human_index
    )
    object_edge_index, object_edge_features = edge_tensors(
        graph["object_edges"], object_index
    )

    # Index pair-level spatial features by the human/object IDs.
    pair_spatial = {
        (edge["subject_id"], edge["object_id"]):
            edge["spatial_features"]
        for edge in graph["hoi_edges"]
    }

    candidates = graph["labeled_candidates"]

    subject_indices = []
    object_indices = []
    verb_ids = []
    scores = []
    spatial_features = []
    labels = []

    for candidate in candidates:
        subject_id = candidate["subject_id"]
        object_id = candidate["object_id"]

        subject_indices.append(human_index[subject_id])
        object_indices.append(object_index[object_id])
        verb_ids.append(candidate["verb_id"])
        scores.append(candidate["score"])
        labels.append(candidate["label"])

        pair_key = (subject_id, object_id)
        spatial = pair_spatial[pair_key]

        spatial_features.append(
            [
                spatial["dx"],
                spatial["dy"],
                spatial["distance"],
                spatial["iou"],
            ]
        )

    return {
        "human_categories": torch.tensor(
            [graph["humans"][i]["category"] for i in human_ids],
            dtype=torch.long,
        ),
        "human_boxes": torch.tensor(
            [
                box_features(graph["humans"][i], width, height)
                for i in human_ids
            ],
            dtype=torch.float32,
        ),
        "object_categories": torch.tensor(
            [graph["objects"][i]["category"] for i in object_ids],
            dtype=torch.long,
        ),
        "object_boxes": torch.tensor(
            [
                box_features(graph["objects"][i], width, height)
                for i in object_ids
            ],
            dtype=torch.float32,
        ),
        "human_edge_index": human_edge_index,
        "human_edge_features": human_edge_features,
        "object_edge_index": object_edge_index,
        "object_edge_features": object_edge_features,
        "candidate_subject_index": torch.tensor(
            subject_indices, dtype=torch.long
        ),
        "candidate_object_index": torch.tensor(
            object_indices, dtype=torch.long
        ),
        "candidate_verb_ids": torch.tensor(
            verb_ids, dtype=torch.long
        ),
        "candidate_scores": torch.tensor(
            scores, dtype=torch.float32
        ),
        "candidate_spatial_features": torch.tensor(
            spatial_features, dtype=torch.float32
        ),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }
