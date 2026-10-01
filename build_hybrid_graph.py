import math

import torch


STRUCTURED_FILE = (
    "structured_hoi/structured_chunk_0000.pt"
)

IMAGE_INDEX = 0

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 427

K_NEIGHBORS = 3
def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def box_center(box):
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def spatial_features(box_a, box_b):
    center_a = box_center(box_a)
    center_b = box_center(box_b)

    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    image_diagonal = math.sqrt(
        IMAGE_WIDTH ** 2
        + IMAGE_HEIGHT ** 2
    )

    return {
        "dx": dx / IMAGE_WIDTH,
        "dy": dy / IMAGE_HEIGHT,
        "distance": distance / image_diagonal,
        "iou": compute_iou(box_a, box_b),
    }


def node_features(node):
    bbox = node["bbox"]

    return [
        node["category"],
        bbox[0] / IMAGE_WIDTH,
        bbox[1] / IMAGE_HEIGHT,
        bbox[2] / IMAGE_WIDTH,
        bbox[3] / IMAGE_HEIGHT,
    ]
def hoi_edge_features(edge):
    features = [0.0] * 117

    for verb_id, verb_info in edge["verb_scores"].items():
        features[verb_id - 1] = verb_info["score"]

    return features
def build_knn_edges(nodes, k):
    node_items = list(nodes.items())
    edges = []

    for node_id, node in node_items:

        distances = []

        for other_id, other in node_items:

            if node_id == other_id:
                continue

            distance = math.sqrt(
                (
                    box_center(node["bbox"])[0]
                    - box_center(other["bbox"])[0]
                ) ** 2
                +
                (
                    box_center(node["bbox"])[1]
                    - box_center(other["bbox"])[1]
                ) ** 2
            )

            distances.append(
                (
                    distance,
                    other_id,
                    other,
                )
            )

        distances.sort(
            key=lambda item: item[0]
        )

        for distance, other_id, other in distances[:k]:

            features = spatial_features(
                node["bbox"],
                other["bbox"],
            )

            edges.append(
                {
                    "source": node_id,
                    "target": other_id,
                    "features": features,
                }
            )

    return edges
def build_graph(image):

    hois = image["hois"]

    humans = {}
    objects = {}
    pair_relations = {}

    for hoi in hois:

        subject_id = hoi["subject_id"]
        object_id = hoi["object_id"]

        if subject_id not in humans:
            humans[subject_id] = {
                "node_id": f"h_{subject_id}",
                "category": hoi["subject_category"],
                "bbox": hoi["subject_bbox"],
            }

        if object_id not in objects:
            objects[object_id] = {
                "node_id": f"o_{object_id}",
                "category": hoi["object_category"],
                "bbox": hoi["object_bbox"],
            }

        pair = (
            subject_id,
            object_id,
        )

        if pair not in pair_relations:
            pair_relations[pair] = {
                "subject_id": subject_id,
                "object_id": object_id,
                "verb_scores": {},
            }

        pair_relations[pair]["verb_scores"][
            hoi["verb_id"]
        ] = {
            "verb": hoi["verb"],
            "score": hoi["score"],
        }

    object_edges = build_knn_edges(
        objects,
        K_NEIGHBORS,
    )

    human_edges = build_knn_edges(
        humans,
        K_NEIGHBORS,
    )


    return {
        "humans": humans,
        "objects": objects,
        "hoi_edges": pair_relations,
        "object_edges": object_edges,
        "human_edges": human_edges,
    }


def main():

    data = torch.load(
        STRUCTURED_FILE,
        map_location="cpu",
        weights_only=False,
    )

    image = data["images"][IMAGE_INDEX]

    graph = build_graph(image)

    print("\nHybrid graph inspection")
    print("=" * 70)

    print(
        "Image ID:",
        image["image_id"]
    )

    print(
        "Human nodes:",
        len(graph["humans"])
    )

    print(
        "Object nodes:",
        len(graph["objects"])
    )

    print(
        "H-O edges:",
        len(graph["hoi_edges"])
    )

    print(
        "Object-object edges:",
        len(graph["object_edges"])
    )

    print(
        "Human-human edges:",
        len(graph["human_edges"])
    )

    print("\nHuman node feature example")
    print("-" * 40)

    human_id, human = next(
        iter(graph["humans"].items())
    )

    print(
        "Node ID:",
        human_id
    )

    print(
        "Raw node:",
        human
    )

    print(
        "Features:",
        node_features(human)
    )

    print("\nObject node feature example")
    print("-" * 40)

    object_id, object_node = next(
        iter(graph["objects"].items())
    )

    print(
        "Node ID:",
        object_id
    )

    print(
        "Raw node:",
        object_node
    )

    print(
        "Features:",
        node_features(object_node)
    )

    print("\nH-O edge examples")
    print("-" * 40)

    for index, edge in enumerate(
        graph["hoi_edges"].values()
    ):

        print(
            f"{edge['subject_id']} -> "
            f"{edge['object_id']}:"
        )

        for verb_id, verb_info in sorted(
            edge["verb_scores"].items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[:5]:

            print(
                f"  verb_id={verb_id}, "
                f"{verb_info['verb']}, "
                f"score={verb_info['score']:.6f}"
            )

        if index >= 4:
            break

    print("\nH-O edge feature vector")
    print("-" * 40)

    first_edge = next(
        iter(graph["hoi_edges"].values())
    )

    features = hoi_edge_features(first_edge)

    print(
        "Feature length:",
        len(features)
    )

    print(
        "Non-zero features:",
        sum(
            1
            for value in features
            if value != 0.0
        )
    )

    print(
        "Highest score:",
        max(features)
    )

    print(
        "Highest-scoring verb ID:",
        features.index(max(features)) + 1
    )

    print("\nObject-object spatial examples")
    print("-" * 40)

    for edge in graph["object_edges"][:5]:

        features = edge["features"]

        print(
            f"{edge['source']} <-> "
            f"{edge['target']}: "
            f"dx={features['dx']:.2f}, "
            f"dy={features['dy']:.2f}, "
            f"distance={features['distance']:.2f}, "
            f"IoU={features['iou']:.4f}"
        )

    print("\nHuman-human spatial examples")
    print("-" * 40)

    for edge in graph["human_edges"][:5]:

        features = edge["features"]

        print(
            f"{edge['source']} <-> "
            f"{edge['target']}: "
            f"dx={features['dx']:.2f}, "
            f"dy={features['dy']:.2f}, "
            f"distance={features['distance']:.2f}, "
            f"IoU={features['iou']:.4f}"
        )


if __name__ == "__main__":
    main()