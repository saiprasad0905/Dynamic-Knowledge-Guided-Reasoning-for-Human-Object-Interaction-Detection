import math

import torch


STRUCTURED_FILE = (
    "structured_hoi/structured_chunk_0000.pt"
)

IMAGE_INDEX = 0

DATA_DIR = (
    "../hoi_capstone/data/hico_20160224_det"
)

IMAGE_DIR = f"{DATA_DIR}/images/test2015"
ANNOTATIONS_FILE = f"{DATA_DIR}/annotations/test_hico.json"

K_NEIGHBORS = 3

# K_NEIGHBORS = 3
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


def spatial_features(box_a, box_b, image_width, image_height):
    center_a = box_center(box_a)
    center_b = box_center(box_b)

    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    distance = math.sqrt(dx * dx + dy * dy)

    image_diagonal = math.sqrt(
        image_width ** 2 + image_height ** 2
    )

    return {
        "dx": dx / image_width,
        "dy": dy / image_height,
        "distance": distance / image_diagonal,
        "iou": compute_iou(box_a, box_b),
    }


def node_features(node, image_width, image_height):
    bbox = node["bbox"]

    return [
        node["category"],
        bbox[0] / image_width,
        bbox[1] / image_height,
        bbox[2] / image_width,
        bbox[3] / image_height,
    ]
def hoi_edge_features(edge):
    features = [0.0] * 117

    for verb_id, verb_info in edge["verb_scores"].items():
        features[verb_id - 1] = verb_info["score"]

    return features
def build_knn_edges(nodes, k, image_width, image_height):
    node_items = list(nodes.items())
    edges = []

    for node_id, node in node_items:
        distances = []

        for other_id, other in node_items:
            if node_id == other_id:
                continue

            center_a = box_center(node["bbox"])
            center_b = box_center(other["bbox"])

            distance = math.sqrt(
                (center_a[0] - center_b[0]) ** 2
                + (center_a[1] - center_b[1]) ** 2
            )

            distances.append((distance, other_id, other))

        distances.sort(key=lambda item: item[0])

        for distance, other_id, other in distances[:k]:
            features = spatial_features(
                node["bbox"],
                other["bbox"],
                image_width,
                image_height,
            )

            edges.append({
                "source": node_id,
                "target": other_id,
                "features": features,
            })

    return edges
def build_graph(image, image_width, image_height):
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

        pair = (subject_id, object_id)

        if pair not in pair_relations:
            pair_relations[pair] = {
                "subject_id": subject_id,
                "object_id": object_id,
                "verb_scores": {},
            }

        pair_relations[pair]["verb_scores"][hoi["verb_id"]] = {
            "verb": hoi["verb"],
            "score": hoi["score"],
        }

    object_edges = build_knn_edges(
        objects, K_NEIGHBORS, image_width, image_height
    )
    human_edges = build_knn_edges(
        humans, K_NEIGHBORS, image_width, image_height
    )

    return {
        "image_width": image_width,
        "image_height": image_height,
        "humans": humans,
        "objects": objects,
        "hoi_edges": pair_relations,
        "object_edges": object_edges,
        "human_edges": human_edges,
    }

def main():
    import json
    from pathlib import Path
    from PIL import Image

    data = torch.load(
        STRUCTURED_FILE,
        map_location="cpu",
        weights_only=False,
    )

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as file:
        annotations = json.load(file)

    image = data["images"][IMAGE_INDEX]
    image_id = image["image_id"]

    if not 0 <= image_id < len(annotations):
        raise ValueError(f"Invalid image ID: {image_id}")

    filename = annotations[image_id]["file_name"]
    image_path = Path(IMAGE_DIR) / filename

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as source_image:
        image_width, image_height = source_image.size

    graph = build_graph(image, image_width, image_height)

    print("\nHybrid graph inspection")
    print("=" * 70)
    print("Image ID:", image_id)
    print("Filename:", filename)
    print("Image dimensions:", image_width, "x", image_height)
    print("Human nodes:", len(graph["humans"]))
    print("Object nodes:", len(graph["objects"]))
    print("H-O edges:", len(graph["hoi_edges"]))
    print("Object-object edges:", len(graph["object_edges"]))
    print("Human-human edges:", len(graph["human_edges"]))

    if graph["humans"]:
        human_id, human = next(iter(graph["humans"].items()))
        print("\nHuman node feature example")
        print("Node ID:", human_id)
        print("Features:", node_features(human, image_width, image_height))

    if graph["objects"]:
        object_id, object_node = next(iter(graph["objects"].items()))
        print("\nObject node feature example")
        print("Node ID:", object_id)
        print("Features:", node_features(object_node, image_width, image_height))

    if graph["hoi_edges"]:
        first_edge = next(iter(graph["hoi_edges"].values()))
        features = hoi_edge_features(first_edge)
        print("\nH-O edge feature vector")
        print("Feature length:", len(features))
        print("Non-zero features:", sum(v != 0.0 for v in features))
        print("Highest score:", max(features))
        print("Highest-scoring verb ID:", features.index(max(features)) + 1)

    print("\nObject-object spatial examples")
    for edge in graph["object_edges"][:5]:
        print(edge["source"], "->", edge["target"], edge["features"])

    print("\nHuman-human spatial examples")
    for edge in graph["human_edges"][:5]:
        print(edge["source"], "->", edge["target"], edge["features"])

if __name__ == "__main__":
    main()