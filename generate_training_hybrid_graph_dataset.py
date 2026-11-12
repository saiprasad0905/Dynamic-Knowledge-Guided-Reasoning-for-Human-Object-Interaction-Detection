
import json
from pathlib import Path

import torch
from PIL import Image

from build_hybrid_graph import (
    K_NEIGHBORS,
    build_knn_edges,
    spatial_features,
)

ANNOTATIONS_FILE = (
    Path("..") / "hoi_capstone" / "data" /
    "hico_20160224_det" / "annotations" /
    "trainval_hico.json"
)

IMAGE_DIR = (
    Path("..") / "hoi_capstone" / "data" /
    "hico_20160224_det" / "images" / "train2015"
)

INPUT_DIR = Path("training_labeled_hoi")
OUTPUT_DIR = Path("training_hybrid_graph_dataset")
NUM_CHUNKS = 5


def clip_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(float(x1), float(width))),
        max(0.0, min(float(y1), float(height))),
        max(0.0, min(float(x2), float(width))),
        max(0.0, min(float(y2), float(height))),
    ]


def process_chunk(chunk_path, annotations):
    source = torch.load(
        chunk_path,
        map_location="cpu",
        weights_only=False,
    )

    graphs = []

    for image in source["images"]:
        image_id = int(image["image_id"])

        if not 0 <= image_id < len(annotations):
            raise ValueError(
                f"Invalid training image ID: {image_id}"
            )

        filename = annotations[image_id]["file_name"]
        image_path = IMAGE_DIR / filename

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as img:
            width, height = img.size

        humans = {}
        objects = {}
        pair_relations = {}
        labeled_candidates = []

        for candidate in image["hois"]:
            subject_id = candidate["subject_id"]
            object_id = candidate["object_id"]

            subject_bbox = clip_bbox(
                candidate["subject_bbox"], width, height
            )
            object_bbox = clip_bbox(
                candidate["object_bbox"], width, height
            )

            for bbox in (subject_bbox, object_bbox):
                if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                    raise ValueError(
                        f"Invalid bbox in image {image_id}: {bbox}"
                    )

            if subject_id not in humans:
                humans[subject_id] = {
                    "node_id": f"h_{subject_id}",
                    "category": candidate["subject_category"],
                    "bbox": subject_bbox,
                }

            if object_id not in objects:
                objects[object_id] = {
                    "node_id": f"o_{object_id}",
                    "category": candidate["object_category"],
                    "bbox": object_bbox,
                }

            pair = (subject_id, object_id)

            if pair not in pair_relations:
                pair_relations[pair] = {
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "verb_candidates": [],
                }

            # Keep every candidate and its label.
            candidate_record = {
                "query_id": candidate["query_id"],
                "subject_id": subject_id,
                "object_id": object_id,
                "verb_id": candidate["verb_id"],
                "verb": candidate["verb"],
                "score": float(candidate["score"]),
                "label": int(candidate["label"]),
                "subject_iou": float(candidate["subject_iou"]),
                "object_iou": float(candidate["object_iou"]),
            }

            pair_relations[pair]["verb_candidates"].append(
                candidate_record
            )
            labeled_candidates.append(candidate_record)

        human_edges = build_knn_edges(
            humans, K_NEIGHBORS, width, height
        )
        object_edges = build_knn_edges(
            objects, K_NEIGHBORS, width, height
        )

        hoi_edges = []

        for pair, relation in pair_relations.items():
            subject_id, object_id = pair

            features = spatial_features(
                humans[subject_id]["bbox"],
                objects[object_id]["bbox"],
                width,
                height,
            )

            hoi_edges.append({
                "subject_id": subject_id,
                "object_id": object_id,
                "spatial_features": features,
                "verb_candidates": relation["verb_candidates"],
            })

        graphs.append({
            "image_id": image_id,
            "filename": filename,
            "image_width": width,
            "image_height": height,
            "humans": humans,
            "objects": objects,
            "hoi_edges": hoi_edges,
            "human_edges": human_edges,
            "object_edges": object_edges,
            "labeled_candidates": labeled_candidates,
            "num_positive_candidates": sum(
                item["label"] == 1 for item in labeled_candidates
            ),
            "num_negative_candidates": sum(
                item["label"] == 0 for item in labeled_candidates
            ),
        })

    output_path = OUTPUT_DIR / chunk_path.name.replace(
        "training_labeled_chunk_",
        "training_graph_chunk_",
    )

    torch.save(
        {
            "graphs": graphs,
            "start_index": source["start_index"],
            "end_index": source["end_index"],
            "k_neighbors": K_NEIGHBORS,
        },
        output_path,
    )

    return len(graphs), output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as file:
        annotations = json.load(file)

    chunk_paths = sorted(
        INPUT_DIR.glob("training_labeled_chunk_*.pt")
    )

    if len(chunk_paths) != NUM_CHUNKS:
        raise ValueError(
            f"Expected {NUM_CHUNKS} chunks; found {len(chunk_paths)}"
        )

    total_images = 0
    total_candidates = 0
    total_positive = 0
    total_negative = 0

    for index, chunk_path in enumerate(chunk_paths, start=1):
        count, output_path = process_chunk(
            chunk_path, annotations
        )

        saved = torch.load(
            output_path,
            map_location="cpu",
            weights_only=False,
        )

        candidates = [
            candidate
            for graph in saved["graphs"]
            for candidate in graph["labeled_candidates"]
        ]

        positive = sum(c["label"] == 1 for c in candidates)
        negative = sum(c["label"] == 0 for c in candidates)

        total_images += count
        total_candidates += len(candidates)
        total_positive += positive
        total_negative += negative

        print(
            f"[{index}/{NUM_CHUNKS}] Saved {output_path}: "
            f"{count} images, {len(candidates)} candidates, "
            f"{positive} positive, {negative} negative"
        )

    print("\nTraining graph generation finished.")
    print("Images:", total_images)
    print("Candidates:", total_candidates)
    print("Positive labels:", total_positive)
    print("Negative labels:", total_negative)
    print("Output directory:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
