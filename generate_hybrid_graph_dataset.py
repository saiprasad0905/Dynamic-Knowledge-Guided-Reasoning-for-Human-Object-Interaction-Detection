import json
from pathlib import Path

import torch
from PIL import Image

from build_hybrid_graph import (
    ANNOTATIONS_FILE,
    IMAGE_DIR,
    K_NEIGHBORS,
    build_graph,
    build_knn_edges,
)

STRUCTURED_DIR = Path("structured_hoi")
OUTPUT_DIR = Path("hybrid_graph_dataset")
NUM_CHUNKS = 20

def clip_bbox(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox

    return [
        max(0.0, min(float(x1), float(image_width))),
        max(0.0, min(float(y1), float(image_height))),
        max(0.0, min(float(x2), float(image_width))),
        max(0.0, min(float(y2), float(image_height))),
    ]

def validate_graph(graph, image_width, image_height):
    humans = graph["humans"]
    objects = graph["objects"]

    for node in list(humans.values()) + list(objects.values()):
        x1, y1, x2, y2 = node["bbox"]

        if not (0 <= x1 < x2 <= image_width):
            raise ValueError(f"Invalid horizontal bbox: {node['bbox']}")

        if not (0 <= y1 < y2 <= image_height):
            raise ValueError(f"Invalid vertical bbox: {node['bbox']}")

    for edge in graph["hoi_edges"].values():
        if edge["subject_id"] not in humans:
            raise ValueError("H-O edge references missing human node")
        if edge["object_id"] not in objects:
            raise ValueError("H-O edge references missing object node")

        for verb_id in edge["verb_scores"]:
            if not 1 <= int(verb_id) <= 117:
                raise ValueError(f"Invalid verb ID: {verb_id}")

    for node_type in ("human_edges", "object_edges"):
        nodes = humans if node_type == "human_edges" else objects

        for edge in graph[node_type]:
            if edge["source"] not in nodes or edge["target"] not in nodes:
                raise ValueError(f"{node_type} references a missing node")


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
            raise ValueError(f"Image ID outside annotation range: {image_id}")

        filename = annotations[image_id]["file_name"]
        image_path = Path(IMAGE_DIR) / filename

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as img:
            width, height = img.size

        graph = build_graph(image, width, height)

        for node in graph["humans"].values():
            node["bbox"] = clip_bbox(
                node["bbox"], width, height
            )

        for node in graph["objects"].values():
            node["bbox"] = clip_bbox(
                node["bbox"], width, height
            )

        graph["human_edges"] = build_knn_edges(
            graph["humans"], K_NEIGHBORS, width, height
        )

        graph["object_edges"] = build_knn_edges(
            graph["objects"], K_NEIGHBORS, width, height
        )

        validate_graph(graph, width, height)

        graph_record = {
            "image_id": image_id,
            "filename": filename,
            "image_width": width,
            "image_height": height,
            "num_human_nodes": len(graph["humans"]),
            "num_object_nodes": len(graph["objects"]),
            "humans": graph["humans"],
            "objects": graph["objects"],
            "hoi_edges": graph["hoi_edges"],
            "human_edges": graph["human_edges"],
            "object_edges": graph["object_edges"],
        }

        graphs.append(graph_record)

    output_path = OUTPUT_DIR / chunk_path.name.replace(
        "structured_chunk_", "graph_chunk_"
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

    chunk_paths = sorted(STRUCTURED_DIR.glob("structured_chunk_*.pt"))

    if len(chunk_paths) != NUM_CHUNKS:
        raise ValueError(
            f"Expected {NUM_CHUNKS} structured chunks; found {len(chunk_paths)}"
        )

    total_images = 0

    for index, chunk_path in enumerate(chunk_paths, start=1):
        output_path = OUTPUT_DIR / chunk_path.name.replace(
            "structured_chunk_", "graph_chunk_"
        )

        if output_path.exists():
            existing = torch.load(
                output_path,
                map_location="cpu",
                weights_only=False,
            )
            count = len(existing["graphs"])
            print(f"[{index}/{NUM_CHUNKS}] Reusing {output_path}: {count} images")
        else:
            count, output_path = process_chunk(chunk_path, annotations)
            print(f"[{index}/{NUM_CHUNKS}] Saved {output_path}: {count} images")

        total_images += count

    print("\nGraph dataset generation finished.")
    print("Chunks:", len(chunk_paths))
    print("Images represented:", total_images)
    print("Output directory:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()