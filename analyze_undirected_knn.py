import math
import torch
from pathlib import Path


CHUNK_DIR = Path("structured_hoi")
K_VALUES = [1, 2, 3, 5, 10]


def box_center(box):
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def center_distance(box_a, box_b):
    center_a = box_center(box_a)
    center_b = box_center(box_b)

    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    return math.sqrt(dx * dx + dy * dy)


def collect_nodes(image):
    humans = {}
    objects = {}

    for hoi in image["hois"]:
        subject_id = hoi["subject_id"]
        object_id = hoi["object_id"]

        if subject_id not in humans:
            humans[subject_id] = {
                "category": hoi["subject_category"],
                "bbox": hoi["subject_bbox"],
            }

        if object_id not in objects:
            objects[object_id] = {
                "category": hoi["object_category"],
                "bbox": hoi["object_bbox"],
            }

    return humans, objects


def knn_undirected_edges(nodes, k, same_category=False):
    items = list(nodes.items())
    directed_candidates = {}

    for i, (node_id_a, node_a) in enumerate(items):
        candidates = []

        for j, (node_id_b, node_b) in enumerate(items):
            if i == j:
                continue

            if same_category:
                if node_a["category"] != node_b["category"]:
                    continue

            distance = center_distance(
                node_a["bbox"],
                node_b["bbox"],
            )

            candidates.append(
                (distance, node_id_b)
            )

        candidates.sort(key=lambda item: item[0])

        for distance, node_id_b in candidates[:k]:
            pair = tuple(
                sorted((node_id_a, node_id_b))
            )

            if pair not in directed_candidates:
                directed_candidates[pair] = distance
            else:
                directed_candidates[pair] = min(
                    directed_candidates[pair],
                    distance,
                )

    return directed_candidates


def main():
    chunk_files = sorted(
        CHUNK_DIR.glob("structured_chunk_*.pt")
    )

    results = {
        k: {
            "hh": [],
            "oo": [],
            "total": [],
        }
        for k in K_VALUES
    }

    total_images = 0

    print("Undirected KNN graph analysis")
    print("=" * 70)

    for chunk_file in chunk_files:
        print("Processing:", chunk_file.name)

        data = torch.load(
            chunk_file,
            map_location="cpu",
            weights_only=False,
        )

        for image in data["images"]:
            humans, objects = collect_nodes(image)

            for k in K_VALUES:
                hh_edges = knn_undirected_edges(
                    humans,
                    k,
                    same_category=False,
                )

                oo_edges = knn_undirected_edges(
                    objects,
                    k,
                    same_category=True,
                )

                hh_count = len(hh_edges)
                oo_count = len(oo_edges)

                results[k]["hh"].append(hh_count)
                results[k]["oo"].append(oo_count)
                results[k]["total"].append(
                    hh_count + oo_count
                )

            total_images += 1

    print("\nResults")
    print("-" * 70)
    print("Total images:", total_images)

    for k in K_VALUES:
        hh = results[k]["hh"]
        oo = results[k]["oo"]
        total = results[k]["total"]

        print(f"\nK = {k}")

        print(
            f"  H-H edges: "
            f"min={min(hh)}, "
            f"max={max(hh)}, "
            f"avg={sum(hh) / len(hh):.2f}"
        )

        print(
            f"  O-O edges: "
            f"min={min(oo)}, "
            f"max={max(oo)}, "
            f"avg={sum(oo) / len(oo):.2f}"
        )

        print(
            f"  Total contextual edges: "
            f"min={min(total)}, "
            f"max={max(total)}, "
            f"avg={sum(total) / len(total):.2f}"
        )


if __name__ == "__main__":
    main()