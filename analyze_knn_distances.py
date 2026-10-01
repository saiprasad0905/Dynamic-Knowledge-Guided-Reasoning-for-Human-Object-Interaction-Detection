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


def nearest_distances(nodes, k, same_category=False):
    items = list(nodes.items())
    selected_distances = []

    for i, (_, node_a) in enumerate(items):
        distances = []

        for j, (_, node_b) in enumerate(items):
            if i == j:
                continue

            if same_category:
                if node_a["category"] != node_b["category"]:
                    continue

            distance = center_distance(
                node_a["bbox"],
                node_b["bbox"],
            )

            distances.append(distance)

        distances.sort()

        selected_distances.extend(
            distances[:k]
        )

    return selected_distances


def main():
    chunk_files = sorted(
        CHUNK_DIR.glob("structured_chunk_*.pt")
    )

    results = {
        k: {
            "hh": [],
            "oo": [],
        }
        for k in K_VALUES
    }

    total_images = 0

    print("KNN neighbor distance analysis")
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
                hh = nearest_distances(
                    humans,
                    k,
                    same_category=False,
                )

                oo = nearest_distances(
                    objects,
                    k,
                    same_category=True,
                )

                results[k]["hh"].extend(hh)
                results[k]["oo"].extend(oo)

            total_images += 1

    print("\nResults")
    print("-" * 70)
    print("Total images:", total_images)

    for k in K_VALUES:
        hh = results[k]["hh"]
        oo = results[k]["oo"]

        print(f"\nK = {k}")

        print(
            f"  H-H selected neighbors: "
            f"{len(hh)}"
        )

        print(
            f"  H-H average distance: "
            f"{sum(hh) / len(hh):.2f} px"
        )

        print(
            f"  H-H maximum distance: "
            f"{max(hh):.2f} px"
        )

        print(
            f"  O-O selected neighbors: "
            f"{len(oo)}"
        )

        print(
            f"  O-O average distance: "
            f"{sum(oo) / len(oo):.2f} px"
        )

        print(
            f"  O-O maximum distance: "
            f"{max(oo):.2f} px"
        )


if __name__ == "__main__":
    main()