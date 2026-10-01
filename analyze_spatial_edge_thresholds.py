import math
import torch
from pathlib import Path


CHUNK_DIR = Path("structured_hoi")

DISTANCE_THRESHOLDS = [50, 100, 150, 200, 300]


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


def collect_edges(image):
    hois = image["hois"]

    humans = {}
    objects = {}

    for hoi in hois:
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

    human_distances = []
    object_distances = []

    human_items = list(humans.items())
    object_items = list(objects.items())

    for i in range(len(human_items)):
        _, human_a = human_items[i]

        for j in range(i + 1, len(human_items)):
            _, human_b = human_items[j]

            distance = center_distance(
                human_a["bbox"],
                human_b["bbox"],
            )

            human_distances.append(distance)

    for i in range(len(object_items)):
        _, object_a = object_items[i]

        for j in range(i + 1, len(object_items)):
            _, object_b = object_items[j]

            if object_a["category"] != object_b["category"]:
                continue

            distance = center_distance(
                object_a["bbox"],
                object_b["bbox"],
            )

            object_distances.append(distance)

    return human_distances, object_distances


def main():
    chunk_files = sorted(
        CHUNK_DIR.glob("structured_chunk_*.pt")
    )

    human_distances = []
    object_distances = []

    total_images = 0

    print("Spatial edge threshold analysis")
    print("=" * 70)

    for chunk_file in chunk_files:
        print("Processing:", chunk_file.name)

        data = torch.load(
            chunk_file,
            map_location="cpu",
            weights_only=False,
        )

        for image in data["images"]:
            humans, objects = collect_edges(image)

            human_distances.extend(humans)
            object_distances.extend(objects)

            total_images += 1

    print("\nDataset")
    print("-" * 70)
    print("Images:", total_images)

    print("\nHuman-human spatial edges")
    print("-" * 70)

    total_human = len(human_distances)

    print("Total possible pairs:", total_human)

    for threshold in DISTANCE_THRESHOLDS:
        count = sum(
            distance <= threshold
            for distance in human_distances
        )

        percentage = (
            100.0 * count / total_human
            if total_human > 0
            else 0.0
        )

        print(
            f"Distance <= {threshold:3d}: "
            f"{count:8d} "
            f"({percentage:6.2f}%)"
        )

    print("\nObject-object spatial edges")
    print("-" * 70)

    total_object = len(object_distances)

    print("Total possible same-category pairs:", total_object)

    for threshold in DISTANCE_THRESHOLDS:
        count = sum(
            distance <= threshold
            for distance in object_distances
        )

        percentage = (
            100.0 * count / total_object
            if total_object > 0
            else 0.0
        )

        print(
            f"Distance <= {threshold:3d}: "
            f"{count:8d} "
            f"({percentage:6.2f}%)"
        )


if __name__ == "__main__":
    main()