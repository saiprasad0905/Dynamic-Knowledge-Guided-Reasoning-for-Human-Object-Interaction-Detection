import glob
import torch


CHUNK_DIR = "structured_hoi"

SCORE_THRESHOLD = 0.01
OBJECT_IOU_THRESHOLD = 0.90
HUMAN_IOU_THRESHOLD = 0.50


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0.0, x2 - x1) *
        max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, box_a[2] - box_a[0]) *
        max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0]) *
        max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def main():
    files = sorted(
        glob.glob(f"{CHUNK_DIR}/structured_chunk_*.pt")
    )

    total_images = 0
    images_with_shared_objects = 0
    shared_object_relationships = 0

    for file_path in files:
        chunk = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )

        for image in chunk["images"]:
            total_images += 1

            # Keep only high-confidence HOI records.
            candidates = [
                hoi
                for hoi in image["hois"]
                if hoi["score"] >= SCORE_THRESHOLD
            ]

            # Remove multi-verb duplication.
            # Each human-object pair appears only once.
            unique_pairs = {}

            for hoi in candidates:
                pair = (
                    hoi["subject_id"],
                    hoi["object_id"],
                )

                if (
                        pair not in unique_pairs
                        or hoi["score"] > unique_pairs[pair]["score"]
                ):
                    unique_pairs[pair] = hoi

            pairs = list(unique_pairs.values())

            found_in_image = False

            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    first = pairs[i]
                    second = pairs[j]

                    if first["object_category"] != second["object_category"]:
                        continue

                    if first["subject_id"] == second["subject_id"]:
                        continue

                    object_iou = compute_iou(
                        first["object_bbox"],
                        second["object_bbox"],
                    )

                    human_iou = compute_iou(
                        first["subject_bbox"],
                        second["subject_bbox"],
                    )

                    if (
                        object_iou >= OBJECT_IOU_THRESHOLD
                        and human_iou < HUMAN_IOU_THRESHOLD
                    ):
                        shared_object_relationships += 1
                        found_in_image = True

            if found_in_image:
                images_with_shared_objects += 1

    print("Total images:", total_images)
    print(
        "Images with shared-object pattern:",
        images_with_shared_objects,
    )
    print(
        "Unique shared-object pair relationships:",
        shared_object_relationships,
    )


if __name__ == "__main__":
    main()