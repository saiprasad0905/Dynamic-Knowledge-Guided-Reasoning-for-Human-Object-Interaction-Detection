import glob
import numpy as np
import torch


CHUNK_DIR = "structured_hoi"
SCORE_THRESHOLD = 0.01


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

    object_ious = []

    for file_path in files:
        chunk = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )

        for image in chunk["images"]:
            candidates = [
                hoi
                for hoi in image["hois"]
                if hoi["score"] >= SCORE_THRESHOLD
            ]

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

                    object_ious.append(object_iou)

    values = np.array(object_ious)

    print("Total qualifying pair comparisons:", len(values))

    if len(values) == 0:
        return

    print("Mean object IoU:", round(float(values.mean()), 4))
    print("Median object IoU:", round(float(np.median(values)), 4))
    print("90th percentile:", round(float(np.percentile(values, 90)), 4))
    print("95th percentile:", round(float(np.percentile(values, 95)), 4))
    print("99th percentile:", round(float(np.percentile(values, 99)), 4))

    print()
    print("IoU threshold counts:")

    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95]:
        count = int((values >= threshold).sum())
        percentage = 100.0 * count / len(values)
        print(
            f">= {threshold:.2f}: "
            f"{count} ({percentage:.2f}%)"
        )


if __name__ == "__main__":
    main()