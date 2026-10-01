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

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def main():
    files = sorted(glob.glob(f"{CHUNK_DIR}/structured_chunk_*.pt"))

    total_images = 0
    images_with_shared_objects = 0
    shared_object_pairs = 0

    for file_path in files:
        chunk = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )

        for image in chunk["images"]:
            total_images += 1

            candidates = [
                hoi
                for hoi in image["hois"]
                if hoi["score"] >= SCORE_THRESHOLD
            ]

            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    first = candidates[i]
                    second = candidates[j]

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
                        shared_object_pairs += 1
                        images_with_shared_objects += 1
                        break

                else:
                    continue

                break

    print("Total images:", total_images)
    print("Images with shared-object pattern:", images_with_shared_objects)
    print("Candidate pair relationships:", shared_object_pairs)


if __name__ == "__main__":
    main()