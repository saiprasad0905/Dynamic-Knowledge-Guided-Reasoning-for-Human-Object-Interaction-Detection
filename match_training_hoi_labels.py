
import json
from pathlib import Path

import torch


ANNOTATIONS_PATH = (
    Path("..")
    / "hoi_capstone"
    / "data"
    / "hico_20160224_det"
    / "annotations"
    / "trainval_hico.json"
)

INPUT_DIR = Path("training_structured_hoi")
OUTPUT_DIR = Path("training_labeled_hoi")

IOU_THRESHOLD = 0.5

VALID_OBJ_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90,
)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0



def match_candidate(candidate, annotation):
    objects = annotation["annotations"]

    pred_subject_category = VALID_OBJ_IDS[
        candidate["subject_category"]
    ]
    pred_object_category = VALID_OBJ_IDS[
        candidate["object_category"]
    ]

    best_subject_iou = 0.0
    best_object_iou = 0.0
    matched = False

    for hoi in annotation["hoi_annotation"]:
        gt_subject = objects[hoi["subject_id"]]
        gt_object = objects[hoi["object_id"]]

        if gt_subject["category_id"] != pred_subject_category:
            continue
        if gt_object["category_id"] != pred_object_category:
            continue

        subject_iou = box_iou(
            candidate["subject_bbox"], gt_subject["bbox"]
        )
        object_iou = box_iou(
            candidate["object_bbox"], gt_object["bbox"]
        )

        # Keep IoUs from the same annotated subject-object pair.
        if subject_iou + object_iou > best_subject_iou + best_object_iou:
            best_subject_iou = subject_iou
            best_object_iou = object_iou

        if (
            subject_iou >= IOU_THRESHOLD
            and object_iou >= IOU_THRESHOLD
            and candidate["verb_id"] == hoi["category_id"]
        ):
            matched = True

    return matched, best_subject_iou, best_object_iou


def main():
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as file:
        annotations = json.load(file)

    annotation_by_id = {
        image_id: annotation
        for image_id, annotation in enumerate(annotations)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_candidates = 0
    total_positive = 0

    for chunk_index in range(5):
        input_path = (
            INPUT_DIR
            / f"training_structured_chunk_{chunk_index:04d}.pt"
        )

        chunk = torch.load(
            input_path,
            map_location="cpu",
            weights_only=False,
        )

        labeled_images = []

        for image in chunk["images"]:
            image_id = image["image_id"]

            if image_id not in annotation_by_id:
                raise IndexError(
                    f"No training annotation for image ID {image_id}"
                )

            annotation = annotation_by_id[image_id]
            labeled_candidates = []

            for candidate in image["hois"]:
                label, subject_iou, object_iou = match_candidate(
                    candidate, annotation
                )

                labeled_candidate = dict(candidate)
                labeled_candidate["label"] = int(label)
                labeled_candidate["subject_iou"] = subject_iou
                labeled_candidate["object_iou"] = object_iou

                labeled_candidates.append(labeled_candidate)
                total_candidates += 1
                total_positive += int(label)

            labeled_images.append(
                {
                    "image_id": image_id,
                    "hois": labeled_candidates,
                }
            )

        output_path = (
            OUTPUT_DIR
            / f"training_labeled_chunk_{chunk_index:04d}.pt"
        )

        torch.save(
            {
                "images": labeled_images,
                "start_index": chunk["start_index"],
                "end_index": chunk["end_index"],
            },
            output_path,
        )

        print(
            f"Saved {output_path}: "
            f"{len(labeled_images)} images"
        )

    print("Total candidates:", total_candidates)
    print("Positive labels:", total_positive)
    print("Negative labels:", total_candidates - total_positive)


if __name__ == "__main__":
    main()
