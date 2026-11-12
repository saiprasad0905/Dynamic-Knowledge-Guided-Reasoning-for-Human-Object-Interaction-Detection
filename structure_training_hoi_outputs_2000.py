
import json
from pathlib import Path

import numpy as np
import torch


CHUNK_DIR = Path("training_evaluation_chunks_2000")
OUTPUT_DIR = Path("training_structured_hoi_2000")

CORRECT_MAT_PATH = (
    Path("..")
    / "hoi_capstone"
    / "data"
    / "hico_20160224_det"
    / "annotations"
    / "corre_hico.npy"
)

VERB_MAPPING_PATH = (
    Path("..")
    / "hoi_capstone"
    / "data"
    / "processed"
    / "verb_mapping.json"
)

MAX_HOIS_PER_IMAGE = 100


def build_structured_hois(prediction, correct_mat, verb_mapping):
    labels = prediction["labels"].cpu().numpy()
    boxes = prediction["boxes"].cpu().numpy()
    verb_scores = prediction["verb_scores"].cpu().numpy()
    sub_ids = prediction["sub_ids"].cpu().numpy()
    obj_ids = prediction["obj_ids"].cpu().numpy()

    num_queries, num_verbs = verb_scores.shape
    candidates = []

    for query_id in range(num_queries):
        subject_id = int(sub_ids[query_id])
        object_id = int(obj_ids[query_id])

        subject_category = int(labels[subject_id])
        object_category = int(labels[object_id])

        subject_bbox = boxes[subject_id].tolist()
        object_bbox = boxes[object_id].tolist()

        for verb_index in range(num_verbs):
            score = float(verb_scores[query_id, verb_index])

            if correct_mat[verb_index, object_category] == 0:
                continue

            verb_id = verb_index + 1
            mapping = verb_mapping.get(str(verb_id), {})

            candidates.append(
                {
                    "query_id": query_id,
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "subject_category": subject_category,
                    "object_category": object_category,
                    "subject_bbox": subject_bbox,
                    "object_bbox": object_bbox,
                    "verb_id": verb_id,
                    "verb": mapping.get("verb", "unknown"),
                    "score": score,
                }
            )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates[:MAX_HOIS_PER_IMAGE]


def process_chunk(chunk_index, correct_mat, verb_mapping):
    chunk_path = CHUNK_DIR / f"training_chunk_{chunk_index:04d}.pt"

    if not chunk_path.exists():
        raise FileNotFoundError(f"Missing chunk: {chunk_path}")

    chunk = torch.load(
        chunk_path,
        map_location="cpu",
        weights_only=False,
    )

    structured_images = []

    for prediction, target in zip(chunk["preds"], chunk["gts"]):
        image_id = int(target["id"])

        hois = build_structured_hois(
            prediction,
            correct_mat,
            verb_mapping,
        )

        structured_images.append(
            {
                "image_id": image_id,
                "hois": hois,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR / f"training_structured_chunk_{chunk_index:04d}.pt"
    )

    torch.save(
        {
            "images": structured_images,
            "start_index": chunk["start_index"],
            "end_index": chunk["end_index"],
        },
        output_path,
    )

    print(
        f"Saved training structured chunk {chunk_index + 1}: "
        f"{output_path} ({len(structured_images)} images)"
    )


if __name__ == "__main__":
    correct_mat = np.load(CORRECT_MAT_PATH)

    with open(VERB_MAPPING_PATH, "r", encoding="utf-8") as file:
        verb_mapping = json.load(file)

    total_chunks = 20

    for chunk_index in range(total_chunks):
        output_path = (
            OUTPUT_DIR
            / f"training_structured_chunk_{chunk_index:04d}.pt"
        )

        if output_path.exists():
            print(
                f"Training structured chunk {chunk_index + 1}/"
                f"{total_chunks} already exists: {output_path}"
            )
            continue

        print(
            f"\n========== PROCESSING TRAINING STRUCTURED CHUNK "
            f"{chunk_index + 1}/{total_chunks} =========="
        )

        process_chunk(
            chunk_index=chunk_index,
            correct_mat=correct_mat,
            verb_mapping=verb_mapping,
        )
