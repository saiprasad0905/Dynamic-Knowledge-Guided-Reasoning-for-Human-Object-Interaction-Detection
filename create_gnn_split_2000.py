
import json
import random
from pathlib import Path

import torch

DATA_DIR = Path("training_hybrid_graph_dataset_2000")
OUTPUT_DIR = Path("gnn_experiment_2000")
SPLIT_FILE = OUTPUT_DIR / "image_split.json"

SEED = 42
TRAIN_IMAGES = 1600
VALIDATION_IMAGES = 400


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_ids = []

    for chunk_path in sorted(
        DATA_DIR.glob("training_graph_chunk_*.pt")
    ):
        data = torch.load(
            chunk_path,
            map_location="cpu",
            weights_only=False,
        )

        image_ids.extend(
            graph["image_id"] for graph in data["graphs"]
        )

    if len(image_ids) != 2000:
        raise ValueError(
            f"Expected 500 images, found {len(image_ids)}"
        )

    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image IDs found")

    shuffled_ids = image_ids.copy()
    random.Random(SEED).shuffle(shuffled_ids)

    train_ids = sorted(shuffled_ids[:TRAIN_IMAGES])
    validation_ids = sorted(
        shuffled_ids[TRAIN_IMAGES:]
    )

    if set(train_ids) & set(validation_ids):
        raise ValueError("Train/validation leakage detected")

    result = {
        "seed": SEED,
        "train_image_ids": train_ids,
        "validation_image_ids": validation_ids,
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("Saved split:", SPLIT_FILE)
    print("Training images:", len(train_ids))
    print("Validation images:", len(validation_ids))
    print("Overlapping IDs:", len(set(train_ids) & set(validation_ids)))
    print("Random seed:", SEED)


if __name__ == "__main__":
    main()
