
import json
import random
from pathlib import Path

import torch

SMALL_DATA_DIR = Path("training_hybrid_graph_dataset")
LARGE_DATA_DIR = Path("training_hybrid_graph_dataset_2000")

OUTPUT_DIR = Path("gnn_controlled_experiment")
SPLIT_FILE = OUTPUT_DIR / "image_split.json"

SEED = 42
VALIDATION_IMAGES = 100


def load_image_ids(data_dir, expected_count):
    image_ids = []

    for chunk_path in sorted(
        data_dir.glob("training_graph_chunk_*.pt")
    ):
        data = torch.load(
            chunk_path,
            map_location="cpu",
            weights_only=False,
        )
        image_ids.extend(
            graph["image_id"] for graph in data["graphs"]
        )

    if len(image_ids) != expected_count:
        raise ValueError(
            f"{data_dir}: expected {expected_count} images, "
            f"found {len(image_ids)}"
        )

    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"Duplicate image IDs in {data_dir}")

    return image_ids


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    small_ids = load_image_ids(SMALL_DATA_DIR, 500)
    large_ids = load_image_ids(LARGE_DATA_DIR, 2000)

    # Both datasets should contain the same initial 500 image IDs.
    if set(small_ids) != set(large_ids[:500]):
        raise ValueError(
            "The 500-image dataset does not match the first "
            "500 image IDs in the 2000-image dataset."
        )

    # Select the same 100 validation images for both experiments.
    shuffled_ids = small_ids.copy()
    random.Random(SEED).shuffle(shuffled_ids)
    validation_ids = sorted(shuffled_ids[:VALIDATION_IMAGES])

    validation_set = set(validation_ids)

    # Both training sets exclude every shared validation image.
    small_train_ids = sorted(set(small_ids) - validation_set)
    large_train_ids = sorted(set(large_ids) - validation_set)

    if len(small_train_ids) != 400:
        raise ValueError("Expected 400 small-experiment training images")

    if len(large_train_ids) != 1900:
        raise ValueError("Expected 1900 large-experiment training images")

    if set(small_train_ids) & validation_set:
        raise ValueError("Small train/validation leakage")

    if set(large_train_ids) & validation_set:
        raise ValueError("Large train/validation leakage")

    result = {
        "seed": SEED,
        "validation_image_ids": validation_ids,
        "small_train_image_ids": small_train_ids,
        "large_train_image_ids": large_train_ids,
        "small_train_count": len(small_train_ids),
        "large_train_count": len(large_train_ids),
        "validation_count": len(validation_ids),
        "note": (
            "Controlled dataset-size split. Both training sets use "
            "the same 100 held-out validation images."
        ),
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("Saved controlled split:", SPLIT_FILE)
    print("Small training images:", len(small_train_ids))
    print("Large training images:", len(large_train_ids))
    print("Shared validation images:", len(validation_ids))
    print(
        "Small train/validation overlap:",
        len(set(small_train_ids) & validation_set),
    )
    print(
        "Large train/validation overlap:",
        len(set(large_train_ids) & validation_set),
    )


if __name__ == "__main__":
    main()
