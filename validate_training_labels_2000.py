
from pathlib import Path
import torch

LABEL_DIR = Path("training_labeled_hoi_2000")

total_images = 0
total_candidates = 0
total_positive = 0
total_negative = 0
all_image_ids = []
errors = []

for chunk_index in range(20):
    path = LABEL_DIR / (
        f"training_labeled_chunk_{chunk_index:04d}.pt"
    )

    if not path.exists():
        errors.append(f"Missing file: {path}")
        continue

    data = torch.load(
        path, map_location="cpu", weights_only=False
    )

    images = data["images"]
    expected_start = chunk_index * 100
    expected_end = expected_start + 100

    if len(images) != 100:
        errors.append(
            f"{path}: expected 100 images, found {len(images)}"
        )

    if (
        data.get("start_index") != expected_start
        or data.get("end_index") != expected_end
    ):
        errors.append(f"{path}: incorrect chunk index range")

    chunk_candidates = 0
    chunk_positive = 0
    chunk_negative = 0

    for image in images:
        image_id = image.get("image_id")

        if not isinstance(image_id, int):
            errors.append(f"{path}: invalid image ID {image_id}")
        else:
            all_image_ids.append(image_id)

        for hoi in image["hois"]:
            label = hoi.get("label")

            if label not in (0, 1):
                errors.append(
                    f"{path}: invalid label {label}"
                )
                continue

            chunk_candidates += 1

            if label == 1:
                chunk_positive += 1
            else:
                chunk_negative += 1

    total_images += len(images)
    total_candidates += chunk_candidates
    total_positive += chunk_positive
    total_negative += chunk_negative

    print(
        f"Chunk {chunk_index}: "
        f"images={len(images)}, "
        f"candidates={chunk_candidates}, "
        f"positive={chunk_positive}, "
        f"negative={chunk_negative}"
    )

unique_ids = len(all_image_ids) == len(set(all_image_ids))

if not unique_ids:
    errors.append("Duplicate image IDs detected")

print("\nOVERALL")
print("Images:", total_images)
print("Unique image IDs:", len(set(all_image_ids)))
print("Candidates:", total_candidates)
print("Positive labels:", total_positive)
print("Negative labels:", total_negative)

if total_images != 2000:
    errors.append(f"Expected 2000 images, found {total_images}")

if total_candidates != 200000:
    errors.append(
        f"Expected 200000 candidates, found {total_candidates}"
    )

if total_positive + total_negative != total_candidates:
    errors.append("Positive and negative counts do not sum to total")

if errors:
    print("\nVALIDATION ERRORS")
    for error in errors:
        print(error)
    print("\nALL LABEL CHUNKS VALID: False")
else:
    print("\nNo missing chunks, duplicate IDs, or invalid labels detected.")
    print("ALL LABEL CHUNKS VALID: True")
