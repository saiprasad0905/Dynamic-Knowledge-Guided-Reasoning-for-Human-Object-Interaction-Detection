
from pathlib import Path
import torch

LABEL_DIR = Path("training_labeled_hoi")

total_images = 0
total_candidates = 0
total_positive = 0
total_negative = 0
errors = []

for chunk_index in range(5):
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
    chunk_candidates = 0
    chunk_positive = 0
    chunk_negative = 0

    for image in images:
        for hoi in image["hois"]:
            label = hoi.get("label")

            if label not in (0, 1):
                errors.append(
                    f"Invalid label in {path}: {label}"
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

print("\nOVERALL")
print("Images:", total_images)
print("Candidates:", total_candidates)
print("Positive labels:", total_positive)
print("Negative labels:", total_negative)

if errors:
    print("\nVALIDATION ERRORS")
    for error in errors:
        print(error)
else:
    print("\nNo missing chunks or invalid labels detected.")

if (
    total_images != 500
    or total_candidates != 50000
    or total_positive + total_negative != total_candidates
):
    print("WARNING: Check the totals above.")
