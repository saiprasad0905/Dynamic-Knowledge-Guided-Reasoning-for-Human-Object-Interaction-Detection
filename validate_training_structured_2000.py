
from pathlib import Path
import torch

directory = Path("training_structured_hoi_2000")
expected_chunks = 20
all_valid = True
all_image_ids = []
total_candidates = 0

for i in range(expected_chunks):
    path = directory / f"training_structured_chunk_{i:04d}.pt"

    if not path.exists():
        print(f"Chunk {i}: MISSING FILE")
        all_valid = False
        continue

    data = torch.load(path, map_location="cpu", weights_only=False)
    images = data.get("images", [])

    expected_start = i * 100
    expected_end = expected_start + 100

    valid = (
        len(images) == 100
        and data.get("start_index") == expected_start
        and data.get("end_index") == expected_end
        and all(
            isinstance(image.get("image_id"), int)
            and isinstance(image.get("hois"), list)
            and len(image["hois"]) <= 100
            for image in images
        )
    )

    image_ids = [image["image_id"] for image in images]
    all_image_ids.extend(image_ids)
    candidate_count = sum(len(image["hois"]) for image in images)
    total_candidates += candidate_count

    print(
        f"Chunk {i}: images={len(images)}, "
        f"candidates={candidate_count}, valid={valid}"
    )

    all_valid = all_valid and valid

unique_ids = len(set(all_image_ids)) == len(all_image_ids)
correct_total = len(all_image_ids) == 2000

print("\nTotal images:", len(all_image_ids))
print("Unique image IDs:", len(set(all_image_ids)))
print("Total HOI candidates:", total_candidates)
print("Image IDs unique:", unique_ids)

all_valid = all_valid and unique_ids and correct_total
print("ALL STRUCTURED CHUNKS VALID:", all_valid)
