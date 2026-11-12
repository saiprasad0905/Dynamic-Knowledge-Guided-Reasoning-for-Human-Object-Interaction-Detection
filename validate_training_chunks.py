
from pathlib import Path
import torch

directory = Path("training_evaluation_chunks_500")
all_valid = True

for i in range(5):
    path = directory / f"training_chunk_{i:04d}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)

    expected_start = i * 100
    expected_end = expected_start + 100

    valid = (
        len(data["preds"]) == 100
        and len(data["gts"]) == 100
        and data["start_index"] == expected_start
        and data["end_index"] == expected_end
        and all(
            tuple(pred["verb_scores"].shape) == (100, 117)
            and tuple(pred["boxes"].shape) == (200, 4)
            for pred in data["preds"]
        )
        and all(
            target["id"] == expected_start + j
            for j, target in enumerate(data["gts"])
        )
    )

    print(
        f"Chunk {i}: "
        f"predictions={len(data['preds'])}, "
        f"targets={len(data['gts'])}, "
        f"range={data['start_index']}-{data['end_index']}, "
        f"valid={valid}"
    )

    all_valid = all_valid and valid

print("ALL CHUNKS VALID:", all_valid)
