
import torch
from pathlib import Path

output_dir = Path("hybrid_graph_dataset")
chunk_files = sorted(output_dir.glob("graph_chunk_*.pt"))

total_images = 0
image_ids = []

print("Chunks:", len(chunk_files))

for file_path in chunk_files:
    data = torch.load(
        file_path,
        map_location="cpu",
        weights_only=False,
    )

    graphs = data["graphs"]
    count = len(graphs)

    total_images += count
    image_ids.extend(graph["image_id"] for graph in graphs)

    print(f"{file_path.name}: {count} images")

print("\nTotal images:", total_images)
print("Unique image IDs:", len(set(image_ids)))
print("Duplicate IDs:", len(image_ids) - len(set(image_ids)))
print("Expected images: 9546")

if (
    len(chunk_files) == 20
    and total_images == 9546
    and len(set(image_ids)) == 9546
):
    print("\nPASS: Image count and ID uniqueness verified.")
else:
    print("\nCHECK REQUIRED: Dataset counts do not match expectations.")
