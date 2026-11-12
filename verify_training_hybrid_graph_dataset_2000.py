
from pathlib import Path
import torch

DATA_DIR = Path("training_hybrid_graph_dataset_2000")
EXPECTED_CHUNKS = 20
EXPECTED_IMAGES = 2000
EXPECTED_CANDIDATES = 200000

chunk_files = sorted(
    DATA_DIR.glob("training_graph_chunk_*.pt")
)

total_images = 0
total_candidates = 0
total_positive = 0
total_negative = 0
image_ids = []
errors = []

print("Training hybrid graph verification")
print("=" * 65)
print("Chunks found:", len(chunk_files))

if len(chunk_files) != EXPECTED_CHUNKS:
    errors.append(
        f"Expected {EXPECTED_CHUNKS} chunks, found {len(chunk_files)}"
    )

for chunk_path in chunk_files:
    data = torch.load(
        chunk_path,
        map_location="cpu",
        weights_only=False,
    )

    graphs = data.get("graphs", [])
    chunk_candidates = 0
    chunk_positive = 0
    chunk_negative = 0

    for graph in graphs:
        image_id = graph["image_id"]
        image_ids.append(image_id)

        humans = graph["humans"]
        objects = graph["objects"]
        hoi_edges = graph["hoi_edges"]
        human_edges = graph["human_edges"]
        object_edges = graph["object_edges"]
        candidates = graph["labeled_candidates"]

        if graph["num_positive_candidates"] != sum(
            c["label"] == 1 for c in candidates
        ):
            errors.append(
                f"Positive count mismatch in image {image_id}"
            )

        if graph["num_negative_candidates"] != sum(
            c["label"] == 0 for c in candidates
        ):
            errors.append(
                f"Negative count mismatch in image {image_id}"
            )

        for edge in hoi_edges:
            if edge["subject_id"] not in humans:
                errors.append(
                    f"Missing human node in image {image_id}"
                )

            if edge["object_id"] not in objects:
                errors.append(
                    f"Missing object node in image {image_id}"
                )

            for candidate in edge["verb_candidates"]:
                if candidate["label"] not in (0, 1):
                    errors.append(
                        f"Invalid edge label in image {image_id}"
                    )

        for edge in human_edges:
            if (
                edge["source"] not in humans
                or edge["target"] not in humans
            ):
                errors.append(
                    f"Invalid human spatial edge in image {image_id}"
                )

        for edge in object_edges:
            if (
                edge["source"] not in objects
                or edge["target"] not in objects
            ):
                errors.append(
                    f"Invalid object spatial edge in image {image_id}"
                )

        chunk_candidates += len(candidates)
        chunk_positive += sum(
            c["label"] == 1 for c in candidates
        )
        chunk_negative += sum(
            c["label"] == 0 for c in candidates
        )

    total_images += len(graphs)
    total_candidates += chunk_candidates
    total_positive += chunk_positive
    total_negative += chunk_negative

    print(
        f"{chunk_path.name}: "
        f"images={len(graphs)}, "
        f"candidates={chunk_candidates}, "
        f"positive={chunk_positive}, "
        f"negative={chunk_negative}"
    )

print("\nOverall")
print("-" * 65)
print("Images:", total_images)
print("Unique image IDs:", len(set(image_ids)))
print("Candidates:", total_candidates)
print("Positive labels:", total_positive)
print("Negative labels:", total_negative)

if len(set(image_ids)) != len(image_ids):
    errors.append("Duplicate image IDs found")

if total_images != EXPECTED_IMAGES:
    errors.append("Unexpected total image count")

if total_candidates != EXPECTED_CANDIDATES:
    errors.append("Unexpected candidate count")

if total_positive + total_negative != total_candidates:
    errors.append("Label totals do not match candidate total")

if errors:
    print("\nVALIDATION ERRORS")
    for error in errors:
        print("-", error)
else:
    print("\nPASS: Graph structure and candidate labels verified.")
