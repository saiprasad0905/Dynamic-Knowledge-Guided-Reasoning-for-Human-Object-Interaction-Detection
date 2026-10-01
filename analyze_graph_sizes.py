import glob
import torch

CHUNK_DIR = "structured_hoi"
THRESHOLDS = [0.01, 0.05, 0.10]


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def main():
    files = sorted(glob.glob(f"{CHUNK_DIR}/structured_chunk_*.pt"))

    totals = {
        threshold: {
            "images": 0,
            "human_nodes": 0,
            "object_nodes": 0,
            "edges": 0,
            "shared_object_relationships": 0,
        }
        for threshold in THRESHOLDS
    }

    for file_path in files:
        chunk = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False
        )

        for image in chunk["images"]:

            for threshold in THRESHOLDS:

                candidates = [
                    hoi
                    for hoi in image["hois"]
                    if hoi["score"] >= threshold
                ]

                # Keep one highest-scoring HOI for each unique
                # subject-object query pair.
                unique_pairs = {}

                for hoi in candidates:
                    pair = (
                        hoi["subject_id"],
                        hoi["object_id"]
                    )

                    if (
                        pair not in unique_pairs
                        or hoi["score"] > unique_pairs[pair]["score"]
                    ):
                        unique_pairs[pair] = hoi

                pairs = list(unique_pairs.values())

                # Unique entity IDs.
                human_nodes = {
                    hoi["subject_id"]
                    for hoi in pairs
                }

                object_nodes = {
                    hoi["object_id"]
                    for hoi in pairs
                }

                # Each unique subject-object pair is one graph edge.
                edges = len(pairs)

                # Find shared-object relationships.
                shared_object_relationships = 0

                for i in range(len(pairs)):
                    for j in range(i + 1, len(pairs)):

                        first = pairs[i]
                        second = pairs[j]

                        # Same object category.
                        if (
                            first["object_category"]
                            != second["object_category"]
                        ):
                            continue

                        # Different human detections.
                        if (
                            first["subject_id"]
                            == second["subject_id"]
                        ):
                            continue

                        object_iou = compute_iou(
                            first["object_bbox"],
                            second["object_bbox"]
                        )

                        human_iou = compute_iou(
                            first["subject_bbox"],
                            second["subject_bbox"]
                        )

                        # Candidate shared-object pattern.
                        if (
                            object_iou >= 0.90
                            and human_iou < 0.50
                        ):
                            shared_object_relationships += 1

                totals[threshold]["images"] += 1
                totals[threshold]["human_nodes"] += len(human_nodes)
                totals[threshold]["object_nodes"] += len(object_nodes)
                totals[threshold]["edges"] += edges
                totals[threshold][
                    "shared_object_relationships"
                ] += shared_object_relationships

    print("\nActual graph entity analysis")
    print("=" * 70)

    for threshold in THRESHOLDS:

        data = totals[threshold]

        print(f"\nScore threshold >= {threshold}")
        print("-" * 40)

        print("Images:", data["images"])
        print("Unique human nodes:", data["human_nodes"])
        print("Unique object nodes:", data["object_nodes"])
        print("Unique human-object edges:", data["edges"])
        print(
            "Shared-object relationships:",
            data["shared_object_relationships"]
        )


if __name__ == "__main__":
    main()