
import json
from pathlib import Path

import torch
from train_gnn_controlled import load_graphs

EXPERIMENT_DIR = Path("gnn_controlled_experiment")
SPLIT_FILE = EXPERIMENT_DIR / "image_split.json"
OUTPUT_FILE = EXPERIMENT_DIR / "qpic_ap_controlled.json"

DATASETS = [
    ("400-image dataset", Path("training_hybrid_graph_dataset"), 500),
    ("1900-image dataset", Path("training_hybrid_graph_dataset_2000"), 2000),
]


def calculate_average_precision(scores, labels):
    scores = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    positives = int(labels.sum().item())
    if positives == 0:
        return 0.0

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[order]

    cumulative_true_positives = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)

    precision_at_rank = cumulative_true_positives / ranks
    average_precision = (
        precision_at_rank[sorted_labels == 1].sum().item() / positives
    )

    return average_precision


def main():
    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    validation_ids = split["validation_image_ids"]
    results = {}

    for name, data_dir, expected_count in DATASETS:
        graphs = load_graphs(data_dir, expected_count)

        scores = []
        labels = []

        for image_id in validation_ids:
            graph = graphs[image_id]

            for candidate in graph["labeled_candidates"]:
                scores.append(float(candidate["score"]))
                labels.append(int(candidate["label"]))

        ap = calculate_average_precision(scores, labels)

        results[name] = {
            "validation_images": len(validation_ids),
            "validation_candidates": len(labels),
            "positive_candidates": sum(labels),
            "qpic_candidate_ap": ap,
        }

        print(f"\n{name}")
        print("-" * 40)
        print("Validation images:", len(validation_ids))
        print("Validation candidates:", len(labels))
        print("Positive candidates:", sum(labels))
        print("QPIC candidate AP:", round(ap, 6))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nSaved AP results:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
