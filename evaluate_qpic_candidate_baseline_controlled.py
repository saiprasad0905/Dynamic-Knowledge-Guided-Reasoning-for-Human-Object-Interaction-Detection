
import json
from pathlib import Path

import torch

from train_gnn_controlled import load_graphs, calculate_metrics

EXPERIMENT_DIR = Path("gnn_controlled_experiment")
SPLIT_FILE = EXPERIMENT_DIR / "image_split.json"
OUTPUT_FILE = EXPERIMENT_DIR / "qpic_baseline_controlled.json"

DATASETS = [
    {
        "name": "400-image dataset",
        "data_dir": Path("training_hybrid_graph_dataset"),
        "expected_graphs": 500,
    },
    {
        "name": "1900-image dataset",
        "data_dir": Path("training_hybrid_graph_dataset_2000"),
        "expected_graphs": 2000,
    },
]


def main():
    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    validation_ids = split["validation_image_ids"]
    results = []

    for config in DATASETS:
        graphs = load_graphs(
            config["data_dir"],
            config["expected_graphs"],
        )

        scores = []
        labels = []

        for image_id in validation_ids:
            for candidate in graphs[image_id]["labeled_candidates"]:
                scores.append(float(candidate["score"]))
                labels.append(float(candidate["label"]))

        scores = torch.tensor(scores, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        metrics = calculate_metrics(
            torch.logit(scores.clamp(1e-6, 1 - 1e-6)),
            labels,
        )

        result = {
            "dataset": config["name"],
            "validation_images": len(validation_ids),
            "validation_candidates": len(labels),
            "precision_at_0_5": metrics["precision"],
            "recall_at_0_5": metrics["recall"],
            "f1_at_0_5": metrics["f1"],
            "accuracy_at_0_5": metrics["accuracy"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
        }
        results.append(result)

        print(f"\nQPIC baseline: {config['name']}")
        print("-" * 48)
        for key, value in result.items():
            print(f"{key}: {value}")

        del graphs

    output = {
        "validation_image_ids": validation_ids,
        "note": (
            "QPIC candidate scores evaluated at threshold 0.5 on the "
            "controlled shared validation image IDs. Metrics are "
            "candidate-level, not official HICO-DET mAP."
        ),
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\nSaved baseline results:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
