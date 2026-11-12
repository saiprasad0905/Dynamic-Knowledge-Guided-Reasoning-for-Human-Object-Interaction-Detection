
import json
from pathlib import Path

import torch

from train_gnn import load_graphs, calculate_metrics

EXPERIMENT_DIR = Path("gnn_experiment")
SPLIT_FILE = EXPERIMENT_DIR / "image_split.json"


def main():
    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    graphs = load_graphs()

    scores = []
    labels = []

    for image_id in split["validation_image_ids"]:
        for candidate in graphs[image_id]["labeled_candidates"]:
            scores.append(float(candidate["score"]))
            labels.append(float(candidate["label"]))

    scores = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    metrics = calculate_metrics(
        torch.logit(scores.clamp(1e-6, 1 - 1e-6)),
        labels,
    )

    print("QPIC CANDIDATE BASELINE")
    print("Validation images:", len(split["validation_image_ids"]))
    print("Validation candidates:", len(labels))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1 score:", round(metrics["f1"], 4))
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("True positives:", metrics["tp"])
    print("False positives:", metrics["fp"])
    print("False negatives:", metrics["fn"])
    print("True negatives:", metrics["tn"])


if __name__ == "__main__":
    main()
