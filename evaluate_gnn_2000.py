
import json
from pathlib import Path

import torch

from gnn_model import HybridHOIGNN
from train_gnn_2000 import load_graphs, evaluate

EXPERIMENT_DIR = Path("gnn_experiment_2000")
CHECKPOINT_FILE = EXPERIMENT_DIR / "best_gnn_model.pt"
SPLIT_FILE = EXPERIMENT_DIR / "image_split.json"


def main():
    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    graphs = load_graphs()

    model = HybridHOIGNN(
        num_object_categories=80,
        num_verbs=118,
        hidden_dim=64,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(
        model,
        graphs,
        split["validation_image_ids"],
    )

    print("BEST CHECKPOINT EVALUATION")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Saved validation loss:", checkpoint["validation_loss"])
    print("Recomputed validation loss:", metrics["loss"])
    print("Validation images:", metrics["images"])
    print("Validation candidates:", metrics["candidates"])
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
