
import json
from pathlib import Path

import torch

from gnn_model import HybridHOIGNN, graph_to_tensors
from train_gnn_controlled import load_graphs, evaluate

EXPERIMENT_DIR = Path("gnn_controlled_experiment")
SPLIT_FILE = EXPERIMENT_DIR / "image_split.json"
OUTPUT_FILE = EXPERIMENT_DIR / "controlled_evaluation.json"

MODELS = [
    {
        "name": "400-image model",
        "data_dir": Path("training_hybrid_graph_dataset"),
        "expected_graphs": 500,
        "train_ids_key": "small_train_image_ids",
        "checkpoint": EXPERIMENT_DIR / "best_gnn_model_400.pt",
    },
    {
        "name": "1900-image model",
        "data_dir": Path("training_hybrid_graph_dataset_2000"),
        "expected_graphs": 2000,
        "train_ids_key": "large_train_image_ids",
        "checkpoint": EXPERIMENT_DIR / "best_gnn_model_1900.pt",
    },
]


def calculate_average_precision(scores, labels):
    """Compute candidate-level AP by ranking scores in descending order."""
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.bool)

    positive_count = int(labels.sum().item())
    if positive_count == 0:
        return 0.0

    order = torch.argsort(scores, descending=True, stable=True)
    ranked_labels = labels[order].to(torch.float64)

    cumulative_tp = torch.cumsum(ranked_labels, dim=0)
    ranks = torch.arange(1, len(labels) + 1, dtype=torch.float64)
    precision_at_rank = cumulative_tp / ranks

    return float(
        (precision_at_rank * ranked_labels).sum().item()
        / positive_count
    )


def main():
    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    validation_ids = split["validation_image_ids"]
    validation_set = set(validation_ids)

    results = []

    for config in MODELS:
        train_ids = split[config["train_ids_key"]]

        if len(train_ids) + len(validation_ids) != config["expected_graphs"]:
            raise ValueError(
                f"Unexpected split sizes for {config['name']}"
            )

        if set(train_ids) & validation_set:
            raise ValueError(
                f"Train/validation leakage for {config['name']}"
            )

        checkpoint = torch.load(
            config["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )

        graphs = load_graphs(
            config["data_dir"],
            config["expected_graphs"],
        )

        if not validation_set.issubset(graphs):
            raise ValueError(
                f"Missing validation graphs for {config['name']}"
            )

        model = HybridHOIGNN(
            num_object_categories=80,
            num_verbs=118,
            hidden_dim=64,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        metrics = evaluate(model, graphs, validation_ids)

        all_scores = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for image_id in validation_ids:
                graph = graphs[image_id]
                tensors = graph_to_tensors(graph)
                logits = model(tensors)

                all_scores.extend(torch.sigmoid(logits).tolist())
                all_labels.extend(tensors["labels"].tolist())

        ap = calculate_average_precision(all_scores, all_labels)

        result = {
            "model": config["name"],
            "training_images": len(train_ids),
            "validation_images": metrics["images"],
            "validation_candidates": metrics["candidates"],
            "checkpoint_epoch": checkpoint["epoch"],
            "saved_best_validation_loss": checkpoint["validation_loss"],
            "recomputed_validation_loss": metrics["loss"],
            "precision_at_0_5": metrics["precision"],
            "recall_at_0_5": metrics["recall"],
            "f1_at_0_5": metrics["f1"],
            "accuracy_at_0_5": metrics["accuracy"],
            "candidate_level_ap": ap,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
        }
        results.append(result)

        print(f"\n{config['name']}")
        print("-" * 48)
        for key, value in result.items():
            print(f"{key}: {value}")

        del model, graphs

    output = {
        "validation_image_ids": validation_ids,
        "validation_images": len(validation_ids),
        "note": (
            "Both models use the same held-out validation image IDs. "
            "Metrics are candidate-level under the project's label-matching "
            "rule; AP is not official HICO-DET mAP. The small and large "
            "graph datasets were generated separately, so candidate-level "
            "counts may differ if graph construction differs."
        ),
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\nSaved comparison:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
