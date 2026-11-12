
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_model import HybridHOIGNN, graph_to_tensors


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("training_hybrid_graph_dataset_2000")
EXPERIMENT_DIR = Path("gnn_experiment_2000")
SPLIT_FILE = EXPERIMENT_DIR / "image_split.json"

CHECKPOINT_FILE = EXPERIMENT_DIR / "best_gnn_model.pt"
HISTORY_FILE = EXPERIMENT_DIR / "training_history.json"

SEED = 42
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_DIM = 64

DEVICE = torch.device("cpu")


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_graphs():
    """Load training graphs without changing the saved dataset."""
    graphs_by_id = {}

    for path in sorted(DATA_DIR.glob("training_graph_chunk_*.pt")):
        chunk = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        for graph in chunk["graphs"]:
            image_id = graph["image_id"]

            if image_id in graphs_by_id:
                raise ValueError(f"Duplicate image ID: {image_id}")

            graphs_by_id[image_id] = graph

    if len(graphs_by_id) != 2000:
        raise ValueError(
            f"Expected 2000 graphs, found {len(graphs_by_id)}"
        )

    return graphs_by_id


def calculate_metrics(logits, labels):
    """Calculate candidate-level metrics using threshold 0.5."""
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= 0.5
    actual = labels >= 0.5

    tp = int((predictions & actual).sum().item())
    fp = int((predictions & ~actual).sum().item())
    fn = int((~predictions & actual).sum().item())
    tn = int((~predictions & ~actual).sum().item())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (
        (tp + tn) / (tp + fp + fn + tn)
        if tp + fp + fn + tn
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate(model, graphs, image_ids):
    """Evaluate using ordinary, unweighted binary cross-entropy."""
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_labels = []
    candidate_count = 0

    with torch.no_grad():
        for image_id in image_ids:
            tensors = graph_to_tensors(graphs[image_id])

            logits = model(tensors)
            labels = tensors["labels"]

            loss = F.binary_cross_entropy_with_logits(
                logits, labels
            )

            count = labels.numel()
            total_loss += loss.item() * count
            candidate_count += count

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    metrics = calculate_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / candidate_count
    metrics["images"] = len(image_ids)
    metrics["candidates"] = candidate_count

    return metrics


def main():
    set_seeds(SEED)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    train_ids = split["train_image_ids"]
    validation_ids = split["validation_image_ids"]

    if len(train_ids) != 1600 or len(validation_ids) != 400:
        raise ValueError("Unexpected train/validation split sizes.")

    if set(train_ids) & set(validation_ids):
        raise ValueError("Train/validation image leakage detected.")

    graphs = load_graphs()

    if not set(train_ids + validation_ids).issubset(graphs):
        raise ValueError("Some split image IDs are missing from the dataset.")

    print("Loading completed.")
    print("Training images:", len(train_ids))
    print("Validation images:", len(validation_ids))
    print("Device:", DEVICE)

    # Compute class weighting from training labels only.
    positive_count = 0
    negative_count = 0

    for image_id in train_ids:
        for candidate in graphs[image_id]["labeled_candidates"]:
            if candidate["label"] == 1:
                positive_count += 1
            else:
                negative_count += 1

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain both classes.")

    pos_weight = negative_count / positive_count

    print("Training positive candidates:", positive_count)
    print("Training negative candidates:", negative_count)
    print("Positive-class loss weight:", round(pos_weight, 4))

    model = HybridHOIGNN(
        num_object_categories=80,
        num_verbs=118,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    weighted_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=DEVICE)
    )

    history = []
    best_validation_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        total_train_candidates = 0

        shuffled_ids = train_ids.copy()
        random.shuffle(shuffled_ids)

        for image_id in shuffled_ids:
            tensors = graph_to_tensors(graphs[image_id])
            labels = tensors["labels"]

            optimizer.zero_grad()

            logits = model(tensors)
            loss = weighted_loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            count = labels.numel()
            total_train_loss += loss.item() * count
            total_train_candidates += count

        train_loss = (
            total_train_loss / total_train_candidates
        )

        validation = evaluate(model, graphs, validation_ids)

        record = {
            "epoch": epoch,
            "train_weighted_loss": train_loss,
            "validation": validation,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train weighted loss: {train_loss:.4f} | "
            f"val loss: {validation['loss']:.4f} | "
            f"precision: {validation['precision']:.4f} | "
            f"recall: {validation['recall']:.4f} | "
            f"F1: {validation['f1']:.4f}"
        )

        # Select the checkpoint using unweighted validation loss.
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": best_validation_loss,
                    "config": {
                        "hidden_dim": HIDDEN_DIM,
                        "num_object_categories": 80,
                        "num_verbs": 118,
                        "learning_rate": LEARNING_RATE,
                        "seed": SEED,
                    },
                },
                CHECKPOINT_FILE,
            )

            print("  Saved best checkpoint.")

        with open(HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

    print("\nTraining completed.")
    print("Best checkpoint:", CHECKPOINT_FILE)
    print("Training history:", HISTORY_FILE)
    print("Best validation loss:", round(best_validation_loss, 4))


if __name__ == "__main__":
    main()
