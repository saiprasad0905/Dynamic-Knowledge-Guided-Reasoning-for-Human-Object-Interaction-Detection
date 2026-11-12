
import json
from pathlib import Path

import matplotlib.pyplot as plt

HISTORY_FILE = Path("gnn_experiment/training_history.json")
OUTPUT_DIR = Path("gnn_experiment")


def main():
    with open(HISTORY_FILE, "r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = [row["epoch"] for row in history]
    train_loss = [
        row["train_weighted_loss"] for row in history
    ]
    validation_loss = [
        row["validation"]["loss"] for row in history
    ]
    validation_f1 = [
        row["validation"]["f1"] for row in history
    ]

    best_loss_index = min(
        range(len(validation_loss)),
        key=lambda i: validation_loss[i],
    )
    best_f1_index = max(
        range(len(validation_f1)),
        key=lambda i: validation_f1[i],
    )

    # Plot 1: Training and validation losses.
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, marker="o",
             label="Training weighted loss")
    plt.plot(epochs, validation_loss, marker="o",
             label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GNN Training and Validation Loss")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    loss_path = OUTPUT_DIR / "gnn_loss_curve.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    # Plot 2: Validation F1 across epochs.
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, validation_f1, marker="o",
             label="Validation F1")
    plt.scatter(
        [epochs[best_f1_index]],
        [validation_f1[best_f1_index]],
        label="Best validation F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("GNN Validation F1 by Epoch")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    f1_path = OUTPUT_DIR / "gnn_validation_f1.png"
    plt.savefig(f1_path, dpi=150)
    plt.close()

    print("Training epochs:", len(epochs))
    print(
        "Best validation loss:",
        round(validation_loss[best_loss_index], 4),
        "at epoch",
        epochs[best_loss_index],
    )
    print(
        "Best validation F1 at threshold 0.5:",
        round(validation_f1[best_f1_index], 4),
        "at epoch",
        epochs[best_f1_index],
    )
    print("Saved loss plot:", loss_path)
    print("Saved F1 plot:", f1_path)


if __name__ == "__main__":
    main()
