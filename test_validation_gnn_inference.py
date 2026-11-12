import torch
from pathlib import Path
# from sklearn.metrics import average_precision_score

from gnn_model import HybridHOIGNN, graph_to_tensors

GRAPH_DIR = Path("training_hybrid_graph_dataset_2000")
TARGET_IMAGE = "HICO_train2015_00000005.jpg"
CHECKPOINT_FILE = Path(
    "gnn_controlled_experiment/best_gnn_model_1900.pt"
)


def main():
    print("Searching for held-out validation graph...")

    graph = None
    for path in sorted(GRAPH_DIR.glob("*.pt")):
        data = torch.load(
            path, map_location="cpu", weights_only=False
        )
        for candidate_graph in data["graphs"]:
            if candidate_graph.get("filename") == TARGET_IMAGE:
                graph = candidate_graph
                print("Graph file:", path)
                break
        if graph is not None:
            break

    if graph is None:
        raise FileNotFoundError(
            f"Could not find validation graph: {TARGET_IMAGE}"
        )

    print("Image:", graph["filename"])
    print("Image ID:", graph.get("image_id"))

    print("\nLoading trained GNN checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT_FILE, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]

    model = HybridHOIGNN(
        num_object_categories=config["num_object_categories"],
        num_verbs=config["num_verbs"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Ground-truth labels are retained only for evaluation below.
    # The model receives graph features and candidate inputs.
    tensors = graph_to_tensors(graph)

    print("Running trained GNN...")
    with torch.no_grad():
        logits = model(tensors)
        probabilities = torch.sigmoid(logits).reshape(-1)

    candidates = graph["labeled_candidates"]
    labels = torch.tensor(
        [int(c["label"]) for c in candidates],
        dtype=torch.int64,
    )

    if len(probabilities) != len(candidates):
        raise ValueError(
            f"Output/candidate mismatch: {len(probabilities)} vs "
            f"{len(candidates)}"
        )
    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite GNN probabilities detected.")

    # Average precision, computed without scikit-learn
    sorted_indices = torch.argsort(
        probabilities, descending=True
    )
    sorted_labels = labels[sorted_indices].float()

    if sorted_labels.sum() > 0:
        cumulative_tp = torch.cumsum(sorted_labels, dim=0)
        ranks = torch.arange(
            1, len(sorted_labels) + 1, dtype=torch.float32
        )
        precision_at_rank = cumulative_tp / ranks
        ap = (
                     precision_at_rank * sorted_labels
             ).sum() / sorted_labels.sum()
        ap = ap.item()
    else:
        ap = float("nan")
    predicted = probabilities >= 0.5
    tp = int(((predicted) & (labels == 1)).sum())
    fp = int(((predicted) & (labels == 0)).sum())
    fn = int(((~predicted) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    order = torch.argsort(probabilities, descending=True)[:10]

    print("\nCheckpoint epoch:", checkpoint["epoch"])
    print("Candidates scored:", len(probabilities))
    print("Positive candidates:", int(labels.sum()))
    print("Candidate AP:", f"{ap:.4f}")
    print("Precision at threshold 0.50:", f"{precision:.4f}")
    print("Recall at threshold 0.50:", f"{recall:.4f}")
    print("F1 at threshold 0.50:", f"{f1:.4f}")

    print("\nTop 10 GNN-ranked candidates:")
    for rank, idx in enumerate(order.tolist(), start=1):
        candidate = candidates[idx]
        print(
            f"{rank}. verb={candidate['verb']} | "
            f"QPIC={float(candidate['score']):.4f} | "
            f"GNN={probabilities[idx].item():.4f} | "
            f"GT_label={int(labels[idx])}"
        )

    print("\nPASS: Held-out validation inference completed.")


if __name__ == "__main__":
    main()
