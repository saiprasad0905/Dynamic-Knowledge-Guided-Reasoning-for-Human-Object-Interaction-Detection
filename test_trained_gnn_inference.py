
import torch

from gnn_model import HybridHOIGNN, graph_to_tensors


GRAPH_FILE = (
    "training_hybrid_graph_dataset_2000/"
    "training_graph_chunk_0000.pt"
)

CHECKPOINT_FILE = (
    "gnn_controlled_experiment/"
    "best_gnn_model_1900.pt"
)


def main():
    print("Loading saved graph...")
    data = torch.load(
        GRAPH_FILE,
        map_location="cpu",
        weights_only=False,
    )
    graph = data["graphs"][0]

    print("Loading trained checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["config"]
    model = HybridHOIGNN(
        num_object_categories=config["num_object_categories"],
        num_verbs=config["num_verbs"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tensors = graph_to_tensors(graph)

    print("Running trained GNN...")
    with torch.no_grad():
        logits = model(tensors)
        probabilities = torch.sigmoid(logits)

    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite GNN output detected.")

    order = torch.argsort(
        probabilities, descending=True
    )[:10]

    print("\nCheckpoint epoch:", checkpoint["epoch"])
    print("Image:", graph["filename"])
    print("Candidates scored:", len(probabilities))
    print("Finite outputs: True")
    print("\nTop 10 GNN-ranked candidates:")

    for rank, index in enumerate(order.tolist(), start=1):
        candidate = graph["labeled_candidates"][index]
        print(
            f"{rank}. verb={candidate['verb']} | "
            f"QPIC={candidate['score']:.4f} | "
            f"GNN={probabilities[index].item():.4f}"
        )

    print("\nPASS: Trained checkpoint inference completed.")


if __name__ == "__main__":
    main()
