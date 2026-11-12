import json
import torch
from pathlib import Path
from PIL import Image, ImageDraw

from gnn_model import HybridHOIGNN, graph_to_tensors

IMAGE_PATH = Path(
    "../hoi_capstone/data/hico_20160224_det/images/"
    "train2015/HICO_train2015_00000005.jpg"
)
GRAPH_DIR = Path("training_hybrid_graph_dataset_2000")
CHECKPOINT_FILE = Path(
    "gnn_controlled_experiment/best_gnn_model_1900.pt"
)
OUTPUT_DIR = Path("gnn_controlled_experiment")
TARGET_IMAGE = "HICO_train2015_00000005.jpg"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading validation graph...")
    graph = None

    for path in sorted(GRAPH_DIR.glob("*.pt")):
        data = torch.load(
            path, map_location="cpu", weights_only=False
        )
        for item in data["graphs"]:
            if item.get("filename") == TARGET_IMAGE:
                graph = item
                break
        if graph is not None:
            break

    if graph is None:
        raise FileNotFoundError(TARGET_IMAGE)

    print("Loading trained GNN...")
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

    tensors = graph_to_tensors(graph)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensors)).reshape(-1)

    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite GNN scores.")

    candidates = graph["labeled_candidates"]
    if len(candidates) != len(probabilities):
        raise ValueError("Candidate count does not match GNN outputs.")

    ranked_indices = torch.argsort(
        probabilities, descending=True
    ).tolist()

    # Save every candidate and score for reproducible analysis.
    results = {
        "image": graph["filename"],
        "image_id": graph.get("image_id"),
        "checkpoint": str(CHECKPOINT_FILE),
        "checkpoint_epoch": checkpoint["epoch"],
        "candidate_count": len(candidates),
        "results": [],
    }

    for rank, idx in enumerate(ranked_indices, start=1):
        candidate = candidates[idx]
        results["results"].append({
            "rank": rank,
            "query_id": int(candidate["query_id"]),
            "subject_id": int(candidate["subject_id"]),
            "object_id": int(candidate["object_id"]),
            "verb": str(candidate["verb"]),
            "verb_id": int(candidate["verb_id"]),
            "qpic_score": float(candidate["score"]),
            "gnn_score": float(probabilities[idx]),
            "ground_truth_label": int(candidate["label"]),
        })

    json_path = OUTPUT_DIR / "validation_image_4_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Open original image and draw the highest-ranked interaction pair.
    image = Image.open(IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    best = candidates[ranked_indices[0]]
    subject_id = int(best["subject_id"])
    object_id = int(best["object_id"])

    human = graph["humans"].get(subject_id)
    obj = graph["objects"].get(object_id)

    if human is None or obj is None:
        raise KeyError(
            f"Could not find subject {subject_id} or object {object_id}"
        )

    def clamp_box(box):
        x1, y1, x2, y2 = map(float, box)
        return (
            max(0, min(width - 1, x1)),
            max(0, min(height - 1, y1)),
            max(0, min(width - 1, x2)),
            max(0, min(height - 1, y2)),
        )

    human_box = clamp_box(human["bbox"])
    object_box = clamp_box(obj["bbox"])

    # Draw the person and object boxes.
    draw.rectangle(human_box, outline="lime", width=3)
    draw.rectangle(object_box, outline="red", width=3)

    best_score = float(probabilities[ranked_indices[0]])
    best_verb = str(best["verb"])
    qpic_score = float(best["score"])

    # Add a readable panel with the top five GNN candidates.
    panel_height = 145
    panel = Image.new("RGB", (width, panel_height), "white")
    panel_draw = ImageDraw.Draw(panel)

    panel_draw.text(
        (10, 8),
        f"QPIC + Hybrid GNN | {TARGET_IMAGE}",
        fill="black",
    )
    panel_draw.text(
        (10, 29),
        f"Best: {best_verb} | GNN={best_score:.4f} | "
        f"QPIC={qpic_score:.4f}",
        fill="black",
    )
    panel_draw.text(
        (10, 50), "Top 5 GNN-ranked candidates:", fill="black"
    )

    for i, idx in enumerate(ranked_indices[:5], start=1):
        candidate = candidates[idx]
        line = (
            f"{i}. {candidate['verb']} | "
            f"GNN={float(probabilities[idx]):.4f} | "
            f"QPIC={float(candidate['score']):.4f}"
        )
        panel_draw.text((10, 70 + (i - 1) * 14), line, fill="black")

    # Combine image and ranking panel.
    output_image = Image.new(
        "RGB", (width, height + panel_height), "white"
    )
    output_image.paste(image, (0, 0))
    output_image.paste(panel, (0, height))

    image_out = OUTPUT_DIR / "validation_image_4_visualization.jpg"
    output_image.save(image_out, quality=95)

    print("\nSaved visualization:", image_out)
    print("Saved candidate JSON:", json_path)
    print("Image dimensions:", image.size)
    print("Candidates scored:", len(candidates))
    print("\nTop 5 GNN-ranked interactions:")

    for item in results["results"][:5]:
        print(
            f"{item['rank']}. {item['verb']} | "
            f"GNN={item['gnn_score']:.4f} | "
            f"QPIC={item['qpic_score']:.4f} | "
            f"GT={item['ground_truth_label']}"
        )

    print("\nPASS: Visualization and JSON results saved.")


if __name__ == "__main__":
    main()
