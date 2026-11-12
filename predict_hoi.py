
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models._meta import _COCO_CATEGORIES

from datasets.hico import make_hico_transforms
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from build_hybrid_graph import build_knn_edges, spatial_features
from gnn_model import HybridHOIGNN, graph_to_tensors


ROOT = Path(__file__).resolve().parent
QPIC_CHECKPOINT = ROOT / "params" / "qpic_resnet50_hico.pth"
GNN_CHECKPOINT = (
    ROOT / "gnn_controlled_experiment" / "best_gnn_model_1900.pt"
)
VERB_MAPPING_PATH = (
    ROOT / "assets"
    / "verb_mapping.json"
)
CORRECT_MAT_PATH = (
    ROOT / "assets" / "corre_hico.npy"
)
OUTPUT_DIR = ROOT / "predictions"

VALID_OBJ_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90,
)


class Args:
    hoi = True
    dataset_file = "hico"
    num_obj_classes = 80
    num_verb_classes = 117
    backbone = "resnet50"
    pretrained = ""
    device = "cpu"
    num_queries = 100
    aux_loss = True
    masks = False
    frozen_weights = None
    position_embedding = "sine"
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    dim_feedforward = 2048
    enc_layers = 6
    dec_layers = 6
    pre_norm = False
    lr_backbone = 1e-5
    dilation = False
    backbone_name = "resnet50"
    train_backbone = True
    return_interm_layers = False
    set_cost_class = 1
    set_cost_bbox = 5
    set_cost_giou = 2
    set_cost_obj_class = 1
    set_cost_verb_class = 1
    bbox_loss_coef = 5
    giou_loss_coef = 2
    obj_loss_coef = 1
    verb_loss_coef = 1
    eos_coef = 0.1
    verb_loss_type = "focal"
    subject_category_id = 0


def load_models():
    if not QPIC_CHECKPOINT.is_file():
        raise FileNotFoundError(QPIC_CHECKPOINT)
    if not GNN_CHECKPOINT.is_file():
        raise FileNotFoundError(GNN_CHECKPOINT)

    print("Loading QPIC...")
    qpic, _, postprocessors = build_model(Args())
    checkpoint = torch.load(
        QPIC_CHECKPOINT, map_location="cpu", weights_only=False
    )
    qpic.load_state_dict(checkpoint["model"], strict=True)
    qpic.eval()

    print("Loading trained GNN...")
    saved = torch.load(
        GNN_CHECKPOINT, map_location="cpu", weights_only=False
    )
    config = saved["config"]
    gnn = HybridHOIGNN(
        num_object_categories=config["num_object_categories"],
        num_verbs=config["num_verbs"],
        hidden_dim=config["hidden_dim"],
    )
    gnn.load_state_dict(saved["model_state_dict"], strict=True)
    gnn.eval()

    return qpic, postprocessors, gnn, saved


def make_candidates(result, outputs, width, height, correct_mat, verb_mapping,
                    object_threshold, max_candidates):
    obj_prob = torch.softmax(outputs["pred_obj_logits"][0], dim=-1)
    obj_scores, _ = obj_prob[:, :-1].max(dim=-1)

    candidates = []
    humans = {}
    objects = {}
    pair_relations = {}

    labels = result["labels"]
    boxes = result["boxes"]
    sub_ids = result["sub_ids"]
    obj_ids = result["obj_ids"]
    verb_scores = result["verb_scores"]

    for query_id in range(len(verb_scores)):
        subject_index = int(sub_ids[query_id])
        object_index = int(obj_ids[query_id])
        subject_box = boxes[subject_index].tolist()
        object_box = boxes[object_index].tolist()

        if float(obj_scores[query_id]) < object_threshold:
            continue

        subject_category = int(labels[subject_index])
        object_category = int(labels[object_index])

        if not 0 <= object_category < len(VALID_OBJ_IDS):
            continue

        subject_box = boxes[subject_index].tolist()
        object_box = boxes[object_index].tolist()

        subject_box = [
            max(0.0, min(float(width), float(subject_box[0]))),
            max(0.0, min(float(height), float(subject_box[1]))),
            max(0.0, min(float(width), float(subject_box[2]))),
            max(0.0, min(float(height), float(subject_box[3]))),
        ]
        object_box = [
            max(0.0, min(float(width), float(object_box[0]))),
            max(0.0, min(float(height), float(object_box[1]))),
            max(0.0, min(float(width), float(object_box[2]))),
            max(0.0, min(float(height), float(object_box[3]))),
        ]

        if subject_box[0] >= subject_box[2] or subject_box[1] >= subject_box[3]:
            continue
        if object_box[0] >= object_box[2] or object_box[1] >= object_box[3]:
            continue

        # Keep the original QPIC query indices as node IDs.
        if subject_index not in humans:
            humans[subject_index] = {
                "node_id": f"h_{subject_index}",
                "category": 0,
                "bbox": subject_box,
            }
        if object_index not in objects:
            objects[object_index] = {
                "node_id": f"o_{object_index}",
                "category": object_category,
                "bbox": object_box,
            }

        for verb_index in range(verb_scores.shape[1]):
            # Match the object-verb compatibility filtering used in training.
            if correct_mat[verb_index, object_category] == 0:
                continue

            score = float(verb_scores[query_id, verb_index])
            verb_id = verb_index + 1
            mapping = verb_mapping.get(str(verb_id), {})
            verb_name = mapping.get("verb", "unknown")

            candidates.append({
                "query_id": query_id,
                "subject_id": subject_index,
                "object_id": object_index,
                "verb_id": verb_id,
                "verb": verb_name,
                "score": score,
                "label": 0,
            })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    candidates = candidates[:max_candidates]

    # Only nodes referenced by retained candidates belong in this graph.
    used_humans = {c["subject_id"] for c in candidates}
    used_objects = {c["object_id"] for c in candidates}
    humans = {k: v for k, v in humans.items() if k in used_humans}
    objects = {k: v for k, v in objects.items() if k in used_objects}

    for candidate in candidates:
        pair = (candidate["subject_id"], candidate["object_id"])
        if pair not in pair_relations:
            pair_relations[pair] = {
                "subject_id": pair[0],
                "object_id": pair[1],
                "verb_candidates": [],
            }
        pair_relations[pair]["verb_candidates"].append(candidate)

    hoi_edges = []
    for (subject_id, object_id), relation in pair_relations.items():
        hoi_edges.append({
            "subject_id": subject_id,
            "object_id": object_id,
            "spatial_features": spatial_features(
                humans[subject_id]["bbox"],
                objects[object_id]["bbox"],
                width,
                height,
            ),
            "verb_candidates": relation["verb_candidates"],
        })

    graph = {
        "image_width": width,
        "image_height": height,
        "humans": humans,
        "objects": objects,
        "human_edges": build_knn_edges(humans, 3, width, height),
        "object_edges": build_knn_edges(objects, 3, width, height),
        "hoi_edges": hoi_edges,
        "labeled_candidates": candidates,
    }

    return graph, obj_scores


def draw_prediction(image, graph, candidate, gnn_score, qpic_score,
                    object_score, output_path):
    width, height = image.size
    draw = ImageDraw.Draw(image)

    human = graph["humans"][candidate["subject_id"]]
    obj = graph["objects"][candidate["object_id"]]

    hbox = tuple(map(float, human["bbox"]))
    obox = tuple(map(float, obj["bbox"]))

    draw.rectangle(hbox, outline="#00D26A", width=4)
    draw.rectangle(obox, outline="#FF4545", width=4)

    object_model_id = int(obj["category"])
    coco_id = VALID_OBJ_IDS[object_model_id]
    object_name = _COCO_CATEGORIES[coco_id]

    verb = candidate["verb"].replace("_", " ")
    label = f"Person {verb} {object_name}"
    score_line = (
        f"GNN score: {gnn_score:.2%} | "
        f"QPIC score: {qpic_score:.2%}"
    )

    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    panel_width = min(width, max(text_width, 330) + 20)
    panel_height = text_height + 38

    # Put the label panel at the top so it remains readable.
    draw.rectangle(
        (0, 0, panel_width, panel_height),
        fill="#101820",
    )
    draw.text((10, 5), label, fill="white", font=font)
    draw.text(
        (10, text_height + 12),
        score_line,
        fill="#7CFFB2",
        font=small_font,
    )

    draw.text(
        (hbox[0], max(0, hbox[1] - 20)),
        "PERSON",
        fill="#00A854",
        font=small_font,
    )
    draw.text(
        (obox[0], max(0, obox[1] - 20)),
        object_name.upper(),
        fill="#FF4545",
        font=small_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)



def run_inference(
    image,
    qpic,
    postprocessors,
    gnn,
    gnn_checkpoint,
    correct_mat,
    verb_mapping,
    object_threshold=0.50,
    max_candidates=100,
    output_path=None,
):
    """Run QPIC + Hybrid GNN using already-loaded models."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")

    if not 0.0 <= object_threshold <= 1.0:
        raise ValueError("object_threshold must be between 0 and 1")

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")

    image = image.convert("RGB")
    original = image.copy()
    width, height = image.size

    transform = make_hico_transforms("val")
    image_tensor, _ = transform(image, None)
    samples = nested_tensor_from_tensor_list([image_tensor])

    with torch.inference_mode():
        outputs = qpic(samples)
        result = postprocessors["hoi"](
            outputs, torch.tensor([[height, width]])
        )[0]

        graph, obj_scores = make_candidates(
            result,
            outputs,
            width,
            height,
            correct_mat,
            verb_mapping,
            object_threshold,
            max_candidates,
        )

        if not graph["labeled_candidates"]:
            raise RuntimeError(
                "No compatible HOI candidates were generated. "
                "Try a lower object threshold."
            )

        tensors = graph_to_tensors(graph)
        probabilities = torch.sigmoid(gnn(tensors)).reshape(-1)

    if len(probabilities) != len(graph["labeled_candidates"]):
        raise RuntimeError("Candidate count and GNN output count differ.")

    if not torch.isfinite(probabilities).all():
        raise RuntimeError("GNN produced non-finite scores.")

    best_index = int(torch.argmax(probabilities))
    best = graph["labeled_candidates"][best_index]
    best_score = float(probabilities[best_index])

    if output_path is None:
        raise ValueError("output_path must be provided")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    object_query_score = float(obj_scores[best["query_id"]])

    draw_prediction(
        original,
        graph,
        best,
        best_score,
        float(best["score"]),
        object_query_score,
        output_path,
    )

    ranked = torch.argsort(probabilities, descending=True).tolist()
    results = []

    for rank, idx in enumerate(ranked, start=1):
        c = graph["labeled_candidates"][idx]
        object_model_id = graph["objects"][c["object_id"]]["category"]
        coco_id = VALID_OBJ_IDS[object_model_id]

        results.append({
            "rank": rank,
            "query_id": int(c["query_id"]),
            "subject_id": int(c["subject_id"]),
            "object_id": int(c["object_id"]),
            "object": _COCO_CATEGORIES[coco_id],
            "verb": c["verb"],
            "verb_id": int(c["verb_id"]),
            "qpic_score": float(c["score"]),
            "object_score": float(obj_scores[c["query_id"]]),
            "gnn_score": float(probabilities[idx]),
            "subject_bbox": graph["humans"][c["subject_id"]]["bbox"],
            "object_bbox": graph["objects"][c["object_id"]]["bbox"],
        })

    payload = {
        "image_width": width,
        "image_height": height,
        "qpic_checkpoint": str(QPIC_CHECKPOINT),
        "gnn_checkpoint": str(GNN_CHECKPOINT),
        "gnn_checkpoint_epoch": int(gnn_checkpoint["epoch"]),
        "candidate_count": len(results),
        "best_prediction": results[0],
        "ranked_predictions": results,
        "score_note": (
            "GNN score is sigmoid output, not a calibrated probability. "
            "QPIC score is the QPIC verb score, already combined with "
            "object confidence by the model postprocessor."
        ),
    }

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Run QPIC + trained Hybrid HOI GNN on one image."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--object-threshold", type=float, default=0.50)
    parser.add_argument("--max-candidates", type=int, default=100)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not 0.0 <= args.object_threshold <= 1.0:
        raise ValueError("--object-threshold must be between 0 and 1")

    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be at least 1")

    with open(VERB_MAPPING_PATH, "r", encoding="utf-8") as f:
        verb_mapping = json.load(f)

    correct_mat = np.load(CORRECT_MAT_PATH)

    qpic, postprocessors, gnn, gnn_checkpoint = load_models()

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_path = output_dir / f"{image_path.stem}_hoi.jpg"

    print(f"Running QPIC on {image_path} ({image.width}x{image.height})...")

    payload = run_inference(
        image=image,
        qpic=qpic,
        postprocessors=postprocessors,
        gnn=gnn,
        gnn_checkpoint=gnn_checkpoint,
        correct_mat=correct_mat,
        verb_mapping=verb_mapping,
        object_threshold=args.object_threshold,
        max_candidates=args.max_candidates,
        output_path=output_path,
    )

    payload["input_image"] = str(image_path)

    json_output = output_dir / f"{image_path.stem}_hoi.json"

    with json_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    best = payload["best_prediction"]

    print("\nTop interaction:")
    print(f"Person {best['verb'].replace('_', ' ')} {best['object']}")
    print(f"GNN score: {best['gnn_score']:.4f}")
    print(f"QPIC score: {best['qpic_score']:.4f}")
    print("\nSaved image:", output_path)
    print("Saved JSON:", json_output)


if __name__ == "__main__":
    main()


