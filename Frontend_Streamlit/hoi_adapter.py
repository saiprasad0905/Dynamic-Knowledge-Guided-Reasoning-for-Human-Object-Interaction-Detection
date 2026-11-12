
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
QPIC_DIR = ROOT.parent

if str(QPIC_DIR) not in sys.path:
    sys.path.insert(0, str(QPIC_DIR))

from predict_hoi import (
    load_models,
    run_inference,
    VERB_MAPPING_PATH,
    CORRECT_MAT_PATH,
)


@st.cache_resource(show_spinner="Loading QPIC and trained GNN models...")
def get_model_resources():
    """Load models and supporting data once per Streamlit process."""

    qpic, postprocessors, gnn, gnn_checkpoint = load_models()

    with open(VERB_MAPPING_PATH, "r", encoding="utf-8") as f:
        verb_mapping = json.load(f)

    correct_mat = np.load(CORRECT_MAT_PATH)

    return (
        qpic,
        postprocessors,
        gnn,
        gnn_checkpoint,
        correct_mat,
        verb_mapping,
    )


def predict_image(
    image: Image.Image,
    object_threshold: float = 0.50,
    max_candidates: int = 100,
) -> dict:
    """Run QPIC + Hybrid GNN inference using cached models."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")

    if image.width < 2 or image.height < 2:
        raise ValueError("Image dimensions are too small")

    if not 0.0 <= object_threshold <= 1.0:
        raise ValueError("Object threshold must be between 0 and 1")

    if max_candidates < 1:
        raise ValueError("Maximum candidates must be at least 1")

    image = image.convert("RGB")

    (
        qpic,
        postprocessors,
        gnn,
        gnn_checkpoint,
        correct_mat,
        verb_mapping,
    ) = get_model_resources()

    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="hoi_result_") as temp_dir:
        output_path = Path(temp_dir) / "annotated.jpg"

        payload = run_inference(
            image=image,
            qpic=qpic,
            postprocessors=postprocessors,
            gnn=gnn,
            gnn_checkpoint=gnn_checkpoint,
            correct_mat=correct_mat,
            verb_mapping=verb_mapping,
            object_threshold=object_threshold,
            max_candidates=max_candidates,
            output_path=output_path,
        )

        with Image.open(output_path) as result_image:
            annotated_image = result_image.convert("RGB").copy()

    inference_seconds = time.perf_counter() - start_time

    predictions = []

    for item in payload.get("ranked_predictions", [])[:20]:
        predictions.append({
            "rank": item["rank"],
            "subject": "person",
            "action": item["verb"].replace("_", " "),
            "object": item["object"],
            "gnn_score": round(item["gnn_score"], 4),
            "qpic_score": round(item["qpic_score"], 4),
            "object_score": round(item["object_score"], 4),
        })

    return {
        "annotated_image": annotated_image,
        "predictions": predictions,
        "payload": payload,
        "inference_seconds": round(inference_seconds, 2),
        "object_threshold": object_threshold,
        "max_candidates": max_candidates,
        "is_mock": False,
        "score_note": payload.get("score_note", ""),
    }

