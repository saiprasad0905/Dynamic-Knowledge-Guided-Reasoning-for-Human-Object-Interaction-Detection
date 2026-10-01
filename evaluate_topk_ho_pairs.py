import copy
from pathlib import Path

import torch

from datasets.hico_eval import HICOEvaluator


CHUNK_DIR = Path("evaluation_chunks")

TOP_K_VALUES = [3, 5, 10, 15, 20]


def filter_prediction(prediction, top_k):
    """
    Keep the top-K QPIC query pairs according to
    the strongest verb score for each query.
    """

    verb_scores = prediction["verb_scores"]

    query_scores = verb_scores.max(dim=1).values

    _, indices = torch.topk(
        query_scores,
        k=min(top_k, len(query_scores)),
    )

    indices = indices.sort().values

    filtered = {
        "labels": prediction["labels"].clone(),
        "boxes": prediction["boxes"].clone(),
        "verb_scores": prediction["verb_scores"][indices].clone(),
        "sub_ids": prediction["sub_ids"][indices].clone(),
        "obj_ids": prediction["obj_ids"][indices].clone(),
    }

    return filtered


def evaluate_top_k(preds, gts, dataset_info, top_k):
    filtered_preds = [
        filter_prediction(
            prediction,
            top_k,
        )
        for prediction in preds
    ]

    evaluator = HICOEvaluator(
        filtered_preds,
        gts,
        dataset_info["subject_category_id"],
        dataset_info["rare_triplets"],
        dataset_info["non_rare_triplets"],
        dataset_info["correct_mat"],
    )

    return evaluator.evaluate()


def main():

    print("Loading evaluation chunks...")
    print("=" * 70)

    chunk_files = sorted(
        CHUNK_DIR.glob("chunk_*.pt")
    )

    if not chunk_files:
        raise FileNotFoundError(
            "No evaluation chunks found."
        )

    preds = []
    gts = []

    for chunk_file in chunk_files:

        print(
            f"Loading: {chunk_file.name}"
        )

        chunk = torch.load(
            chunk_file,
            map_location="cpu",
            weights_only=False,
        )

        preds.extend(chunk["preds"])
        gts.extend(chunk["gts"])

    print()
    print("Total predictions:", len(preds))
    print("Total ground truths:", len(gts))

    if len(preds) != len(gts):
        raise RuntimeError(
            "Prediction/ground-truth count mismatch."
        )

    # Load dataset metadata needed by HICOEvaluator.
    from pathlib import Path
    from types import SimpleNamespace

    from datasets.hico import (
        HICODetection,
        make_hico_transforms,
    )

    ROOT = (
        Path("..")
        / "hoi_capstone"
        / "data"
        / "hico_20160224_det"
    )

    args = SimpleNamespace(
        num_queries=100,
    )

    dataset = HICODetection(
        "val",
        ROOT / "images" / "test2015",
        ROOT / "annotations" / "test_hico.json",
        make_hico_transforms("val"),
        args.num_queries,
    )

    dataset.set_rare_hois(
        ROOT
        / "annotations"
        / "trainval_hico.json"
    )

    dataset.load_correct_mat(
        ROOT
        / "annotations"
        / "corre_hico.npy"
    )

    dataset_info = {
        "subject_category_id": 0,
        "rare_triplets": dataset.rare_triplets,
        "non_rare_triplets": dataset.non_rare_triplets,
        "correct_mat": dataset.correct_mat,
    }

    print()
    print("=" * 70)
    print("TOP-K H-O PAIR ABLATION")
    print("=" * 70)

    for top_k in TOP_K_VALUES:

        print()
        print(
            f"========== TOP-{top_k} =========="
        )

        stats = evaluate_top_k(
            preds,
            gts,
            dataset_info,
            top_k,
        )

        for key, value in stats.items():
            print(
                f"{key}: {value}"
            )


if __name__ == "__main__":
    main()