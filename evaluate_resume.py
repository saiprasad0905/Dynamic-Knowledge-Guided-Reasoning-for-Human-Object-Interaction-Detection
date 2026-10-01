import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset

import util.misc as utils
from datasets.hico import HICODetection, make_hico_transforms
from engine import HICOEvaluator
from models import build_model


ROOT = Path("..") / "hoi_capstone" / "data" / "hico_20160224_det"
CHECKPOINT = Path("params") / "qpic_resnet50_hico.pth"
NUM_IMAGES = 9546
BATCH_SIZE = 2
CHUNK_SIZE = 500
CHUNK_DIR = Path("evaluation_chunks")


def main():
    args = SimpleNamespace(
        hoi=True,
        dataset_file="hico",
        num_obj_classes=80,
        num_verb_classes=117,
        backbone="resnet50",
        pretrained="",
        device="cpu",
        num_queries=100,
        aux_loss=True,
        masks=False,
        frozen_weights=None,
        position_embedding="sine",
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        dim_feedforward=2048,
        enc_layers=6,
        dec_layers=6,
        pre_norm=False,
        lr_backbone=1e-5,
        dilation=False,
        set_cost_class=1,
        set_cost_bbox=5,
        set_cost_giou=2,
        set_cost_obj_class=1,
        set_cost_verb_class=1,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        obj_loss_coef=1,
        verb_loss_coef=1,
        eos_coef=0.1,
        verb_loss_type="focal",
        subject_category_id=0,
    )

    print("Building QPIC model...")
    model, _, postprocessors = build_model(args)
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    print("Loading HICO-DET test dataset...")
    dataset = HICODetection(
        "val",
        ROOT / "images" / "test2015",
        ROOT / "annotations" / "test_hico.json",
        make_hico_transforms("val"),
        args.num_queries,
    )

    dataset.set_rare_hois(ROOT / "annotations" / "trainval_hico.json")
    dataset.load_correct_mat(ROOT / "annotations" / "corre_hico.npy")

    total_images = min(NUM_IMAGES, len(dataset))
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Total test images: {total_images}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk directory: {CHUNK_DIR}")

    total_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_index in range(total_chunks):

        start_index = chunk_index * CHUNK_SIZE
        end_index = min(start_index + CHUNK_SIZE, total_images)

        chunk_path = CHUNK_DIR / f"chunk_{chunk_index:04d}.pt"

        if chunk_path.exists():
            print(
                f"Chunk {chunk_index + 1}/{total_chunks} already exists: "
                f"{chunk_path}"
            )
            continue

        print()
        print(
            f"========== PROCESSING CHUNK "
            f"{chunk_index + 1}/{total_chunks} =========="
        )
        print(f"Image range: {start_index} - {end_index - 1}")

        subset = Subset(
            dataset,
            range(start_index, end_index),
        )

        sampler = SequentialSampler(subset)

        data_loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=0,
        )

        chunk_preds = []
        chunk_gts = []

        with torch.no_grad():

            for batch_index, (samples, targets) in enumerate(data_loader):

                samples = samples.to("cpu")

                outputs = model(samples)

                orig_target_sizes = torch.stack(
                    [target["orig_size"] for target in targets],
                    dim=0,
                )

                results = postprocessors["hoi"](
                    outputs,
                    orig_target_sizes,
                )

                chunk_preds.extend(results)
                chunk_gts.extend(copy.deepcopy(targets))

                processed = min(
                    (batch_index + 1) * BATCH_SIZE,
                    len(subset),
                )

                print(
                    f"Chunk progress: {processed}/{len(subset)}"
                )

        # Deduplicate by image ID, matching the official evaluation logic.
        img_ids = [target["id"] for target in chunk_gts]

        _, indices = __import__("numpy").unique(
            img_ids,
            return_index=True,
        )

        chunk_preds = [
            prediction
            for i, prediction in enumerate(chunk_preds)
            if i in indices
        ]

        chunk_gts = [
            target
            for i, target in enumerate(chunk_gts)
            if i in indices
        ]

        torch.save(
            {
                "preds": chunk_preds,
                "gts": chunk_gts,
                "start_index": start_index,
                "end_index": end_index,
            },
            chunk_path,
        )

        print(
            f"Saved chunk {chunk_index + 1}/{total_chunks}: "
            f"{chunk_path}"
        )
        print(
            f"Saved images: {len(chunk_gts)}"
        )

    print()
    print("========== LOADING ALL SAVED CHUNKS ==========")

    preds = []
    gts = []

    for chunk_index in range(total_chunks):

        chunk_path = CHUNK_DIR / f"chunk_{chunk_index:04d}.pt"

        if not chunk_path.exists():
            raise FileNotFoundError(
                f"Missing chunk file: {chunk_path}"
            )

        chunk = torch.load(
            chunk_path,
            map_location="cpu",
            weights_only=False,
        )

        preds.extend(chunk["preds"])
        gts.extend(chunk["gts"])

        print(
            f"Loaded chunk {chunk_index + 1}/{total_chunks}: "
            f"{len(chunk['gts'])} images"
        )

    print(f"Total predictions loaded: {len(preds)}")
    print(f"Total ground truths loaded: {len(gts)}")

    # Final safety deduplication by image ID.
    img_ids = [target["id"] for target in gts]

    _, indices = __import__("numpy").unique(
        img_ids,
        return_index=True,
    )

    preds = [
        prediction
        for i, prediction in enumerate(preds)
        if i in indices
    ]

    gts = [
        target
        for i, target in enumerate(gts)
        if i in indices
    ]

    print(f"After deduplication: {len(gts)} images")

    evaluator = HICOEvaluator(
        preds,
        gts,
        args.subject_category_id,
        dataset.rare_triplets,
        dataset.non_rare_triplets,
        dataset.correct_mat,
    )

    stats = evaluator.evaluate()

    print()
    print("========== QPIC FULL TEST EVALUATION ==========")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()





