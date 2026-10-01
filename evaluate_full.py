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

    subset = Subset(dataset, range(min(NUM_IMAGES, len(dataset))))
    sampler = SequentialSampler(subset)

    data_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=0,
    )

    print(f"Evaluating first {len(subset)} test images...")
    preds = []
    gts = []

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

            preds.extend(results)
            gts.extend(copy.deepcopy(targets))

            print(
                f"Processed {min((batch_index + 1) * BATCH_SIZE, len(subset))}"
                f"/{len(subset)} images"
            )

    img_ids = [target["id"] for target in gts]
    _, indices = __import__("numpy").unique(img_ids, return_index=True)

    preds = [prediction for i, prediction in enumerate(preds) if i in indices]
    gts = [target for i, target in enumerate(gts) if i in indices]

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
    print("========== QPIC 100-IMAGE TEST EVALUATION ==========")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
