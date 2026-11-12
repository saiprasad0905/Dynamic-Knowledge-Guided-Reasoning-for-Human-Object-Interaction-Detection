
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset

import util.misc as utils
from datasets.hico import HICODetection, make_hico_transforms
from models import build_model


ROOT = Path("..") / "hoi_capstone" / "data" / "hico_20160224_det"
CHECKPOINT = Path("params") / "qpic_resnet50_hico.pth"
OUTPUT_DIR = Path("training_evaluation_chunks_2000")

# Keep the first run small.

# Training inference configuration
NUM_IMAGES = 2000
BATCH_SIZE = 2
CHUNK_SIZE = 100



def build_args():
    return SimpleNamespace(
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


def main():
    print("Building QPIC model...")
    args = build_args()
    model, _, postprocessors = build_model(args)

    print("Loading checkpoint:", CHECKPOINT)
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to("cpu")
    model.eval()

    print("Loading training annotations...")
    dataset = HICODetection(
        "val",
        ROOT / "images" / "train2015",
        ROOT / "annotations" / "trainval_hico.json",
        make_hico_transforms("val"),
        args.num_queries,
    )

    print("Dataset records:", len(dataset))

    total_images = min(NUM_IMAGES, len(dataset))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Images selected:", total_images)
    print("Chunk size:", CHUNK_SIZE)
    print("Output directory:", OUTPUT_DIR.resolve())

    total_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_index in range(total_chunks):
        start_index = chunk_index * CHUNK_SIZE
        end_index = min(start_index + CHUNK_SIZE, total_images)

        output_path = OUTPUT_DIR / f"training_chunk_{chunk_index:04d}.pt"

        if output_path.exists():
            print("Skipping existing chunk:", output_path)
            continue

        print(
            f"\nProcessing chunk {chunk_index + 1}/{total_chunks}: "
            f"images {start_index} to {end_index - 1}"
        )

        subset = Subset(dataset, range(start_index, end_index))

        data_loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            sampler=SequentialSampler(subset),
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=0,
        )

        predictions = []
        ground_truths = []

        with torch.no_grad():
            for samples, targets in data_loader:
                samples = samples.to("cpu")
                outputs = model(samples)

                original_sizes = torch.stack(
                    [target["orig_size"] for target in targets],
                    dim=0,
                )

                results = postprocessors["hoi"](
                    outputs,
                    original_sizes,
                )

                predictions.extend(results)
                ground_truths.extend(copy.deepcopy(targets))

                print(
                    f"Chunk progress: "
                    f"{len(ground_truths)}/{len(subset)}"
                )

        if len(predictions) != len(ground_truths):
            raise RuntimeError(
                "Prediction and ground-truth counts do not match."
            )

        torch.save(
            {
                "preds": predictions,
                "gts": ground_truths,
                "start_index": start_index,
                "end_index": end_index,
            },
            output_path,
        )

        print("Saved:", output_path)
        print("Images saved:", len(ground_truths))

    print("\nTraining prediction generation finished.")
    print("Chunks expected:", total_chunks)




if __name__ == "__main__":
    main()
