import torch
from pathlib import Path
from torchvision.ops import nms

from datasets.hico import HICODetection, make_hico_transforms
from models import build_model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_IMAGES = 100
OBJECT_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.50

DATA_ROOT = (
    Path("..")
    / "hoi_capstone"
    / "data"
    / "hico_20160224_det"
)

ANNOTATION_FILE = DATA_ROOT / "annotations" / "trainval_hico.json"
IMAGE_ROOT = DATA_ROOT / "images" / "train2015"
CHECKPOINT_PATH = Path("params") / "qpic_resnet50_hico.pth"

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# QPIC model configuration
# ---------------------------------------------------------

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


args = Args()

print("Building QPIC model...")
model, criterion, postprocessors = build_model(args)
model.to(DEVICE)

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

print("QPIC model loaded successfully.")


# ---------------------------------------------------------
# Load HICO-DET dataset
# ---------------------------------------------------------

dataset = HICODetection(
    "val",
    IMAGE_ROOT,
    ANNOTATION_FILE,
    make_hico_transforms("val"),
    args.num_queries,
)

print(f"HICO dataset loaded: {len(dataset)} images")
print(f"Evaluating first {NUM_IMAGES} images")


# ---------------------------------------------------------
# Process images and collect object predictions
# ---------------------------------------------------------

all_predictions = []
all_ground_truths = []

with torch.no_grad():
    for image_index in range(NUM_IMAGES):
        image_tensor, target = dataset[image_index]

        samples = image_tensor.unsqueeze(0).to(DEVICE)

        outputs = model(samples)

        target_sizes = torch.stack([target["orig_size"]])

        results = postprocessors["hoi"](outputs, target_sizes)
        result = results[0]

        object_scores = torch.softmax(
            outputs["pred_obj_logits"][0], dim=-1
        )[:, :-1].max(dim=-1).values.cpu()

        all_predictions.append({
            "image_index": image_index,
            "result": result,
            "object_scores": object_scores,
        })

        all_ground_truths.append(target)

        if (image_index + 1) % 10 == 0:
            print(f"Processed {image_index + 1}/{NUM_IMAGES} images")

print("Object prediction extraction complete.")


# ---------------------------------------------------------
# Helper: convert normalized GT boxes to original pixels
# ---------------------------------------------------------

def gt_boxes_to_original_pixels(target):
    return target["boxes"].clone()



# ---------------------------------------------------------
# Object detection evaluation helpers
# ---------------------------------------------------------

def box_iou(box_a, box_b):
    if len(box_a) == 0 or len(box_b) == 0:
        return torch.zeros((len(box_a), len(box_b)))

    area_a = (
        (box_a[:, 2] - box_a[:, 0]).clamp(min=0)
        * (box_a[:, 3] - box_a[:, 1]).clamp(min=0)
    )

    area_b = (
        (box_b[:, 2] - box_b[:, 0]).clamp(min=0)
        * (box_b[:, 3] - box_b[:, 1]).clamp(min=0)
    )

    top_left = torch.maximum(
        box_a[:, None, :2],
        box_b[None, :, :2]
    )

    bottom_right = torch.minimum(
        box_a[:, None, 2:],
        box_b[None, :, 2:]
    )

    wh = (bottom_right - top_left).clamp(min=0)

    intersection = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - intersection

    return intersection / union.clamp(min=1e-6)


def match_predictions(pred_boxes, pred_labels, gt_boxes, gt_labels):
    matched_predictions = set()
    matched_ground_truths = set()
    matched_ious = []

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matched_predictions, matched_ground_truths, matched_ious

    ious = box_iou(pred_boxes, gt_boxes)

    candidates = []

    for pred_index in range(len(pred_boxes)):
        for gt_index in range(len(gt_boxes)):
            if pred_labels[pred_index] != gt_labels[gt_index]:
                continue

            iou = float(ious[pred_index, gt_index])

            if iou >= 0.50:
                candidates.append((iou, pred_index, gt_index))

    candidates.sort(reverse=True)

    for iou, pred_index, gt_index in candidates:
        if pred_index in matched_predictions:
            continue

        if gt_index in matched_ground_truths:
            continue

        matched_predictions.add(pred_index)
        matched_ground_truths.add(gt_index)
        matched_ious.append(iou)

    return matched_predictions, matched_ground_truths, matched_ious

# ---------------------------------------------------------
# ---------------------------------------------------------
# Multi-threshold object detection evaluation
# ---------------------------------------------------------

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

valid_obj_ids = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90
)

for threshold in THRESHOLDS:

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_ious_all = []

    total_gt_objects = 0
    total_predictions_before_nms = 0
    total_predictions_after_nms = 0

    for prediction, target in zip(all_predictions, all_ground_truths):

        result = prediction["result"]
        object_scores = prediction["object_scores"]

        selected_queries = torch.where(
            object_scores >= threshold
        )[0]

        pred_boxes = []
        pred_labels = []
        pred_scores = []

        for query_id in selected_queries:

            query_id = int(query_id)
            object_id = int(result["obj_ids"][query_id])
            model_label = int(result["labels"][object_id])
            score = float(object_scores[query_id])
            box = result["boxes"][object_id].float()

            pred_boxes.append(box)
            pred_labels.append(model_label)
            pred_scores.append(score)

        total_predictions_before_nms += len(pred_boxes)

        gt_boxes = gt_boxes_to_original_pixels(target)
        gt_labels = target["labels"].clone()
        total_gt_objects += len(gt_boxes)

        if len(pred_boxes) == 0:
            false_negatives += len(gt_boxes)
            continue

        pred_boxes = torch.stack(pred_boxes)
        pred_labels = torch.tensor(pred_labels, dtype=torch.long)
        pred_scores = torch.tensor(pred_scores, dtype=torch.float32)

        kept_indices = []

        for class_id in pred_labels.unique():

            class_indices = torch.where(
                pred_labels == class_id
            )[0]

            keep = nms(
                pred_boxes[class_indices],
                pred_scores[class_indices],
                NMS_IOU_THRESHOLD
            )

            kept_indices.extend(class_indices[keep].tolist())

        kept_indices = torch.tensor(
            kept_indices,
            dtype=torch.long
        )

        pred_boxes = pred_boxes[kept_indices]
        pred_labels = pred_labels[kept_indices]

        total_predictions_after_nms += len(pred_boxes)

        matched_predictions, matched_ground_truths, matched_ious = (
            match_predictions(
                pred_boxes,
                pred_labels,
                gt_boxes,
                gt_labels
            )
        )

        tp = len(matched_predictions)

        true_positives += tp
        false_positives += len(pred_boxes) - tp
        false_negatives += len(gt_boxes) - len(matched_ground_truths)

        matched_ious_all.extend(matched_ious)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )

    average_iou = (
        sum(matched_ious_all) / len(matched_ious_all)
        if matched_ious_all
        else 0.0
    )

    print()
    print(f"========== THRESHOLD {threshold:.2f} ==========")
    print(f"GT objects                : {total_gt_objects}")
    print(f"Predictions before NMS    : {total_predictions_before_nms}")
    print(f"Predictions after NMS     : {total_predictions_after_nms}")
    print(f"True positives            : {true_positives}")
    print(f"False positives           : {false_positives}")
    print(f"False negatives           : {false_negatives}")
    print(f"Precision                 : {precision:.4f}")
    print(f"Recall                    : {recall:.4f}")
    print(f"Average IoU               : {average_iou:.4f}")


