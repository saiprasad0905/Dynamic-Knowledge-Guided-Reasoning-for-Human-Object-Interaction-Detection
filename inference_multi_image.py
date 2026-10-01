
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models._meta import _COCO_CATEGORIES

from datasets.hico import make_hico_transforms
from models import build_model
from util.misc import nested_tensor_from_tensor_list


class Args:
    hoi = True
    dataset_file = 'hico'
    num_obj_classes = 80
    num_verb_classes = 117
    backbone = 'resnet50'
    pretrained = ''
    device = 'cpu'
    num_queries = 100
    aux_loss = True
    masks = False
    frozen_weights = None
    position_embedding = 'sine'
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    dim_feedforward = 2048
    enc_layers = 6
    dec_layers = 6
    pre_norm = False
    lr_backbone = 1e-5
    dilation = False
    backbone_name = 'resnet50'
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
    verb_loss_type = 'focal'
    subject_category_id = 0


def run_inference(image_path, model, postprocessors, transform, valid_obj_ids, verb_mapping):
    print(f'\nProcessing: {image_path.name}')

    image = Image.open(image_path).convert('RGB')
    print('Image size:', image.size)

    image_tensor, _ = transform(image, None)
    samples = nested_tensor_from_tensor_list([image_tensor])

    print('Running QPIC inference...')

    model.eval()

    with torch.no_grad():
        outputs = model(samples)

    print('Inference successful!')

    target_sizes = torch.tensor([[image.height, image.width]])

    with torch.no_grad():
        results = postprocessors['hoi'](outputs, target_sizes)

    result = results[0]

    print('Post-processing successful!')

    return outputs, result

args = Args()
print('Building QPIC model...')

model, criterion, postprocessors = build_model(args)
model.to('cpu')

checkpoint_path = Path('params/qpic_resnet50_hico.pth')

print('Loading checkpoint:', checkpoint_path)

checkpoint = torch.load(
    checkpoint_path,
    map_location='cpu',
    weights_only=False
)

missing, unexpected = model.load_state_dict(
    checkpoint['model'],
    strict=False
)

print('Missing keys   :', len(missing))
print('Unexpected keys:', len(unexpected))
transform = make_hico_transforms('val')

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

object_threshold = 0.50

image_paths = [
    Path('..') / 'hoi_capstone' / 'data' / 'hico_20160224_det' / 'images' / 'train2015' / 'HICO_train2015_00000001.jpg',
    Path('..') / 'hoi_capstone' / 'data' / 'hico_20160224_det' / 'images' / 'train2015' / 'HICO_train2015_00000002.jpg',
    Path('..') / 'hoi_capstone' / 'data' / 'hico_20160224_det' / 'images' / 'train2015' / 'HICO_train2015_00000009.jpg'
]

with open(
        Path('..') / 'hoi_capstone' / 'data' / 'processed' / 'verb_mapping.json',
        'r',
        encoding='utf-8'
) as file:
    verb_mapping = json.load(file)



for image_path in image_paths:
    outputs, result = run_inference(
        image_path,
        model,
        postprocessors,
        transform,
        valid_obj_ids,
        verb_mapping
    )

    obj_prob = torch.softmax(
        outputs['pred_obj_logits'][0],
        dim=-1
    )

    obj_scores, obj_labels = obj_prob[:, :-1].max(dim=-1)

    selected_queries = torch.where(
        obj_scores >= object_threshold
    )[0]

    print('Object threshold:', object_threshold)
    print('Selected queries:', selected_queries.tolist())


    print('\nTop predicted interactions:')

    for query_id in selected_queries:
        query_id = int(query_id)

        verb_scores = result['verb_scores'][query_id]
        top_verbs = verb_scores.argsort(descending=True)[:5]

        for rank, verb_index in enumerate(top_verbs):
            verb_id = int(verb_index) + 1
            mapping = verb_mapping[str(verb_id)]

            print(
                f'  {rank + 1}. '
                f'verb_id={verb_id}: '
                f'{mapping["object"]} + {mapping["verb"]} '
                f'score={float(verb_scores[verb_index]):.4f}'
            )
