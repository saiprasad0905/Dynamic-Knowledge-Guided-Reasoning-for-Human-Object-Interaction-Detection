import copy
from pathlib import Path

import torch

from datasets.hico import HICODetection, make_hico_transforms
from datasets.hico_eval import HICOEvaluator
from models import build_model


PROJECT_DIR = Path('..') / 'hoi_capstone'
DATA_ROOT = PROJECT_DIR / 'data' / 'hico_20160224_det'

IMAGE_NAMES = [
    'HICO_train2015_00000001.jpg',
    'HICO_train2015_00000002.jpg',
    'HICO_train2015_00000009.jpg'
]


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


dataset = HICODetection(
    'val',
    DATA_ROOT / 'images' / 'train2015',
    DATA_ROOT / 'annotations' / 'trainval_hico.json',
    make_hico_transforms('val'),
    args.num_queries
)

dataset.set_rare_hois(
    DATA_ROOT / 'annotations' / 'trainval_hico.json'
)

dataset.load_correct_mat(
    DATA_ROOT / 'annotations' / 'corre_hico.npy'
)


name_to_index = {
    annotation['file_name']: index
    for index, annotation in enumerate(dataset.annotations)
}


preds = []
gts = []

print('\n' + '=' * 60)
print('QPIC OFFICIAL-STYLE HICO EVALUATION')
print('=' * 60)

for image_name in IMAGE_NAMES:
    dataset_index = name_to_index[image_name]

    print(f'\nProcessing: {image_name}')
    print(f'Dataset index: {dataset_index}')

    image, target = dataset[dataset_index]

    samples = torch.stack([image])

    print('Running QPIC inference...')

    model.eval()

    with torch.no_grad():
        outputs = model(samples)

    target_sizes = torch.stack([
        target['orig_size']
    ])

    results = postprocessors['hoi'](
        outputs,
        target_sizes
    )

    preds.append(results[0])
    gts.append(copy.deepcopy(target))

    print('Inference successful!')
    print('HOI annotations:', target['hois'].shape[0])


print('\nRunning HICOEvaluator...')

evaluator = HICOEvaluator(
    preds,
    gts,
    args.subject_category_id,
    dataset.rare_triplets,
    dataset.non_rare_triplets,
    dataset.correct_mat
)

stats = evaluator.evaluate()

print('\n' + '=' * 60)
print('FINAL QPIC EVALUATION RESULTS')
print('=' * 60)

for key, value in stats.items():
    print(f'{key}: {value:.6f}')
