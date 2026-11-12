# Capstone Project Results Report

## 1. Project Overview

This project investigates Human-Object Interaction (HOI) detection using QPIC predictions and a Hybrid HOI Graph Neural Network (GNN).

The work includes:
- QPIC inference on HICO-DET.
- Structuring predictions into HOI candidates.
- Matching candidates against annotations to generate binary labels.
- Constructing hybrid graphs with human nodes, object nodes, spatial edges, and HOI candidates.
- Training and evaluating GNN models using different training-set sizes.
- Comparing GNN results with a QPIC candidate-level baseline.

## 2. Official HICO-DET Evaluation

The recorded official QPIC test evaluation produced:

| Metric | Result |
|---|---:|
| Official HICO-DET mAP | 0.292625 |
| Rare mAP | 0.220717 |
| Non-rare mAP | 0.314104 |
| Mean maximum recall | 0.632815 |

These figures are separate from the candidate-level metrics reported in the GNN experiments.

## 3. Original 500-Image GNN Experiment

### Dataset and split
- Images processed: 500
- HOI candidates: 50,000
- Positive candidates: 12,529
- Negative candidates: 37,471
- Training images: 400
- Validation images: 100

The graph dataset was verified, and the training/validation split was created using seed 42.

### Results

| Metric | QPIC baseline | GNN |
|---|---:|---:|
| Candidate AP | 0.3077 | 0.4088 |
| F1 at threshold 0.5 | 0.1551 | 0.4684 |

The GNN achieved a candidate AP increase of approximately 0.1011 over the QPIC baseline in this experiment.

The best recorded checkpoint was selected at epoch 8 using validation loss.

## 4. Original 2,000-Image GNN Experiment

### Dataset and split
- Images processed: 2,000
- HOI candidates: 200,000
- Positive candidates: 50,373
- Negative candidates: 149,627
- Training images: 1,600
- Validation images: 400

The graph dataset and graph tensor shapes were verified.

### Results

| Metric | QPIC baseline | GNN |
|---|---:|---:|
| Candidate AP | 0.3025 | 0.5213 |
| F1 at threshold 0.5 | 0.1302 | 0.5082 |

The best recorded checkpoint was selected at epoch 6 using validation loss.

These results use a different validation split from the original 500-image experiment, so the two experiments should not be interpreted as a controlled comparison.

## 5. Controlled Training-Set-Size Experiment

This experiment compares GNNs trained on 400 versus 1,900 images using the same 100 held-out validation images.

### Split verification

| Check | Result |
|---|---:|
| Validation images | 100 |
| Smaller training set | 400 |
| Larger training set | 1,900 |
| Small train/validation overlap | 0 |
| Large train/validation overlap | 0 |
| Small training set is a subset of large training set | Yes |
| Evaluation IDs match saved split | Yes |
| QPIC baseline IDs match saved split | Yes |

Both GNN models were evaluated on 10,000 candidates from the shared validation images. The QPIC baseline used the same validation candidates and labels.

### Results

| Metric | QPIC baseline | GNN: 400 images | GNN: 1,900 images |
|---|---:|---:|---:|
| Precision at 0.5 | 0.8421 | 0.4157 | 0.4691 |
| Recall at 0.5 | 0.0600 | 0.5330 | 0.6902 |
| F1 at 0.5 | 0.1120 | 0.4671 | 0.5586 |
| Accuracy at 0.5 | 0.7464 | 0.6758 | 0.7092 |
| Candidate-level AP | 0.3464 | 0.4106 | 0.5140 |
| Best checkpoint epoch | — | 8 | 18 |
| Best validation loss | — | 0.6151 | 0.5409 |

### Main finding

Increasing the GNN training set from 400 to 1,900 images improved candidate-level AP from approximately 0.4106 to 0.5140.

This is an absolute increase of approximately 0.1034, or a relative increase of approximately 25.2%.

At threshold 0.5, F1 increased from approximately 0.4671 to 0.5586, and recall increased from approximately 0.5330 to 0.6902.

The QPIC baseline retained higher precision at threshold 0.5, but its recall was substantially lower.

## 6. Interpretation and Limitations

1. Candidate-level AP is not official HICO-DET mAP. The metrics evaluate generated HOI candidates against the project's candidate-label matching rules.
2. Candidate labels may be affected by annotation completeness and the matching criteria used to assign positive and negative labels.
3. The controlled comparison uses one fixed split and one training seed. Additional seeds or repeated experiments would help establish the stability of the observed improvements.
4. The original 500-image and 2,000-image experiments use different validation splits and should not be compared as if they were controlled.
5. The controlled split has no image-level overlap between training and validation. This check does not independently establish that every upstream preprocessing step is free from leakage.
6. Checkpoint selection uses validation loss. Threshold and AP results should be interpreted in the context of the evaluation protocol used.
7. The controlled experiment's saved evaluation records include recomputed validation losses matching the saved best losses.

## 7. Reproducibility and Artifacts

The controlled experiment artifacts are stored in `gnn_controlled_experiment/`:

- `image_split.json`
- `best_gnn_model_400.pt`
- `training_history_400.json`
- `best_gnn_model_1900.pt`
- `training_history_1900.json`
- `controlled_evaluation.json`
- `qpic_baseline_controlled.json`
- `qpic_ap_controlled.json`

The original experiment artifacts are stored separately in `gnn_experiment/` and `gnn_experiment_2000/`.

The corresponding training, evaluation, split-generation, validation, and comparison scripts should be retained alongside the project source.

Before claiming full reproducibility, verify the required datasets, checkpoint files, Python environment, and input paths are available. Large datasets and model checkpoints require an explicit storage policy and should not automatically be committed to Git.

## 8. Conclusion

The recorded experiments show that the Hybrid HOI GNN improves candidate-level ranking and recall relative to the QPIC candidate baseline under the project's evaluation rules.

The controlled experiment provides evidence that increasing the training set from 400 to 1,900 images improves candidate-level AP and F1 on a shared validation set.

The findings are promising, but they do not by themselves establish an improvement in official HICO-DET mAP. Further repeated experiments and a clearly defined official benchmark evaluation would strengthen the conclusions.
