# Dynamic Knowledge-Guided Reasoning for Human-Object Interaction Detection

A Capstone Project implementing **QPIC-based Human-Object Interaction (HOI) Detection** with an extension using **Hybrid Graph Neural Networks (GNNs)** for relational reasoning and candidate-level interaction scoring.

## 1. Project Overview

Human-Object Interaction (HOI) detection aims to identify interactions between humans and objects in an image, such as:

- Person riding a bicycle
- Person holding a cup
- Person sitting on a chair
- Person standing beside a motorcycle

This project uses **QPIC (Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information)** as its baseline HOI detection model and extends the pipeline with a Hybrid HOI Graph Neural Network.

The GNN uses graph-based relationships and interaction candidate features to refine the ranking of generated HOI candidates.

### Project Objectives

- Implement QPIC for human-object interaction detection.
- Generate and structure HOI predictions.
- Construct hybrid graphs representing human-object relationships.
- Apply GNN-based reasoning to HOI candidates.
- Investigate the effect of training-set size on GNN performance.
- Compare GNN candidate-level performance against a QPIC candidate baseline.
- Provide an interactive Streamlit interface for inference.

## 2. System Architecture

The project follows this processing pipeline:

```text
                 Input Image
                      |
                      v
               QPIC HOI Detector
                      |
                      v
             HOI Candidate Generation
                      |
                      v
             Structured HOI Predictions
                      |
                      v
              Hybrid Graph Builder
                      |
           +----------+----------+
           |                     |
           v                     v
      Human Nodes            Object Nodes
           |                     |
           +----------+----------+
                      |
                      v
              Graph Relationships
                      |
                      v
                GNN Model
                      |
                      v
          Refined HOI Candidate Scores
                      |
                      v
             Ranked HOI Predictions
```

### Main Components

**QPIC**

QPIC is the baseline HOI detection architecture. It uses a DETR-based object detection framework and transformer-based reasoning to generate human-object interaction predictions.

**Hybrid Graph Construction**

The graph construction stage represents human nodes, object nodes, spatial relationships, and HOI candidates.

**Graph Neural Network**

The GNN processes graph-based features and produces scores used to refine the ranking of HOI candidates.

**Streamlit Frontend**

The frontend provides an interface for submitting images and viewing HOI detection results.

## 3. Technologies Used

- Python
- PyTorch
- QPIC
- DETR
- ResNet-50
- Graph Neural Networks
- NumPy
- SciPy
- Streamlit
- HICO-DET dataset

## 4. Dataset

The project uses the **HICO-DET (Humans Interacting with Common Objects in Context)** dataset.

HICO-DET contains human-object interaction annotations across object categories and interaction classes.

The project uses the dataset for QPIC inference and HOI candidate generation, followed by graph dataset preparation and GNN experiments.

The complete dataset is not included in this GitHub repository. Users must obtain the dataset separately and configure the paths required by the QPIC implementation.

## 5. Repository Structure

```text
Dynamic-Knowledge-Guided-Reasoning/
│
├── Frontend_Streamlit/
│   ├── app.py
│   ├── hoi_adapter.py
│   ├── requirements-ui.txt
│   └── tests/
│
├── assets/
│   ├── corre_hico.npy
│   └── verb_mapping.json
│
├── datasets/
├── models/
├── util/
│
├── gnn_controlled_experiment/
│   ├── best_gnn_model_400.pt
│   ├── best_gnn_model_1900.pt
│   ├── controlled_evaluation.json
│   ├── image_split.json
│   └── training histories and evaluation results
│
├── gnn_experiment/
├── gnn_experiment_2000/
│
├── build_hybrid_graph.py
├── gnn_model.py
├── predict_hoi.py
│
├── train_gnn.py
├── train_gnn_2000.py
├── train_gnn_controlled.py
├── train_gnn_controlled_1900.py
│
├── evaluate_gnn.py
├── evaluate_gnn_2000.py
├── evaluate_gnn_controlled.py
│
├── main.py
├── engine.py
├── requirements.txt
├── CAPSTONE_RESULTS_REPORT.md
└── README.md
```

Additional scripts for data preparation, evaluation, validation, visualization, and graph analysis are included in the repository.

## 6. Installation and Setup

### Clone the repository

```bash
git clone https://github.com/saiprasad0905/Dynamic-Knowledge-Guided-Reasoning-for-Human-Object-Interaction-Detection.git

cd Dynamic-Knowledge-Guided-Reasoning-for-Human-Object-Interaction-Detection
```

### Install backend dependencies

```bash
pip install -r requirements.txt
```

### Install frontend dependencies

```bash
pip install -r Frontend_Streamlit/requirements-ui.txt
```

The dependency versions and environment should be checked against the requirements of the QPIC implementation and the installed PyTorch version.

## 7. Model Weights and Data Requirements

Large pretrained model weights and datasets are not included in this repository.

The pretrained QPIC checkpoint must be obtained separately and placed at:

```text
params/
└── qpic_resnet50_hico.pth
```

The project also contains trained GNN checkpoints in the experiment directories.

For the controlled experiment, the checkpoints include:

- `gnn_controlled_experiment/best_gnn_model_400.pt`
- `gnn_controlled_experiment/best_gnn_model_1900.pt`

Ensure that the required model weights, dataset files, and configured paths are available before running inference or reproducing experiments.

## 8. Running the Streamlit Application

From the repository root:

```bash
streamlit run Frontend_Streamlit/app.py
```

The application provides the interface for image-based HOI inference.

The required pretrained QPIC checkpoint and other configured model assets must be available for inference.

## 9. Training and Evaluation

The repository includes scripts for graph preparation, training, evaluation, and validation.

### GNN Training

The training scripts include:

```text
train_gnn.py
train_gnn_2000.py
train_gnn_controlled.py
train_gnn_controlled_1900.py
```

These correspond to the original and controlled training experiments.

### Evaluation

The evaluation scripts include:

```text
evaluate_gnn.py
evaluate_gnn_2000.py
evaluate_gnn_controlled.py
```

QPIC candidate-level baseline evaluation is implemented separately.

### Experiment Reproducibility

The controlled experiment uses a shared set of 100 validation images to compare models trained on 400 and 1,900 images.

The saved split, checkpoints, training histories, and evaluation results are available in:

```text
gnn_controlled_experiment/
```

The original experiments are stored separately in:

```text
gnn_experiment/
gnn_experiment_2000/
```

Refer to `CAPSTONE_RESULTS_REPORT.md` for the detailed experimental results and evaluation methodology.

## 10. Experimental Results

The project evaluates QPIC using official HICO-DET metrics and evaluates the GNN using candidate-level metrics.

These are different evaluation protocols and should not be directly compared.

### 10.1 Official QPIC HICO-DET Evaluation

| Metric | Result |
|---|---:|
| Official HICO-DET mAP | 0.292625 |
| Rare mAP | 0.220717 |
| Non-rare mAP | 0.314104 |
| Mean maximum recall | 0.632815 |

### 10.2 Controlled GNN Experiment

The controlled experiment compares GNN models trained using 400 and 1,900 images against the same QPIC candidate baseline.

Both models were evaluated on 10,000 candidates from the same 100 held-out validation images.

| Metric | QPIC Baseline | GNN: 400 Images | GNN: 1,900 Images |
|---|---:|---:|---:|
| Precision at 0.5 | 0.8421 | 0.4157 | 0.4691 |
| Recall at 0.5 | 0.0600 | 0.5330 | 0.6902 |
| F1 at 0.5 | 0.1120 | 0.4671 | 0.5586 |
| Accuracy at 0.5 | 0.7464 | 0.6758 | 0.7092 |
| Candidate-level AP | 0.3464 | 0.4106 | 0.5140 |

### 10.3 Main Findings

- The GNN trained on 400 images achieved candidate-level AP of 0.4106.
- Increasing the training set to 1,900 images improved candidate-level AP to 0.5140.
- F1 at threshold 0.5 increased from 0.4671 to 0.5586.
- Recall increased from 0.5330 to 0.6902.
- The QPIC candidate baseline achieved higher precision but substantially lower recall at threshold 0.5.

The controlled experiment therefore provides evidence that increasing the GNN training-set size improved candidate-level ranking and detection performance under the project's evaluation protocol.

### 10.4 Other GNN Experiments

The repository also includes:

- An original 500-image experiment.
- An original 2,000-image experiment.
- Associated training histories, checkpoints, and evaluation artifacts.

Their validation splits differ, so their results should not be interpreted as a controlled comparison.

## 11. Limitations

1. Candidate-level AP is not equivalent to official HICO-DET mAP.
2. Candidate labels depend on the annotation-matching rules used during graph dataset preparation.
3. The controlled experiment uses one fixed split and one training seed.
4. The observed improvements do not establish an improvement in official HICO-DET mAP.
5. Further repeated experiments would help establish the stability of the results.
6. Large datasets and pretrained weights must be obtained separately.

## 12. Conclusion

This project extends QPIC-based Human-Object Interaction detection with a Hybrid HOI Graph Neural Network.

The recorded experiments demonstrate improvements in candidate-level AP, F1, and recall when increasing the GNN training set from 400 to 1,900 images under a shared validation protocol.

The results support the potential of graph-based relational reasoning for refining HOI candidate predictions. However, the candidate-level improvements should not be interpreted as an improvement in official HICO-DET mAP.

## 13. References

**QPIC: Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information**

Masato Tamura, Hiroki Ohashi, and Tomoaki Yoshinaga. CVPR 2021.

Paper: https://arxiv.org/abs/2103.05399

Original QPIC implementation and architecture are retained as the foundation of this project.