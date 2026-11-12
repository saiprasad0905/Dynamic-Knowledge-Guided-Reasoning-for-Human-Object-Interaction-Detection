
# HOI Detection — Streamlit Frontend

Frontend for the QPIC + Hybrid GNN Human–Object Interaction Detection capstone project.

## Current Status

The application currently runs in **mock mode**.

- Image upload and preview are implemented.
- The mock adapter returns illustrative bounding boxes and predictions.
- The prediction table displays subject, action, object, and confidence.
- Automated adapter tests are available.
- Actual QPIC and Hybrid GNN inference has not been integrated.

Mock predictions must not be interpreted as real model results.

## Requirements

- Python 3.11
- pip
- A Windows, macOS, or Linux environment

## Setup on Windows

Open PowerShell in the project directory.

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements-ui.txt
```

### 4. Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, normally:

http://localhost:8501

## Run Tests

```powershell
python -m pytest tests -v
```

## Project Structure

```text
hoi_streamlit_frontend/
├── app.py
├── hoi_adapter.py
├── requirements-ui.txt
├── README.md
└── tests/
    └── test_hoi_adapter.py
```

The `.venv` directory is created locally and should not be transferred.

## Model Integration Plan

The frontend calls `predict_image(image)` from `hoi_adapter.py`.

The adapter must return a dictionary containing:

- `annotated_image`: a PIL image with annotations
- `predictions`: a list of interaction dictionaries
- `is_mock`: a Boolean indicating whether the output is simulated

Each interaction currently uses these fields:

- `subject`
- `action`
- `object`
- `confidence`

This is the frontend's proposed adapter contract, not a description of the native QPIC output.

Before real integration, inspect the existing model inference code and verify how QPIC and the Hybrid GNN produce predictions. Convert those outputs through the adapter without duplicating model-loading logic or changing training scripts and checkpoints.

The meaning of the confidence score must be confirmed during integration.

## Development Rules

- Keep UI code separate from model inference.
- Do not commit virtual environments, model checkpoints, datasets, or credentials.
- Keep paths configurable for different computers.
- Label mock predictions clearly.
- Validate actual inference independently before presenting model results.
