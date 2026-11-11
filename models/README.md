# Model Artifacts

This directory contains the machine-learning artifacts used in the Hybrid Mental Health Prediction System.

## 1. Baseline Models

The following baseline models are tracked in this repository:

| File | Description |
|---|---|
| `baseline_rf_model.pkl` | Saved Random Forest baseline model. |
| `baseline_xgb_model.pkl` | Saved XGBoost baseline model. |

## 2. Deep Learning Checkpoints

The following large model artifacts are maintained separately and are not committed to this repository.

| Artifact | Description | Used for |
|---|---|---|
| `dual_tower_mental_health.pth` | Saved dual-tower model weights. | Notebook 05 experimental architecture. |
| `final_unified_model/` | RoBERTa sequence-classification checkpoint, including model weights, configuration and tokenizer files. | Text classification in the Streamlit application. |

## 3. Local Setup

To run the Streamlit application, obtain the `final_unified_model` checkpoint from the project's authorised storage and place the complete checkpoint directory here:

```text
models/
└── final_unified_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── vocab.json
    └── merges.txt
```

The checkpoint may contain additional configuration and tokenizer files. Preserve all files supplied with it.

The dual-tower checkpoint is a separate artifact and is required only when running the corresponding experiment.

## 4. Important Notes

- Large checkpoints are excluded from ordinary Git tracking to avoid unnecessarily increasing repository size.
- Do not rename or mix the dual-tower and text-classification checkpoints.
- Model performance claims should be interpreted in the context of the datasets and evaluation methodology described in the main README.
- The text-classification score is experimental and is not a clinically calibrated risk probability.

Refer to the main project README for architecture details, experimental results and limitations.
