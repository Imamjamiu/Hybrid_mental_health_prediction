# Dataset Documentation

This directory contains the datasets used during the development and experimentation of the Hybrid Mental Health Prediction System.

The datasets are organised into raw, processed and split data.

## Directory Structure

```text
data/
├── raw/
│   ├── student_depression_dataset.csv
│   └── reddit_dataset.csv
│
├── processed/
│   ├── cleaned_student_dataset.csv
│   ├── cleaned_reddit_dataset.csv
│   └── combined_hybrid_student_reddit_data.csv
│
└── splits/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

## 1. Raw Data

The `raw/` directory contains the original downloaded datasets:

- **Student Depression Dataset:** Structured student information, including academic pressure, sleep duration, financial stress and depression labels.
- **Reddit Mental Health Dataset:** Textual records concerning mental-health-related experiences and emotional well-being.

Raw datasets are retained locally and are not committed to this repository.

## 2. Processed Data

The `processed/` directory contains datasets generated during preprocessing:

- `cleaned_student_dataset.csv` – cleaned structured student records.
- `cleaned_reddit_dataset.csv` – cleaned Reddit text records.
- `combined_hybrid_student_reddit_data.csv` – combined dataset used in the text and hybrid model experiments.

## 3. Dataset Splits

The `splits/` directory contains training, validation and testing datasets generated for model evaluation.

- `train.csv` – training examples.
- `validation.csv` – validation examples.
- `test.csv` – held-out testing examples.

These files are generated during experimentation and are not required to be committed to GitHub.

## 4. Data Preparation

The general data workflow is:

1. Obtain the original datasets from their respective sources.
2. Inspect and clean the student and Reddit datasets.
3. Prepare text representations and labels.
4. Combine the processed records for the relevant experiments.
5. Generate appropriate training, validation and testing splits when required.

Refer to the numbered notebooks in `notebooks/` for the corresponding development and experimentation workflow.

## 5. Data Privacy and Responsible Use

Mental-health-related information is sensitive.

- Original datasets are not included in this repository.
- Do not upload private, personally identifiable or otherwise restricted records.
- Follow the original dataset licences and terms of use.
- Do not redistribute downloaded datasets without the appropriate permission.
- Use the datasets strictly within their permitted research and educational purposes.

The use of these datasets does not establish clinical validity or justify making medical decisions from model predictions.
