# 🧠 Hybrid Mental Health Prediction System

### A Dual-Tower Deep Learning Approach Integrating Structured Student Data and Mental Health Text

![Status](https://img.shields.io/badge/Status-Research%20Project-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Model](https://img.shields.io/badge/Language%20Model-MentalRoBERTa-green)

---

## 📖 Overview

The **Hybrid Mental Health Prediction System** is an experimental machine learning project that investigates how structured student information and unstructured mental-health-related text can be combined for depression-related classification.

The system explores a **dual-tower neural network architecture** in which two separate models process different input modalities:

- **Text tower:** Uses MentalRoBERTa to extract contextual representations from free-text descriptions.
- **Structured-data tower:** Uses a multilayer perceptron (MLP) to process numerical and categorical student information.

The representations from both towers are concatenated and passed through a classification network to produce a binary prediction.

The long-term objective is to develop a system that can accept both structured information and free-text descriptions, while investigating how multimodal learning can support mental-health-related screening research.

**Important:** The current experimental dataset uses synthetically paired records from separate student and Reddit datasets. These pairs do not represent genuine individuals whose structured information and text were collected together. Therefore, the reported results should not be interpreted as evidence of real-world predictive performance or clinical validity.

---

## 🎯 Project Objectives

- Develop a multimodal learning pipeline combining structured data and natural language.
- Explore MentalRoBERTa for contextual text representation.
- Build a neural network for structured student information.
- Implement a dual-tower architecture with feature-level fusion.
- Evaluate the experimental model using classification metrics.
- Investigate the challenges of combining independently collected datasets.
- Establish a foundation for future evaluation using genuinely paired multimodal data.

---

## 🏗️ System Architecture

The project uses a **Dual-Tower Architecture with Late Feature Fusion**.

Each modality is processed independently before the extracted representations are combined.

```text
             INPUT DATA
                  |
          +-------+-------+
          |               |
          v               v
    TEXT INPUT       STRUCTURED INPUT
          |               |
          v               v
    Tokenization     Imputation &
          |          Standardization
          v          One-Hot Encoding
    MentalRoBERTa          |
     Text Tower            v
          |           MLP Network
          |          Structured Tower
          |               |
          v               v
    Text Features    Structured Features
          |               |
          +-------+-------+
                  |
                  v
          FEATURE FUSION
           Concatenation
                  |
                  v
          CLASSIFICATION
             NETWORK
                  |
                  v
          BINARY PREDICTION
                  |
        +---------+---------+
        |                   |
        v                   v
  Non-depression      Depression-related
       label                label
```

### Architecture Components

**1. Text Tower**

- Pretrained model: `mental/mental-roberta-base`
- Extracts contextual text representations.
- Uses the first-token representation from the final hidden state.

**2. Structured-Data Tower**

- Processes numerical and categorical features.
- Uses imputation and preprocessing.
- Neural network layers: 64 and 32 units.
- ReLU activation and dropout.

**3. Feature Fusion**

- Concatenates the text and structured representations.
- Passes the combined representation into a classification network.

**4. Classification Network**

- Fully connected layers with 128 hidden units.
- ReLU activation and dropout.
- Produces two output classes.

---

## ✨ Key Features

- **Multimodal Learning:** Explores the combination of structured and unstructured information.
- **Contextual Text Representation:** Uses MentalRoBERTa to extract contextual language features.
- **Structured Feature Processing:** Uses a dedicated neural network for student information.
- **Feature-Level Fusion:** Combines representations from both modalities before classification.
- **Reproducible Experimentation:** Uses separate training and testing datasets.
- **Model Persistence:** Saves model weights, tokenizer, preprocessing pipeline, and configuration.
- **Evaluation:** Includes accuracy, precision, recall, F1-score, and confusion matrix analysis.

---

## 🧭 Repository Structure

```text
Hybrid_mental_health_prediction/
│
├── app/
│   ├── main.py
│   ├── recommender.py
│   ├── ui_components.py
│   └── utils.py
│
├── models/
│   ├── baseline_rf_model.pkl
│   ├── baseline_xgb_model.pkl
│   ├── hybrid_model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   ├── 01_data_synthesis.ipynb
│   ├── 02_merged_dataset.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_Dual_tower_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── model_factory.py
│   ├── preprocessor.py
│   ├── recommender.py
│   └── text_converter.py
│
├── .gitignore
└── README.md
```

### Notebook Descriptions

| Notebook | Purpose |
|---|---|
| `01_data_synthesis.ipynb` | Initial data synthesis and preparation |
| `02_merged_dataset.ipynb` | Dataset merging and text-data preparation |
| `03_eda_visualization.ipynb` | Exploratory data analysis and visualizations |
| `04_model_training.ipynb` | Earlier model training experiments |
| `05_Dual_tower_model.ipynb` | Dual-tower architecture, training, evaluation, and model persistence |

---

## 📊 Datasets and Methodology

The project explores two independently collected datasets.

### 1. Student Mental Health Dataset

Source: [Kaggle – Student Mental Health](https://www.kaggle.com/datasets/ahmadmideva/student-mental-health)

Contains structured information such as:

- Gender
- Age
- Academic Pressure
- Work Pressure
- CGPA
- Sleep Duration
- Dietary Habits
- Work/Study Hours
- Financial Stress
- Family History of Mental Illness

### 2. Reddit Mental Health Dataset

Source: [Kaggle – Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

Contains text records related to mental health, emotional experiences, and psychological distress.

### 3. Experimental Data Fusion

The two datasets were not originally collected from the same individuals.

For the current dual-tower experiment, records were synthetically paired by matching their existing binary labels. Student records were sampled with replacement to construct pairs with Reddit text records.

The datasets were separately divided into training and testing subsets before the pairing process.

**Methodological limitation:** Label-based synthetic pairing does not establish a genuine relationship between a student's structured information and a particular Reddit post. Since the existing labels influence pair construction, the experiment may introduce label-related bias and inflated evaluation results.

Consequently, this experiment is useful for investigating the implementation of a dual-tower architecture, but it does not establish real-world multimodal predictive performance.

---

## 🧪 Model Training and Evaluation

The current dual-tower experiment uses the following configuration:

| Component | Configuration |
|---|---|
| Text model | MentalRoBERTa |
| Structured tower | Multilayer Perceptron |
| Fusion method | Feature concatenation |
| Maximum text length | 128 tokens |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Loss function | CrossEntropyLoss |
| Training epochs | 3 |
| Data split | 80% training / 20% testing |
| Device | CUDA (when available) |

### Experimental Results

The latest Notebook 05 experiment produced the following results:

| Metric | Result |
|---|---:|
| Training Accuracy | 99.13% |
| Training Loss | 0.0271 |
| Test Accuracy | 98.33% |
| Test Samples | 8,480 |

### Confusion Matrix

| | Predicted Non-depression | Predicted Depression-related |
|---|---:|---:|
| Actual Non-depression | 3,203 | 66 |
| Actual Depression-related | 76 | 5,135 |

The results indicate strong classification performance on the constructed synthetic test pairs.

**Interpretation:** These metrics describe performance on synthetically paired records, not genuine person-level multimodal observations. They should not be interpreted as evidence of generalization to real students, clinical effectiveness, or diagnostic accuracy.

---

## 💾 Model Artifacts

Notebook 05 saves the following experimental artifacts:

| Artifact | Description |
|---|---|
| `hybrid_model.pth` | Trained dual-tower model weights |
| `structured_preprocessor.pkl` | Fitted structured-data preprocessing pipeline |
| Tokenizer directory | MentalRoBERTa tokenizer files |
| `model_config.pkl` | Model and preprocessing configuration |
| `confusion_matrix.png` | Experimental confusion matrix visualization |

The current Notebook 05 artifacts are saved to Google Drive during the experiment. They are not necessarily included in the repository's `models/` directory.

---

## ⚙️ Running the Experiments

The current dual-tower experiment was developed using Google Colab.

1. Clone or download this repository.
2. Open Google Colab.
3. Upload `notebooks/05_Dual_tower_model.ipynb`.
4. Mount Google Drive when prompted by the notebook.
5. Ensure the required datasets are available at the paths expected by the notebook.
6. Execute the notebook cells in order.

The notebook installs or imports the required dependencies and performs data preparation, model initialization, training, evaluation, and artifact saving.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Pretrained Model | MentalRoBERTa |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Visualization | Matplotlib |
| Model Persistence | Joblib, PyTorch |
| Development Environment | Google Colab, PyCharm |
| Version Control | Git, GitHub |

---

## 🚀 Future Improvements

- Evaluate the architecture using genuinely paired structured and text observations.
- Investigate alternative multimodal fusion strategies.
- Conduct ablation studies to measure the contribution of each modality.
- Evaluate performance on independent, appropriately collected test data.
- Improve model calibration and uncertainty estimation.
- Investigate explainability and evidence-grounded predictions.
- Improve the integration between the trained dual-tower model and the web application.
- Explore responsible deployment and human-in-the-loop review.

---

## 📝 License and Disclaimer

**License:** MIT License

This project is intended for educational and research purposes.

It is not a medical diagnostic system and must not be used as a substitute for professional mental healthcare, diagnosis, or treatment.

Mental health predictions can be sensitive and uncertain. Any future real-world application would require appropriate validation, privacy safeguards, ethical review, and professional oversight.

---

## 👥 Contributors

This project was initially developed collaboratively by:

- **imam Jamiu**
- **Robiu Olalere**

Both contributors are acknowledged for their work on the development of the Hybrid Mental Health Prediction System.

**Institution:** University of Ilorin, Nigeria

GitHub: [Imamjamiu](https://github.com/Imamjamiu)

