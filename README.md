# 🧠 Hybrid Mental Health Prediction System

### A Dual-Tower Deep Learning Approach Integrating Structured Student Data and Mental Health Text

![Status](https://img.shields.io/badge/Status-Research%20Project-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Model](https://img.shields.io/badge/Language%20Model-MentalRoBERTa-green)

---

## 📖 Overview

The **Hybrid Mental Health Prediction System** is an experimental machine learning project that investigates how structured student information and unstructured mental-health-related text can be combined for depression-related classification.

The research explores a **dual-tower neural network architecture** in which two separate models process different input modalities:

- **Text tower:** Uses MentalRoBERTa to extract contextual representations from free-text descriptions.
- **Structured-data tower:** Uses a multilayer perceptron (MLP) to process numerical and categorical student information.

The representations from both towers are concatenated and passed through a classification network to produce a binary prediction.

The project has also been extended with a Streamlit application that integrates a RoBERTa-based text classification model, a support resource recommendation module, and an independent crisis-support mechanism.

The long-term objective is to investigate how multimodal learning can support mental-health-related screening research while providing access to relevant educational and support resources.

**Important:** The experimental multimodal dataset uses synthetically paired records from separate student and Reddit datasets. These pairs do not represent genuine individuals whose structured information and text were collected together. Consequently, the reported results should not be interpreted as evidence of real-world predictive performance or clinical validity.

---

## 🎯 Project Objectives

- Develop a multimodal learning pipeline combining structured data and natural language.
- Explore MentalRoBERTa for contextual text representation.
- Build a neural network for structured student information.
- Implement a dual-tower architecture with feature-level fusion.
- Investigate unified architectures for combining independently collected datasets.
- Evaluate experimental models using classification metrics.
- Develop a text-based mental health classification application.
- Implement a context-aware support resource recommendation system.
- Integrate Nigerian emergency, mental health, and student-support resources.
- Investigate the challenges of combining independently collected datasets.
- Establish a foundation for future evaluation using genuinely paired multimodal data.

---

## 🏗️ System Architecture

The research project uses a **Dual-Tower Architecture with Late Feature Fusion**.

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
          |             MLP Network
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
         +--------+--------+
         |                 |
         v                 v
   Non-depression    Depression-related
        label              label
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

### Application Architecture

The Streamlit application is an additional component of the project. It currently uses a fine-tuned RoBERTa-based sequence classification model for journal-text analysis.

The application also integrates a support resource recommender and a separate crisis-support retrieval mechanism.

The deployed text classifier should not be confused with the experimental dual-tower model developed in the research notebooks.

---

## ✨ Key Features

- **Multimodal Learning:** Explores the combination of structured and unstructured information.
- **Contextual Text Representation:** Uses MentalRoBERTa to extract contextual language features.
- **Structured Feature Processing:** Uses a dedicated neural network for student information.
- **Feature-Level Fusion:** Combines representations from both modalities before classification.
- **Unified Architecture Experiments:** Investigates combining independently collected student and Reddit datasets.
- **Text Classification Application:** Provides journal-text classification through a Streamlit interface.
- **Support Resource Recommendation:** Retrieves potentially relevant educational, academic, and mental health support resources.
- **Crisis-Support Integration:** Provides a separate retrieval pathway for designated crisis-support contacts.
- **Reproducible Experimentation:** Uses separate training, validation, and testing datasets.
- **Model Persistence:** Saves model weights, tokenizer, preprocessing pipeline, and configuration.
- **Evaluation:** Includes accuracy, precision, recall, F1-score, and confusion matrix analysis.

---

## 🧭 Repository Structure

```text
Hybrid_mental_health_prediction/
│
├── app/
│   └── main.py
│
├── models/
│   ├── baseline_rf_model.pkl
│   ├── baseline_xgb_model.pkl
│   └── final_unified_model/
│
├── notebooks/
│   ├── 01_data_synthesis.ipynb
│   ├── 02_merged_dataset.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_Dual_tower_model.ipynb
│   └── 06_Unified_architecture.ipynb
│
├── src/
│   ├── classifier.py
│   ├── recommender.py
│   └── resources.csv
│
├── label.py
├── README.md
└── .gitignore
```

### Notebook Descriptions

| Notebook | Purpose |
|---|---|
| `01_data_synthesis.ipynb` | Initial data synthesis and preparation |
| `02_merged_dataset.ipynb` | Dataset merging and text-data preparation |
| `03_eda_visualization.ipynb` | Exploratory data analysis and visualizations |
| `04_model_training.ipynb` | Baseline model training experiments |
| `05_Dual_tower_model.ipynb` | Dual-tower architecture, training, evaluation, and model persistence |
| `06_Unified_architecture.ipynb` | Unified dataset preparation, architecture experiments, training, and source-specific evaluation |

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

For the multimodal experiments, records were synthetically paired by matching their existing binary labels. Student records were sampled with replacement to construct pairs with Reddit text records.

The datasets are processed separately before being combined for the experimental architecture.

The unified architecture experiment uses source-aware stratification to distribute records across training, validation, and testing subsets while preserving the dataset source and binary label proportions.

The latest Notebook 06 experiment reports the following dataset distribution:

| Dataset Split | Records |
|---|---:|
| Training | 49,209 |
| Validation | 10,545 |
| Testing | 10,545 |
| **Total** | **70,299** |

**Methodological limitations:**

- Label-based synthetic pairing does not establish a genuine relationship between a student's structured information and a particular Reddit post.
- Student records sampled with replacement may introduce repeated structured observations.
- The positive-class definitions differ between the datasets.
- Source-specific differences may influence the model's predictions.

The student dataset's positive class represents depression-related information, whereas the Reddit dataset's positive class combines depression-related and suicidal-related content.

Consequently, treating these labels as identical introduces a label-harmonisation limitation.

These experiments are useful for investigating multimodal architectures and dataset integration, but they do not establish real-world multimodal predictive performance.

---

## 🧪 Model Training and Evaluation

The project investigates different machine learning and deep learning approaches.

### 1. Baseline Models

The repository includes baseline experiments involving:

- Random Forest
- XGBoost

These models provide traditional machine learning reference points for the broader research.

### 2. Dual-Tower Model

The dual-tower experiment uses the following configuration:

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

### 3. Dual-Tower Experimental Results

The earlier Notebook 05 experiment produced the following results:

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

These results describe performance on the constructed synthetic test pairs and should not be interpreted as evidence of clinical effectiveness or generalization to genuine multimodal observations.

### 4. Unified Architecture Experimental Results

Notebook 06 investigates a further unified architecture using the combined dataset.

The latest saved evaluation outputs report:

| Evaluation Group | Accuracy |
|---|---:|
| Overall Test Set | 90.25% |
| Student Dataset | 77.85% |
| Reddit Dataset | 98.41% |

The source-specific results are reported separately to examine differences in performance across the two independently collected datasets.

The substantial difference between student and Reddit performance highlights the importance of dataset characteristics, label definitions, and source-specific generalization.

**Interpretation:** These results are experimental and are affected by synthetic pairing, differences in target definitions, and potential dataset-specific patterns. They should not be interpreted as evidence of clinical validity or real-world diagnostic performance.

---

## 💾 Model Artifacts

The project uses different model artifacts for its research experiments and application.

### Research Model Artifacts

| Artifact | Description |
|---|---|
| `hybrid_model.pth` | Trained dual-tower model weights |
| `structured_preprocessor.pkl` | Fitted structured-data preprocessing pipeline |
| Tokenizer directory | MentalRoBERTa tokenizer files |
| `model_config.pkl` | Model and preprocessing configuration |
| `confusion_matrix.png` | Experimental confusion matrix visualization |

The dual-tower experiment saves its artifacts to Google Drive during execution. They are not necessarily included in the public repository.

### Application Model Artifacts

The Streamlit application uses a fine-tuned RoBERTa-based sequence classification checkpoint stored in:

`models/final_unified_model/`

This checkpoint is separate from the experimental dual-tower model.

Large model checkpoints may need to be obtained or generated separately rather than downloaded directly from the repository.

---

## ⚙️ Running the Experiments

The research experiments were developed using Google Colab and PyCharm.

### Dual-Tower Experiment

1. Clone or download this repository.
2. Open Google Colab.
3. Upload `notebooks/05_Dual_tower_model.ipynb`.
4. Mount Google Drive when prompted by the notebook.
5. Ensure the required datasets are available at the paths expected by the notebook.
6. Execute the notebook cells in order.

### Unified Architecture Experiment

1. Open `notebooks/06_Unified_architecture.ipynb`.
2. Ensure the required datasets are available.
3. Verify the configured dataset paths.
4. Execute the notebook cells in order.
5. Review the training, validation, and source-specific evaluation outputs.

The notebooks install or import the required dependencies and perform data preparation, model initialization, training, evaluation, and artifact saving.

### Running the Streamlit Application

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

Windows activation:

```bash
venv\Scripts\activate
```

Install the core application dependencies:

```bash
pip install streamlit torch transformers pandas numpy scikit-learn
```

Ensure the application model checkpoint is available in:

```text
models/final_unified_model/
```

Launch the application:

```bash
streamlit run app/main.py
```

---

## 🧠 Support Resource Recommendation

The project includes a support resource recommendation component implemented in `src/recommender.py`.

It uses transformer-based text representations and cosine similarity to compare journal entries with available support resources.

Resource descriptions and tags are combined to construct resource representations.

The general recommendation mechanism:

1. Receives the user's journal entry.
2. Generates a text representation.
3. Compares it with available resource representations.
4. Calculates cosine similarity.
5. Retrieves up to three eligible resources.

The resource catalogue is maintained in `src/resources.csv`.

It includes:

- Mental health educational resources.
- Academic support resources.
- Financial assistance information.
- Sleep and stress management resources.
- Depression-related self-care resources.
- Nigerian crisis-support contacts.

The recommendation mechanism also applies resource eligibility restrictions. For example, resources intended exclusively for National Open University of Nigeria students are excluded from general recommendations.

**Current limitation:** The existing transformer representation was developed using a sequence classification model rather than a model specifically optimised for semantic retrieval. Recommendation relevance therefore requires further evaluation and improvement.

---

## 🆘 Crisis-Support Integration

The application includes an independent crisis-support mechanism for users who explicitly disclose thoughts of self-harm.

This mechanism is separate from ordinary semantic recommendations and does not depend on the classifier's predicted score.

The dedicated crisis resource identifiers are:

- C001
- C002
- C003
- C004
- C005

The resources include Nigerian emergency and mental health support contacts.

The system is designed to make relevant crisis-support information accessible when users need additional assistance.

**Important:** The application does not replace emergency services, professional mental healthcare, or clinical intervention. Contact availability should be verified periodically.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Pretrained Model | MentalRoBERTa, RoBERTa |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Similarity Search | Cosine Similarity |
| Application Framework | Streamlit |
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
- Improve label harmonisation across datasets.
- Investigate class weighting and its impact on minority-class performance.
- Improve model calibration and uncertainty estimation.
- Investigate explainability and evidence-grounded predictions.
- Develop hybrid semantic and keyword-based resource retrieval.
- Introduce relevance thresholds and recommendation evaluation metrics.
- Improve resource eligibility and geographical filtering.
- Strengthen crisis-support resource verification.
- Improve the integration between the trained multimodal model and the web application.
- Introduce automated testing and model evaluation pipelines.
- Explore responsible deployment and human-in-the-loop review.

---

## 📝 License and Disclaimer

**License:** MIT License (subject to the repository's actual LICENSE file).

This project is intended for educational and research purposes.

It is not a medical diagnostic system and must not be used as a substitute for professional mental healthcare, diagnosis, or treatment.

Mental health predictions can be sensitive and uncertain. Any future real-world application would require appropriate validation, privacy safeguards, ethical review, and professional oversight.

---

## 👨‍💻 Author

**Muhammad Jamiu Imam-Fulani**

Sole Author and Developer

This project was independently researched, developed, implemented, and maintained by the author.

**Institution:** University of Ilorin, Nigeria

GitHub: [@Imamjamiu](https://github.com/Imamjamiu)

Repository: [Hybrid Mental Health Prediction System](https://github.com/Imamjamiu/Hybrid_mental_health_prediction)

---

*Exploring responsible applications of artificial intelligence, multimodal learning, and natural language processing in mental health research.*
