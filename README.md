# Smart MCQ Solver Challenge


 **Live Demo:**  
https://dl-24f1002360-t22026-deploy.streamlit.app/

---

# Overview

The **Smart MCQ Solver Challenge** aims to automatically answer multiple-choice questions by predicting the **Top-3 most likely correct options** from five candidate answers (A–E).

The project compares three different machine learning paradigms under a common preprocessing and evaluation pipeline:

- Traditional Machine Learning
- Transformer Fine-Tuning
- Deep Learning from Scratch

The official evaluation metric is **MAP@3 (Mean Average Precision @ 3)**.

The final solution is deployed as an interactive **Streamlit Web Application**.

---

#  Features

-  Predicts the **Top-3 most probable answers**
-  Interactive Streamlit Web Application
-  Multiple model comparison
-  Deep Learning & Transformer models
-  Performance visualization
-  Kaggle submission generation
-  Experiment tracking using Weights & Biases

---

#  Models Implemented

| Model | Category |
|--------|----------|
| TF-IDF + Logistic Regression | Traditional Machine Learning |
| TF-IDF + XGBoost | Traditional Machine Learning |
| Soft Voting Ensemble | Traditional Machine Learning |
| DeBERTa-v3-base | Transformer Fine-Tuning |
| **BiLSTM (Implemented Completely from Scratch)** | Deep Learning |

---

#  Final Selected Model

After evaluating all models, the **BiLSTM (Built Completely from Scratch)** was selected as the final submission model.

### Why BiLSTM?

-  Highest Kaggle Leaderboard Score
-  Best Overall Accuracy
-  Highest Precision
-  Highest Recall
-  Best F1 Score
-  Competitive MAP@3 Performance

---

#  Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | MAP@3 |
|------|---------:|----------:|--------:|---------:|-------:|
| Logistic Regression | 0.8990 | 1.0000 | 0.4950 | 0.6622 | **0.9742** |
| XGBoost | 0.8795 | 0.9818 | 0.4050 | 0.5735 | 0.9342 |
| Ensemble | 0.8885 | 1.0000 | 0.4425 | 0.6135 | 0.6135 |
| DeBERTa-v3-base | 0.9415 | 0.9609 | 0.7375 | 0.8345 | 0.9488 |
| **BiLSTM (Final Model)** | **0.9650** | **0.9769** | **0.8450** | **0.9062** | **0.9729** |

---

#  Dataset

### Competition

Smart MCQ Solver Challenge

### Training Set

- 2,000 Multiple Choice Questions
- Five Options (A–E)
- One Correct Answer

### Test Set

- 500 Unseen Questions

### Evaluation Metric

**MAP@3 (Mean Average Precision @ 3)**

---

#  Data Preprocessing Pipeline

A common preprocessing pipeline was used across all models.

- Exploratory Data Analysis
- Ranking Dataset Construction (Binary Relevance)
- GroupShuffleSplit
- TF-IDF Feature Engineering
- Hugging Face Tokenization
- Custom Vocabulary Creation
- Sequence Padding
- Text Cleaning

---

#  Project Workflow

```text
Dataset
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Data Preprocessing
   │
   ▼
Ranking Dataset Construction
   │
   ▼
Model Training
   ├── TF-IDF + Logistic Regression
   ├── TF-IDF + XGBoost
   ├── Soft Voting Ensemble
   ├── DeBERTa-v3-base
   └── BiLSTM (Scratch)
   │
   ▼
Model Evaluation
   │
   ▼
Final Model Selection
   │
   ▼
Submission Generation
   │
   ▼
Streamlit Deployment
```

---

#  Project Structure

```text
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preparation.ipynb
│   ├── 03_Traditional_ML.ipynb
│   ├── 04_DeBERTa.ipynb
│   ├── 05_BiLSTM_Scratch.ipynb
│   ├── 06_Model_Comparison.ipynb
│   └── 07_Inference.ipynb
│
├── models/
├── data/
├── deployment/
├── requirements.txt
└── README.md
```

> **Note:** Update the notebook names above if they differ in your repository.

---

#  Streamlit Application

The deployed application allows users to:

- Enter a custom MCQ
- Predict the Top-3 answers
- View prediction confidence
- Explore model performance
- Learn about the project workflow

---

#  Experiment Tracking

All experiments were tracked using **Weights & Biases (W&B)**.

Tracked Metrics:

- Training Loss
- Validation Loss
- Accuracy
- Precision
- Recall
- F1 Score
- MAP@3

---

#  Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- XGBoost
- Streamlit
- Weights & Biases
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

#  Project Highlights

-  Compared Traditional ML, Transformer, and Deep Learning approaches
-  Built a complete ranking pipeline for MCQ solving
-  Implemented a custom BiLSTM architecture from scratch
-  Fine-tuned DeBERTa-v3-base
-  Developed an interactive Streamlit application
-  Tracked all experiments using Weights & Biases
-  Generated Kaggle-compatible submissions
---

#  Useful Links

###  Live Demo

https://dl-24f1002360-t22026-deploy.streamlit.app/

###  Weights & Biases

https://wandb.ai/24f1002360-indian-institute-of-technology-madras/24f1002360-t22026

---
