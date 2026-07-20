# Smart MCQ Solver Challenge

**DLGenAI Project | Jul 2026 | Student: 24f1002360**

**Deployment:** https://your-deployment-link-here

---

## Overview

This project addresses the **Smart MCQ Solver Challenge**, where the objective is to automatically answer multiple-choice questions by predicting the correct option from five candidates (A–E).

The project explores multiple approaches ranging from traditional machine learning to transformer-based models and Retrieval-Augmented Generation (RAG). Model performance is evaluated using **MAP@3 (Mean Average Precision @ 3)**, the official competition metric.

---

## Models

| Model | Type | Validation MAP@3 |
|------|------|-----------------:|
| TF-IDF + Logistic Regression | Traditional ML | 0.748 |
| TF-IDF + XGBoost | Traditional ML | 0.742 |
| Ensemble (LR + XGBoost) | Traditional ML | **0.748** |
| DeBERTa-v3 Base | Transformer Fine-tuning | 0.746 |
| BERT + Retrieval-Augmented Generation (RAG) | Transformer + RAG | 0.740 |

**Final Submission Model:** Traditional Ensemble (TF-IDF + Logistic Regression + XGBoost)

---

## Dataset

**Competition:** Smart MCQ Solver Challenge

**Training Data**
- 2,000 multiple-choice questions
- Five options (A–E)
- One correct answer per question

**Test Data**
- 500 unseen questions
- Predict top-3 ranked answers

**Evaluation Metric**
- MAP@3 (Mean Average Precision @ 3)

---

## Project Structure

```
notebooks/
│
├── 01_EDA.ipynb
├── 02_Data_Preparation.ipynb
├── 03_DeBERTa_Baseline.ipynb
├── 04_DeBERTa_Optimization.ipynb
├── 05_DeBERTa_Inference.ipynb
├── 06_Scratch_TFIDF_LR_XGBoost.ipynb
├── 07_BERT_RAG.ipynb
├── 08_Final_Comparison_and_Inference.ipynb
│
├── milestone-1.ipynb
├── milestone-2.ipynb
├── milestone-3.ipynb
├── milestone-4.ipynb
└── milestone-5.ipynb
```

---

## Project Workflow

1. Exploratory Data Analysis
   - Dataset exploration
   - Answer distribution
   - Question length analysis
   - Duplicate detection

2. Data Preparation
   - Ranking dataset creation
   - Text preprocessing
   - Feature engineering

3. Traditional ML Baseline
   - TF-IDF
   - Logistic Regression
   - XGBoost
   - Ensemble

4. Transformer Fine-tuning
   - DeBERTa Baseline
   - Hyperparameter Optimization
   - Multiple Choice Inference

5. Retrieval-Augmented Generation
   - BERT
   - External Context Retrieval
   - RAG Pipeline

6. Model Comparison

7. Final Inference & Submission Generation

---

## Requirements

```
transformers
torch
datasets
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
wandb
tqdm
sentence-transformers
faiss-cpu
```

---

## Experiment Tracking

All experiments were tracked using [**Weights & Biases (W&B)**](https://wandb.ai/24f1002360-indian-institute-of-technology-madras/24f1002360-t22026).

---

## Quick Start

1. Open the notebooks on Kaggle.
2. Add your `WANDB_API_KEY` to Kaggle Secrets.
3. Run the notebooks separately.
4. Save the models to kaggle datasets and load in the final inference notebook as input.
5. The final `submission.csv` will be generated in:

```
/kaggle/working/
```

---
