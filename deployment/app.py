
import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd

MODEL_DIR = "models"

TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost.pkl")

print("Loading models...")

import joblib

tfidf = joblib.load(TFIDF_PATH)
logistic_model = joblib.load(LR_PATH)
xgb_model = joblib.load(XGB_PATH)

print("Models loaded successfully!")

LABELS = ["A", "B", "C", "D", "E"]


def predict_mcq(question, option_a, option_b, option_c, option_d, option_e):

    
    inputs = [question, option_a, option_b, option_c, option_d, option_e]

    if any(str(x).strip() == "" for x in inputs):
        return (
            " Please fill in the question and all five options.",
            "",
            pd.DataFrame()
        )

    options = [option_a, option_b, option_c, option_d, option_e]

  
    texts = [
        question + " [SEP] " + option
        for option in options
    ]

   
    X = tfidf.transform(texts)

  
    lr_probs = logistic_model.predict_proba(X)[:, 1]
    xgb_probs = xgb_model.predict_proba(X)[:, 1]

   
    scores = (lr_probs + xgb_probs) / 2

    
    ranking = np.argsort(scores)[::-1]

    top3 = [LABELS[i] for i in ranking[:3]]

    best_answer = top3[0]

   
    results = pd.DataFrame({
        "Option": LABELS,
        "Confidence": scores
    })

    results = results.sort_values(
        by="Confidence",
        ascending=False
    )

    results["Confidence"] = (
        results["Confidence"] * 100
    ).round(2).astype(str) + "%"

    top3_text = " → ".join(top3)

    return (
        best_answer,
        top3_text,
        results
    )
    
    
    
    

with gr.Blocks(
    title="Smart MCQ Solver",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
#  Smart MCQ Solver

### Deep Learning & Generative AI Project

**Indian Institute of Technology Madras**

Predict the correct answer for a multiple-choice question using an ensemble of
**TF-IDF + Logistic Regression + XGBoost**.

---
"""
    )

    with gr.Row():

        with gr.Column(scale=2):

            question = gr.Textbox(
                label="Question",
                placeholder="Enter your question here...",
                lines=4
            )

            option_a = gr.Textbox(
                label="Option A",
                placeholder="Enter Option A"
            )

            option_b = gr.Textbox(
                label="Option B",
                placeholder="Enter Option B"
            )

            option_c = gr.Textbox(
                label="Option C",
                placeholder="Enter Option C"
            )

            option_d = gr.Textbox(
                label="Option D",
                placeholder="Enter Option D"
            )

            option_e = gr.Textbox(
                label="Option E",
                placeholder="Enter Option E"
            )

            predict_button = gr.Button(
                "🚀 Predict",
                variant="primary"
            )

        with gr.Column(scale=1):

            best_answer = gr.Textbox(
                label=" Best Prediction",
                interactive=False
            )

            top3 = gr.Textbox(
                label=" Top 3 Predictions",
                interactive=False
            )

            confidence_table = gr.Dataframe(
                headers=["Option", "Confidence"],
                interactive=False,
                wrap=True
            )

    predict_button.click(
        fn=predict_mcq,
        inputs=[
            question,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e
        ],
        outputs=[
            best_answer,
            top3,
            confidence_table
        ]
    )

    gr.Markdown(
        """
---

### Project Information

**Competition:** Smart MCQ Solver Challenge

**Models Used**
- TF-IDF
- Logistic Regression
- XGBoost
- Ensemble Learning

**Evaluation Metric**
- MAP@3 (Mean Average Precision @ 3)


"""
    )
    
    

if __name__ == "__main__":
    demo.launch()