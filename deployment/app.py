import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🏆",
    layout="wide"
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
LR_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost.pkl")


@st.cache_resource
def load_models():
    tfidf = joblib.load(TFIDF_PATH)
    logistic_model = joblib.load(LR_PATH)
    xgb_model = joblib.load(XGB_PATH)
    return tfidf, logistic_model, xgb_model


tfidf, logistic_model, xgb_model = load_models()

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
    
    
    
    

st.title("Smart MCQ Solver")

st.markdown("""
### Deep Learning & Generative AI Project

**Indian Institute of Technology Madras**

Predict the correct answer using an ensemble of:

- TF-IDF
- Logistic Regression
- XGBoost
""")

left, right = st.columns([2,1])

with left:

    question = st.text_area(
        "Question",
        height=120
    )

    option_a = st.text_input("Option A")

    option_b = st.text_input("Option B")

    option_c = st.text_input("Option C")

    option_d = st.text_input("Option D")

    option_e = st.text_input("Option E")

    predict = st.button(
        "Predict Answer",
        use_container_width=True
    )

with right:

    best_placeholder = st.empty()
    top3_placeholder = st.empty()
    table_placeholder = st.empty()

if predict:

    best_answer, top3, confidence_table = predict_mcq(
        question,
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    )

    best_placeholder.success(
        f"###  Best Prediction: {best_answer}"
    )

    top3_placeholder.info(
        f"### Top 3 Predictions\n{top3}"
    )

    table_placeholder.dataframe(
        confidence_table,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

st.subheader("Project Information")

st.markdown("""
**Competition:** Smart MCQ Solver Challenge

### Models Used

- TF-IDF
- Logistic Regression
- XGBoost
- Ensemble Learning

### Evaluation Metric

MAP@3 (Mean Average Precision @3)
""")
