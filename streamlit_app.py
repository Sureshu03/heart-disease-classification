import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Heart Disease Classification App")

# Upload dataset (test data only)
uploaded_file = st.file_uploader("Upload CSV file (test data only)", type="csv")

if uploaded_file is None:
    # Fallback to default heart.csv
    data = pd.read_csv("heart.csv")
    st.info("No file uploaded, using default heart.csv dataset")
else:
    # User uploaded a file
    data = pd.read_csv(uploaded_file)
    st.success("Using uploaded file")

st.write("Dataset Preview:", data.head())

# Model selection dropdown
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

if not data.empty:
    # Read uploaded CSV
    st.write("Uploaded Data Preview:", data.head())

    # Load the chosen model from .pkl file
    model_file = f"model/{model_choice.lower().replace(' ', '-')}.pkl"
    model = pickle.load(open(model_file, "rb"))

    # Separate features and target if target exists
    if "target" in data.columns:
        X_test = data.drop("target", axis=1)
        y_test = data["target"]
    else:
        X_test = data
        y_test = None

    # Predictions
    preds = model.predict(X_test)
    st.write("Predictions:", preds)

    # If target column exists, calculate metrics
    if y_test is not None:
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        mcc = matthews_corrcoef(y_test, preds)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        st.subheader("Evaluation Metrics")
        st.write(f"Accuracy: {acc:.2f}")
        st.write(f"Precision: {prec:.2f}")
        st.write(f"Recall: {rec:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
        st.write(f"MCC: {mcc:.2f}")
        if auc is not None:
            st.write(f"AUC: {auc:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, preds))