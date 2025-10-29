#source .venv/bin/activate

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix
import os

# Page setup 
st.set_page_config(
    page_title="SignalBox — Email Prioritization Model",
    page_icon="📬",
    layout="wide"
)
st.title("SignalBox — Email Prioritization Model")
st.markdown("### Semantic model that classifies emails into: **Prioritize**, **Default**, and **Slow**")


# Load model + embedder 
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("model_artifacts/email_priority_model.joblib")
    model = artifacts["model"]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return model, embedder, artifacts["classes"]

model, embedder, classes = load_artifacts()

# Reasoning dictionary 
reasoning_terms = {
    "Prioritize": [
        "Detected urgency-related keywords (code, verify, alert)",
        "Semantic match with transactional or security tone",
        "Embedding proximity to alert or verification templates"
    ],
    "Slow": [
        "Detected newsletter or low-urgency tone",
        "Match with promotional or feedback-style emails",
        "Low transactional or action-driven language"
    ],
    "Default": [
        "Detected neutral or work-related tone",
        "Semantic similarity to coordination or update messages",
        "No urgency or promotional indicators found"
    ],
    "Unknown": [
        "Low confidence — message may not fit known patterns",
        "Out-of-domain or novel phrasing detected"
    ]
}

# Sidebar: full narrative + metrics 
st.sidebar.title("Project Overview")
st.sidebar.markdown("""
**SignalBox** is an intelligent email prioritization model that classifies messages
into **three urgency tiers**:
- **Prioritize** — security alerts, MFA codes, transactional events  
- **Default** — general communication, coordination  
- **Slow** — newsletters, promotions, surveys  

This project explores how semantic tone and linguistic patterns convey *urgency*.
Clustering and manual relabeling were used to strengthen distinctions between
transactional and conversational content.
""")

# Clustering Insight Section 
st.sidebar.markdown("---")
st.sidebar.subheader("Clustering Insights")
st.sidebar.markdown("""
Using **TF-IDF + KMeans**, clusters revealed distinct tone groups:
- Cluster 0 → transactional / urgent  
- Cluster 1 → conversational / default  
- Cluster 2 → newsletters / slow  

This unsupervised analysis guided the **relabeling** process,
ensuring balanced representation across categories.
""")

cluster_img_path = os.path.join(os.path.dirname(__file__), "assets", "clusters.png")
if os.path.exists(cluster_img_path):
    st.sidebar.image(cluster_img_path, caption="t-SNE Clustering of Email Tones")

# Final Model Section 
st.sidebar.markdown("---")
st.sidebar.subheader("Final Model Design")
st.sidebar.markdown("""
SignalBox leverages **SentenceTransformer embeddings** (`all-MiniLM-L6-v2`) for contextual encoding
and a **Logistic Regression classifier** for interpretability and efficiency.
A lightweight **rejection mechanism** handles low-confidence predictions by returning **'Unknown'**.
""")

# Model Performance Section 
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")

try:
    csv_path = os.path.join(os.path.dirname(__file__), "model_artifacts", "validation_results.csv")
    df = pd.read_csv(csv_path)
    df["true_label"] = df["true_label"].astype(str).str.strip()
    df["predicted_label"] = df["predicted_label"].astype(str).str.strip()

    df = df.dropna(subset=["true_label", "predicted_label"])
    labels = sorted(df["true_label"].unique().tolist())
    cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=labels)

    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        title="Confusion Matrix"
    )
    st.sidebar.plotly_chart(fig, use_container_width=True)

    
    from sklearn.metrics import classification_report

    # Compute full metrics 
    report = classification_report(
        df["true_label"],
        df["predicted_label"],
        output_dict=True,
        zero_division=0
    )

    accuracy = report["accuracy"] * 100
    st.sidebar.markdown(f"**Overall Accuracy:** {accuracy:.2f}%")

    # Convert metrics to a tidy DataFrame
    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "Label"})

    # Keep only the key numeric columns
    metric_cols = ["precision", "recall", "f1-score", "support"]
    report_df = report_df[["Label"] + metric_cols]

    # Round for readability
    report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].round(2)

    # Display in the sidebar
    st.sidebar.markdown("### Detailed Metrics")
    st.sidebar.dataframe(report_df, use_container_width=True, hide_index=True)


    label_counts = df["true_label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    bar_fig = px.bar(
        label_counts, x="Label", y="Count", color="Label", text_auto=True,
        title="Class Distribution"
    )
    st.sidebar.plotly_chart(bar_fig, use_container_width=True)

except Exception as e:
    st.sidebar.error(f"⚠️ Error displaying metrics: {e}")

# Future Work Section 
st.sidebar.markdown("---")
st.sidebar.subheader("Areas for Future Improvement")
st.sidebar.markdown("""
While SignalBox performs well on known message types, it currently assumes
**every email fits one of three classes**.  
Planned improvements:
- Add **embedding-distance thresholding** for better novelty detection  
- Support **human-in-the-loop feedback** for ambiguous cases  
- Track prediction entropy for **uncertainty-aware evaluation**
""")

# Test email interface 
st.markdown("## ✉️ Test an Email")
subject = st.text_input("Subject:")
body = st.text_area("Email Body:", height=200)

# Add a subtle helper note for users
st.caption("💡 *Tip:* You can leave the subject blank if you're pasting a full message for Email Body. "
           "Including both subject and body helps the model interpret urgency more accurately.")

def explain_email(text, threshold=0.20, min_length=5):
    text = text.strip()
    if len(text.split()) < min_length:
        return "Unknown", 0.0, ["Input too short — insufficient context."], [0, 0, 0]

    vec = embedder.encode([text], normalize_embeddings=True)
    probs = model.predict_proba(vec)[0]
    idx = np.argmax(probs)
    confidence = float(probs[idx])
    label = classes[idx]

    # Reject only if extremely low confidence
    if confidence < threshold:
        label = "Unknown"

    reasoning = reasoning_terms.get(label, ["No reasoning available."])
    return label, confidence, reasoning, probs


if st.button("Analyze Email"):
    email_text = f"{subject}. {body}"
    label, confidence, reasoning, probs = explain_email(email_text)

    st.markdown(f"### 🧩 Prediction: **{label}**")
    st.progress(confidence)
    st.markdown(f"**Confidence:** {confidence:.2f}")
    st.markdown(f"**Reasoning:** {'; '.join(reasoning)}")
    


    with st.expander("Probability Distribution"):
        prob_df = pd.DataFrame({"Class": classes, "Probability": probs})
        fig = px.bar(prob_df, x="Class", y="Probability", color="Class", range_y=[0,1])
        st.plotly_chart(fig, use_container_width=True)

# Footer 
st.markdown("---")

st.caption("Built by Sarah Solieman • Semantic ML for Email •")

st.markdown(
    "[View on GitHub](https://github.com/sarahsolieman/SignalBox)",
    unsafe_allow_html=True
)

