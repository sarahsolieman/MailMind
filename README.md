# MailMind — Email Prioritization Model

**MailMind** is an intelligent email classification system that prioritizes messages based on semantic urgency. It assigns each email to one of three categories:

- **Prioritize:** Critical messages such as MFA codes, account alerts, or security notifications  
- **Default:** Standard communication or routine coordination  
- **Slow:** Non-urgent content like newsletters, promotions, or feedback requests  

The goal of this project is to model *urgency perception* in email text through natural language understanding and lightweight, explainable machine learning.

---

## Key Features

- **Semantic understanding:** Uses `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`) to capture context beyond keyword matching.  
- **Transparent modeling:** Employs a logistic regression classifier for interpretability and ease of deployment.  
- **Cluster-informed labeling:** Initial dataset relabeling was guided by unsupervised clustering (TF-IDF + KMeans) to ensure each category reflects realistic tone and content distribution.  
- **Evaluation dashboard:** A Streamlit-based interface visualizes model performance metrics, including precision, recall, F1-score, and confusion matrix.  
- **Interactive inference:** Users can test live email samples, with outputs including the predicted label, confidence score, and a brief reasoning summary.  
- **Open-set recognition:** Incorporates a rejection mechanism for low-confidence or out-of-domain inputs, preventing forced classifications.  

---

## Technical Overview

MailMind’s architecture combines sentence-level embeddings with a lightweight linear classifier.

1. Emails are preprocessed and encoded using semantic embeddings.  
2. A logistic regression model predicts the urgency class based on vectorized representations.  
3. Predictions include confidence, class probabilities, and an interpretable reasoning summary.  
4. A Streamlit dashboard provides visual analysis, performance metrics, and interactive testing.  

---

## Results

The model achieves near-perfect performance on balanced validation data and demonstrates strong generalization to unseen examples, including email verification codes, password resets, and promotional content.

---

## Deployment

The project includes a fully deployable Streamlit application with:

- Interactive input fields for subject and body text  
- Real-time classification and reasoning display  
- Performance visualization via confusion matrix and per-class metrics  

### Run Locally

```bash
# create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# start the app
streamlit run app.py
