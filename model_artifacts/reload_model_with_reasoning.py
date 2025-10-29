
# ======================================================
# 🚀 Reload Email Priority Classifier (with reasoning)
# ======================================================

import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Load artifacts ---
artifacts = joblib.load("model_artifacts/email_priority_model.joblib")
model = artifacts["model"]
classes = artifacts["classes_"]

# --- Load same sentence-transformer ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Lightweight keyword lexicon for 'reasoning' context ---
reasoning_terms = {
    "Prioritize": ["security", "verify", "account", "code", "login", "alert", "expires", "activity", "immediately"],
    "Slow": ["newsletter", "update", "sale", "community", "feedback", "blog", "event", "promotion", "digest"],
    "Default": ["meeting", "attached", "project", "invoice", "report", "schedule", "document", "reminder"]
}

def explain_email(text):
    vec = embedder.encode([text], normalize_embeddings=True)
    probs = model.predict_proba(vec)[0]
    idx = int(np.argmax(probs))
    label = classes[idx]
    confidence = float(probs[idx])

    # Pick top reasoning tokens based on detected label
    reasoning = reasoning_terms.get(label, [])[:5]
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
        "probs": {cls: float(p) for cls, p in zip(classes, probs)}
    }

# --- Example usage ---
if __name__ == "__main__":
    sample = "Your verification code is 482019 — enter it within 5 minutes to continue."
    print(explain_email(sample))
