import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from preprocess import load_and_preprocess

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load and preprocess dataset
df = load_and_preprocess("../data/marketing_data.csv")

X = df["clean_text"]
y_true = df[["scarcity", "social_proof", "urgency", "authority"]]

# Transform text into TF-IDF features
X_tfidf = vectorizer.transform(X)

# Predict
y_pred = model.predict(X_tfidf)

print("========== Model Evaluation ==========\n")

# Overall Accuracy (Exact Match Ratio)
subset_accuracy = accuracy_score(y_true, y_pred)
print(f"Subset Accuracy (Exact Match Ratio): {subset_accuracy:.3f}")

# Hamming Loss
hl = hamming_loss(y_true, y_pred)
print(f"Hamming Loss: {hl:.3f}\n")

# Detailed Classification Report
print("Detailed Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["scarcity", "social_proof", "urgency", "authority"]
))