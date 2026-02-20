import joblib
import numpy as np
from preprocess import clean_text

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

labels = ["scarcity", "social_proof", "urgency", "authority"]

def predict_text(text):
    # Clean input
    cleaned = clean_text(text)

    # Convert to TF-IDF
    vector = vectorizer.transform([cleaned])

    # Get prediction (0 or 1)
    prediction = model.predict(vector)[0]

    # Get probability scores
    probabilities = model.predict_proba(vector)[0]

    result = {}

    for i in range(len(labels)):
        result[labels[i]] = {
            "prediction": int(prediction[i]),
            "confidence": round(float(probabilities[i]), 3)
        }

    return result


if __name__ == "__main__":
    print("==== Persuasion Analysis NLP ====")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Enter marketing text: ")

        if user_input.lower() == "exit":
            print("Exiting program...")
            break

        output = predict_text(user_input)

        print("\nPrediction Results:")
        for label, values in output.items():
            status = "Detected" if values["prediction"] == 1 else "Not Detected"
            print(f"{label.upper()} → {status} (Confidence: {values['confidence']})")

        print("\n----------------------------------\n")