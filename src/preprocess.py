import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)
    return " ".join(tokens)

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(clean_text)
    return df