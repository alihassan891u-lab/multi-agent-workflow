"""Intake agent: classifies an inbound message into a service line and extracts budget and timeline.

Model: TF-IDF (word + character n-grams) + logistic regression, trained on labelled inquiries.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "intake_classifier.joblib"

LABELS = {
    "ai_chatbot": "AI chatbot",
    "voice_agent": "AI voice agent",
    "automation": "Workflow automation",
    "ml_analytics": "Machine learning",
    "ai_integration": "LLM / MCP integration",
    "not_fit": "Not a fit",
}

WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "ten": 10, "twelve": 12}


def build_model():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    return make_pipeline(features, LogisticRegression(C=8, max_iter=2000, class_weight="balanced"))


def train(path: Path = ROOT / "data" / "train_leads.json"):
    rows = json.loads(path.read_text())
    model = build_model().fit([r["text"] for r in rows], [r["label"] for r in rows])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


_model = None


def model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train()
    return _model


def extract_budget(text: str) -> int | None:
    t = text.lower().replace(",", "")
    patterns = [
        r"\$\s?(\d+(?:\.\d+)?)\s?(k)?\b",                 # $8000, $8k, $ 7.5k
        r"(\d+(?:\.\d+)?)\s?(k)?\s?(?:usd|dollars)\b",     # 5000 usd, 4k dollars
        r"budget(?: is| of)?(?: around| about| ~)?\s?(\d+(?:\.\d+)?)\s?(k)?\b",
        r"~\s?\$?(\d+(?:\.\d+)?)\s?(k)\b",
        r"\b(\d+(?:\.\d+)?)(k)\b",
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            value = float(m.group(1)) * (1000 if m.group(2) else 1)
            if value >= 100:          # ignore things like "$49 followers" style noise below 100
                return int(value)
    return None


def extract_timeline_weeks(text: str) -> int | None:
    t = text.lower()
    m = re.search(r"(\d+|one|two|three|four|five|six|eight|ten|twelve)[\s-]+(week|month)s?", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUMBERS[m.group(1)]
        return n * (4 if m.group(2) == "month" else 1)
    if "next week" in t:
        return 1
    return None


def classify(text: str) -> dict:
    clf = model()
    probs = clf.predict_proba([text])[0]
    order = np.argsort(-probs)
    label = clf.classes_[order[0]]
    return {
        "category": str(label),
        "category_name": LABELS[str(label)],
        "confidence": round(float(probs[order[0]]), 3),
        "runner_up": str(clf.classes_[order[1]]),
        "budget": extract_budget(text),
        "timeline_weeks": extract_timeline_weeks(text),
    }


def intake_agent(state: dict) -> dict:
    lead = state["lead"]
    return {"intake": classify(f"{lead['subject']}. {lead['body']}")}
