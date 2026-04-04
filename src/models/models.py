"""
models.py
---------
Three ML components:
  1. ClauseClassifier  — SVM + TF-IDF (multi-class clause category)
  2. RiskScorer        — Linear Regression on human risk ratings
  3. VerdictEngine     — Decision Tree for final explainable verdict
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────────────
# 1. CLAUSE CLASSIFIER
# ──────────────────────────────────────────────────────────────────

class ClauseClassifier:
    """
    SVM with TF-IDF features to classify a clause into a legal category.
    e.g. "We sell your data" → 'data-sharing'
    """

    MODEL_PATH = "models/clause_classifier.pkl"

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=8000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
                min_df=1,
            )),
            ("svm", SVC(
                kernel="linear",
                C=1.0,
                probability=True,
                class_weight="balanced",
            )),
        ])
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, texts: list, categories: list):
        encoded = self.label_encoder.fit_transform(categories)
        self.pipeline.fit(texts, encoded)
        self.is_trained = True
        os.makedirs("models", exist_ok=True)
        joblib.dump(self, self.MODEL_PATH)

    def predict(self, text: str) -> tuple[str, float]:
        """Returns (category, confidence_percent)"""
        if not self.is_trained:
            raise RuntimeError("Classifier not trained.")
        enc = self.pipeline.predict([text])[0]
        proba = self.pipeline.predict_proba([text])[0]
        confidence = round(float(np.max(proba)) * 100, 1)
        category = self.label_encoder.inverse_transform([enc])[0]
        return category, confidence

    def get_classes(self) -> list:
        return list(self.label_encoder.classes_)

    @classmethod
    def load(cls):
        if os.path.exists(cls.MODEL_PATH):
            return joblib.load(cls.MODEL_PATH)
        raise FileNotFoundError("No saved classifier. Run train() first.")


# ──────────────────────────────────────────────────────────────────
# 2. RISK SCORER
# ──────────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Linear Regression trained on human risk ratings (1–10 scale).
    Uses same TF-IDF vocabulary as the classifier for consistency.
    """

    # High-risk keyword weights (used as fallback + boost)
    HIGH_RISK = {
        "sell": 1.5, "selling": 1.5, "sold": 1.5,
        "terminate": 1.3, "terminated": 1.3, "termination": 1.3,
        "arbitration": 1.4, "arbitrate": 1.4,
        "track": 1.2, "tracking": 1.2, "surveillance": 1.4,
        "waive": 1.3, "waiver": 1.3,
        "class action": 1.5,
        "without notice": 1.4, "without warning": 1.4,
        "third part": 1.2,
        "irrevocable": 1.3,
        "indefinitely": 1.2,
        "government": 1.1,
        "law enforcement": 1.2,
    }

    LOW_RISK = {
        "notify": -0.8, "notification": -0.8,
        "delete": -0.7, "deletion": -0.7,
        "consent": -0.9, "explicit consent": -1.2,
        "ownership": -0.8,
        "never share": -1.0, "never sell": -1.0,
        "your right": -0.7,
        "30 days": -0.6, "14 days": -0.5,
        "encrypt": -0.6, "encrypted": -0.6,
    }

    DOC_TYPE_MULTIPLIERS = {
        "Banking / Finance": 1.35,
        "Health App": 1.30,
        "Social Media": 1.15,
        "E-commerce": 1.05,
        "Other": 1.00,
    }

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, texts: list, scores: list):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, scores)
        self.is_trained = True

    def _keyword_score(self, text: str) -> float:
        text_lower = text.lower()
        boost = 0.0
        for kw, weight in self.HIGH_RISK.items():
            if kw in text_lower:
                boost += weight
        for kw, weight in self.LOW_RISK.items():
            if kw in text_lower:
                boost += weight
        return boost

    def predict(self, text: str, doc_type: str = "Other") -> float:
        """Returns risk score 1–10"""
        base = 5.0

        if self.is_trained:
            try:
                X = self.vectorizer.transform([text])
                base = float(self.model.predict(X)[0])
            except Exception:
                pass

        # Apply keyword adjustment on top of regression
        base += self._keyword_score(text)
        multiplier = self.DOC_TYPE_MULTIPLIERS.get(doc_type, 1.0)
        score = float(np.clip(base * multiplier, 1.0, 10.0))
        return round(score, 2)


# ──────────────────────────────────────────────────────────────────
# 3. VERDICT ENGINE (Decision Tree)
# ──────────────────────────────────────────────────────────────────

class VerdictEngine:
    """
    Explainable Decision Tree that produces a final verdict.
    Rules are transparent and can be printed as text.
    """

    VERDICT_LABELS = {
        0: ("✅ Safe to Sign", "#22c55e", "Low overall risk. Standard terms."),
        1: ("⚠️ Sign with Caution", "#f59e0b", "Some clauses need attention before signing."),
        2: ("⛔ Do Not Sign", "#ef4444", "Critical clauses detected. Seek legal advice."),
    }

    CRITICAL_CATEGORIES = {"data-sharing", "tracking", "dispute-resolution", "account-termination"}

    def __init__(self):
        self.tree = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=2,
            class_weight="balanced",
        )
        self.feature_names = [
            "avg_risk_score",
            "max_risk_score",
            "high_risk_clause_count",
            "critical_category_present",
            "very_bad_clause_count",
        ]
        self.is_trained = False

    def _extract_features(self, results: list) -> list:
        scores = [r["risk_score"] for r in results]
        cats = {r["category"] for r in results}
        return [
            np.mean(scores),
            np.max(scores),
            sum(1 for s in scores if s >= 7),
            1 if cats & self.CRITICAL_CATEGORIES else 0,
            sum(1 for s in scores if s >= 9),
        ]

    def _rule_based_verdict(self, features: list) -> int:
        avg, mx, high_count, critical, very_bad = features
        if mx >= 9.5 or very_bad >= 2:
            return 2
        if mx >= 8 or high_count >= 3 or (critical and avg >= 6):
            return 1
        if avg <= 3.5 and high_count == 0:
            return 0
        if avg >= 6.5:
            return 1
        return 0

    def verdict(self, results: list) -> dict:
        """
        Returns a verdict dict:
        {
          label, color, summary,
          avg_score, max_score,
          high_risk_clauses, reasoning
        }
        """
        if not results:
            return {
                "label": "⚪ Insufficient Data",
                "color": "#6b7280",
                "summary": "Not enough clauses to analyze.",
                "avg_score": 0, "max_score": 0,
                "high_risk_clauses": 0, "reasoning": [],
            }

        features = self._extract_features(results)
        avg, mx, high_count, critical, very_bad = features

        # Use trained tree if available, else rule-based
        if self.is_trained:
            verdict_idx = int(self.tree.predict([features])[0])
        else:
            verdict_idx = self._rule_based_verdict(features)

        label, color, summary = self.VERDICT_LABELS[verdict_idx]

        # Build human-readable reasoning
        reasoning = []
        if mx >= 9:
            reasoning.append(f"🔴 At least one clause scored {mx:.1f}/10 — extremely high risk")
        if very_bad >= 1:
            reasoning.append(f"🔴 {very_bad} clause(s) rated 'Very Bad' by human reviewers")
        if high_count >= 3:
            reasoning.append(f"🟠 {high_count} clauses scored above 7/10")
        if critical:
            cats_found = {r["category"] for r in results} & self.CRITICAL_CATEGORIES
            reasoning.append(f"🟠 High-risk categories detected: {', '.join(cats_found)}")
        if avg <= 3:
            reasoning.append(f"🟢 Average risk score is low ({avg:.1f}/10)")
        if not reasoning:
            reasoning.append(f"📊 Average risk score: {avg:.1f}/10")

        # Worst clauses
        worst = sorted(results, key=lambda x: x["risk_score"], reverse=True)[:3]
        for w in worst:
            if w["risk_score"] >= 6:
                reasoning.append(
                    f"   ↳ [{w['category']}] scored {w['risk_score']:.1f}/10"
                )

        return {
            "label": label,
            "color": color,
            "summary": summary,
            "avg_score": round(avg, 2),
            "max_score": round(mx, 2),
            "high_risk_clauses": high_count,
            "reasoning": reasoning,
        }


# ──────────────────────────────────────────────────────────────────
# CONVENIENCE: train all models at once
# ──────────────────────────────────────────────────────────────────

def train_all(df: pd.DataFrame) -> tuple:
    """Train classifier + scorer and return both."""
    classifier = ClauseClassifier()
    classifier.train(df["text"].tolist(), df["category"].tolist())

    scorer = RiskScorer()
    scorer.train(df["text"].tolist(), df["rating_score"].tolist())

    verdict_engine = VerdictEngine()

    print("[models] All models trained.")
    return classifier, scorer, verdict_engine
