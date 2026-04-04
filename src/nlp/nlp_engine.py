"""
nlp_engine.py
-------------
All NLP features (no external LLM required):
  1. Summarizer          — TF-IDF extractive summarization
  2. QuestionSuggester   — Rule-based smart question generation
  3. SemanticSearch      — Cosine similarity clause retrieval (Q&A)
  4. DocumentComparator  — Side-by-side risk comparison
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into clean sentences."""
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def split_clauses(text: str) -> list[str]:
    """
    Split a full T&C document into individual clauses.
    Handles numbered lists, lettered items, and paragraph breaks.
    """
    # Break on numbered/lettered list items
    text = re.sub(r"\n{2,}", "\n\n", text)
    parts = re.split(r"(?:\n\s*\n|\n(?=\s*\d+[\.\)]\s|\s*[a-z][\.\)]\s))", text)

    clauses = []
    for part in parts:
        sentences = split_sentences(part)
        clauses.extend(sentences)

    return [c for c in clauses if len(c) > 40][:80]  # cap at 80 clauses


# ──────────────────────────────────────────────────────────────────
# 1. SUMMARIZER
# ──────────────────────────────────────────────────────────────────

class Summarizer:
    """
    Extractive summarization using TF-IDF sentence scoring.
    Picks the most 'information-dense' sentences.
    No LLM needed.
    """

    def summarize(self, text: str, num_sentences: int = 5) -> dict:
        sentences = split_sentences(text)

        if len(sentences) == 0:
            return {"summary": "No content to summarize.", "key_points": []}

        if len(sentences) <= num_sentences:
            return {
                "summary": " ".join(sentences),
                "key_points": sentences,
            }

        # TF-IDF score each sentence
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(sentences)
            # Score = sum of TF-IDF weights for each sentence
            scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

            # Boost sentences that contain important legal keywords
            BOOST_WORDS = [
                "sell", "terminate", "arbitration", "track", "waive",
                "delete", "share", "collect", "retain", "notify", "consent",
                "ownership", "license", "liability", "warrant",
            ]
            for i, sentence in enumerate(sentences):
                sl = sentence.lower()
                for word in BOOST_WORDS:
                    if word in sl:
                        scores[i] *= 1.3

            # Pick top N by score, but preserve document order
            top_indices = sorted(
                np.argsort(scores)[-num_sentences:].tolist()
            )
            key_points = [sentences[i] for i in top_indices]
            summary = " ".join(key_points)

        except Exception:
            key_points = sentences[:num_sentences]
            summary = " ".join(key_points)

        return {"summary": summary, "key_points": key_points}


# ──────────────────────────────────────────────────────────────────
# 2. QUESTION SUGGESTER
# ──────────────────────────────────────────────────────────────────

class QuestionSuggester:
    """
    Analyzes a T&C document and suggests the most relevant questions
    a user should ask based on detected clause categories and keywords.
    """

    CATEGORY_QUESTIONS = {
        "data-sharing": [
            "Is my personal data sold to third parties?",
            "Who does the company share my data with?",
            "Can I opt out of data sharing?",
        ],
        "tracking": [
            "Does this service track my location?",
            "What data is collected when I use the app?",
            "Is tracking active even when the app is closed?",
        ],
        "account-termination": [
            "Can my account be deleted without warning?",
            "What happens to my data if my account is terminated?",
            "Do I get a refund if my account is terminated early?",
        ],
        "dispute-resolution": [
            "Can I sue this company in court?",
            "Am I waiving my right to a jury trial?",
            "Is arbitration mandatory for disputes?",
        ],
        "ownership": [
            "Who owns the content I upload?",
            "Can the company use my content for advertising?",
            "Do I give up intellectual property rights?",
        ],
        "policy-change": [
            "Will I be notified before the terms change?",
            "How much notice do they give before updating policies?",
            "Does continued use mean I accept new terms?",
        ],
        "data-retention": [
            "How long is my data kept after I leave?",
            "Is my data deleted when I close my account?",
            "What data persists after account deletion?",
        ],
        "liability": [
            "Is the company liable for data breaches?",
            "What protections do I have if the service fails?",
            "Is there a limit on what compensation I can claim?",
        ],
    }

    KEYWORD_TRIGGERS = {
        "arbitration": "dispute-resolution",
        "class action": "dispute-resolution",
        "jury": "dispute-resolution",
        "sell": "data-sharing",
        "sold": "data-sharing",
        "third party": "data-sharing",
        "third-party": "data-sharing",
        "advertiser": "data-sharing",
        "track": "tracking",
        "location": "tracking",
        "gps": "tracking",
        "terminate": "account-termination",
        "suspend": "account-termination",
        "delete your account": "account-deletion",
        "irrevocable": "ownership",
        "royalty-free": "ownership",
        "license": "ownership",
        "retain": "data-retention",
        "retention": "data-retention",
        "liability": "liability",
        "warrant": "liability",
        "notify": "policy-change",
        "update": "policy-change",
        "change": "policy-change",
    }

    ALWAYS_SUGGEST = [
        "What data does this service collect about me?",
        "Can I delete my account and all my data?",
        "Is this service safe to use?",
    ]

    def suggest(self, text: str, max_questions: int = 8) -> list[str]:
        text_lower = text.lower()
        detected_categories = set()

        for keyword, category in self.KEYWORD_TRIGGERS.items():
            if keyword in text_lower:
                detected_categories.add(category)

        questions = list(self.ALWAYS_SUGGEST)

        for category in detected_categories:
            cat_questions = self.CATEGORY_QUESTIONS.get(category, [])
            questions.extend(cat_questions[:2])  # Max 2 per category

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        return unique[:max_questions]


# ──────────────────────────────────────────────────────────────────
# 3. SEMANTIC SEARCH (Question Answering)
# ──────────────────────────────────────────────────────────────────

class SemanticSearch:
    """
    Answers user questions about a T&C document by finding the
    most semantically similar clauses using TF-IDF cosine similarity.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
        )
        self.clause_vectors = None
        self.clauses = []

    def index(self, text: str):
        """Index a document so it can be queried."""
        self.clauses = split_clauses(text)
        if not self.clauses:
            self.clause_vectors = None
            return
        self.clause_vectors = self.vectorizer.fit_transform(self.clauses)

    def answer(self, question: str, top_k: int = 3) -> list[dict]:
        """
        Find the top-k most relevant clauses for a question.
        Returns list of {clause, similarity_score, snippet}
        """
        if self.clause_vectors is None or not self.clauses:
            return [{"clause": "No document indexed yet.", "score": 0.0}]

        try:
            q_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(q_vec, self.clause_vectors).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score < 0.01:
                    continue
                results.append({
                    "clause": self.clauses[idx],
                    "score": round(score * 100, 1),
                    "snippet": self.clauses[idx][:200] + "..."
                    if len(self.clauses[idx]) > 200
                    else self.clauses[idx],
                })

            if not results:
                return [
                    {
                        "clause": "No relevant clause found for your question.",
                        "score": 0.0,
                        "snippet": "Try rephrasing or asking about data, privacy, termination, or disputes.",
                    }
                ]
            return results

        except Exception as e:
            return [{"clause": f"Search error: {str(e)}", "score": 0.0}]


# ──────────────────────────────────────────────────────────────────
# 4. DOCUMENT COMPARATOR
# ──────────────────────────────────────────────────────────────────

class DocumentComparator:
    """
    Compares two T&C documents on:
      - Overall cosine similarity
      - Per-category risk score difference
      - Side-by-side verdict
    """

    def compare(
        self,
        results_a: list[dict],
        results_b: list[dict],
        label_a: str = "Document A",
        label_b: str = "Document B",
        text_a: str = "",
        text_b: str = "",
    ) -> dict:

        def aggregate(results):
            if not results:
                return {"avg": 0, "max": 0, "count": 0, "by_category": {}}
            scores = [r["risk_score"] for r in results]
            by_cat = {}
            for r in results:
                cat = r["category"]
                by_cat.setdefault(cat, [])
                by_cat[cat].append(r["risk_score"])
            by_cat_avg = {k: round(np.mean(v), 2) for k, v in by_cat.items()}
            return {
                "avg": round(np.mean(scores), 2),
                "max": round(np.max(scores), 2),
                "count": len(results),
                "by_category": by_cat_avg,
            }

        stats_a = aggregate(results_a)
        stats_b = aggregate(results_b)

        # Cosine similarity between full documents
        doc_similarity = 0.0
        if text_a and text_b:
            try:
                vec = TfidfVectorizer(stop_words="english")
                vecs = vec.fit_transform([text_a, text_b])
                doc_similarity = round(
                    float(cosine_similarity(vecs[0], vecs[1])[0][0]) * 100, 1
                )
            except Exception:
                pass

        # Which is safer?
        if stats_a["avg"] < stats_b["avg"]:
            safer = label_a
            riskier = label_b
        elif stats_b["avg"] < stats_a["avg"]:
            safer = label_b
            riskier = label_a
        else:
            safer = "Both similar"
            riskier = "Both similar"

        # Category-level comparison
        all_cats = set(stats_a["by_category"]) | set(stats_b["by_category"])
        category_diff = {}
        for cat in all_cats:
            score_a = stats_a["by_category"].get(cat, None)
            score_b = stats_b["by_category"].get(cat, None)
            category_diff[cat] = {"a": score_a, "b": score_b}

        return {
            "label_a": label_a,
            "label_b": label_b,
            "stats_a": stats_a,
            "stats_b": stats_b,
            "doc_similarity_pct": doc_similarity,
            "safer": safer,
            "riskier": riskier,
            "category_diff": category_diff,
        }
