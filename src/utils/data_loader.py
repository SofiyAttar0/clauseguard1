"""
data_loader.py
--------------
Fetches labeled clause data from ToS;DR API.
Falls back to a built-in sample dataset if the API is unavailable.
"""

import requests
import pandas as pd
import numpy as np
import os
import json


RATING_MAP = {
    "good": 1,
    "neutral": 3,
    "bad": 7,
    "very bad": 10,
    "blocker": 10,
}

CATEGORY_RISK_WEIGHTS = {
    "data-sharing": 2.5,
    "tracking": 2.5,
    "account-termination": 2.0,
    "dispute-resolution": 1.8,
    "liability": 1.5,
    "policy-change": 1.2,
    "data-retention": 1.8,
    "ownership": 1.0,
    "account-deletion": 1.0,
    "uncategorized": 1.0,
}


def map_rating(rating: str) -> int:
    return RATING_MAP.get(str(rating).lower().strip(), 3)


def fetch_tosdr_data(max_services: int = 120, save_path: str = "data/clauses.csv") -> pd.DataFrame:
    """
    Pull clause data from the ToS;DR v2 API.
    Saves to CSV so you only fetch once.
    """
    if os.path.exists(save_path):
        print(f"[data_loader] Loading cached data from {save_path}")
        return pd.read_csv(save_path)

    print("[data_loader] Fetching from ToS;DR API...")
    clauses = []

    for service_id in range(1, max_services + 1):
        try:
            r = requests.get(
                f"https://api.tosdr.org/service/v2/?id={service_id}",
                timeout=6,
            )
            data = r.json()
            if "parameters" not in data:
                continue

            service_name = data["parameters"].get("name", "Unknown")
            points = data["parameters"].get("points", [])

            for point in points:
                title = point.get("title", "")
                desc = point.get("description", "")
                text = f"{title}. {desc}".strip()
                cats = point.get("categories", [])
                category = cats[0] if cats else "uncategorized"
                rating = point.get("case", {}).get("classification", "neutral")

                clauses.append(
                    {
                        "service": service_name,
                        "text": text,
                        "category": category,
                        "rating": rating,
                        "rating_score": map_rating(rating),
                    }
                )
        except Exception:
            continue

    df = pd.DataFrame(clauses)
    df.dropna(subset=["text"], inplace=True)
    df = df[df["text"].str.len() > 30].reset_index(drop=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[data_loader] Saved {len(df)} clauses from {df['service'].nunique()} services.")
    return df


def get_sample_data() -> pd.DataFrame:
    """
    Built-in fallback dataset. Covers all major clause categories.
    Use this for instant testing without API access.
    """
    rows = [
        # data-sharing — bad/very bad
        ("We may sell your personal data to third-party advertisers without your consent.", "data-sharing", "bad", 7),
        ("Your information may be shared with our business partners for marketing.", "data-sharing", "bad", 7),
        ("We share data with affiliated companies and subsidiaries.", "data-sharing", "neutral", 3),
        ("We never sell or rent your personal information to third parties.", "data-sharing", "good", 1),
        ("User data is never disclosed to advertisers.", "data-sharing", "good", 1),
        ("We may disclose your data to government agencies upon request without notifying you.", "data-sharing", "very bad", 10),

        # tracking
        ("We track your location continuously, including when the app is running in the background.", "tracking", "very bad", 10),
        ("We collect device identifiers to track usage across sessions.", "tracking", "bad", 7),
        ("We use cookies to improve your experience on this platform.", "tracking", "neutral", 3),
        ("Location data is only collected with your explicit permission.", "tracking", "good", 1),
        ("We do not track users across third-party websites.", "tracking", "good", 1),

        # account-termination
        ("We reserve the right to terminate your account at any time without notice or reason.", "account-termination", "very bad", 10),
        ("Accounts may be suspended if they violate our community guidelines.", "account-termination", "neutral", 3),
        ("You will receive a 14-day notice before account termination.", "account-termination", "good", 1),
        ("We can delete your account without refunding any paid subscriptions.", "account-termination", "bad", 7),

        # dispute-resolution
        ("All disputes must be resolved through binding arbitration. You waive your right to a jury trial.", "dispute-resolution", "very bad", 10),
        ("You waive your right to participate in class action lawsuits.", "dispute-resolution", "very bad", 10),
        ("Disputes may be settled in your local jurisdiction.", "dispute-resolution", "good", 1),
        ("We offer a mediation process before any legal action is required.", "dispute-resolution", "good", 1),
        ("Any legal action must be filed in the courts of Delaware.", "dispute-resolution", "bad", 7),

        # liability
        ("The service is provided as-is with no warranty of any kind.", "liability", "neutral", 3),
        ("We are not liable for any loss of data or damages arising from service interruptions.", "liability", "bad", 7),
        ("Our liability is limited to the amount you paid us in the last 12 months.", "liability", "bad", 7),
        ("We take full responsibility for data breaches caused by our negligence.", "liability", "good", 1),

        # policy-change
        ("We may update these terms at any time without notifying you.", "policy-change", "very bad", 10),
        ("Continued use of the service constitutes acceptance of new terms.", "policy-change", "bad", 7),
        ("We will notify you by email at least 30 days before any material changes.", "policy-change", "good", 1),
        ("Major changes to privacy policy require your explicit re-consent.", "policy-change", "good", 1),

        # account-deletion
        ("You can delete your account and all associated data at any time.", "account-deletion", "good", 1),
        ("Data deletion requests will be processed within 30 days.", "account-deletion", "good", 1),
        ("Some data may be retained for up to 7 years after account deletion.", "account-deletion", "bad", 7),
        ("Deleted accounts cannot be restored and all data is permanently erased.", "account-deletion", "neutral", 3),

        # ownership
        ("You retain full intellectual property rights over content you create.", "ownership", "good", 1),
        ("By uploading content, you grant us an irrevocable, royalty-free license to use it.", "ownership", "bad", 7),
        ("We may use your content for advertising without compensation.", "ownership", "very bad", 10),
        ("We only use your content to provide the service you signed up for.", "ownership", "good", 1),

        # data-retention
        ("We retain your data indefinitely even after account deletion.", "data-retention", "very bad", 10),
        ("Personal data is deleted 90 days after account closure.", "data-retention", "good", 1),
        ("We keep logs of your activity for security purposes for up to 1 year.", "data-retention", "neutral", 3),
        ("Backups may retain your data for an additional 60 days after deletion.", "data-retention", "neutral", 3),
    ]

    return pd.DataFrame(rows, columns=["text", "category", "rating", "rating_score"])


def load_data(use_api: bool = False) -> pd.DataFrame:
    """
    Main entry point. Use use_api=True to fetch from ToS;DR.
    Defaults to sample data for speed during development.
    """
    if use_api:
        try:
            return fetch_tosdr_data()
        except Exception as e:
            print(f"[data_loader] API failed ({e}), falling back to sample data.")
    return get_sample_data()
