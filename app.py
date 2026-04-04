"""
app.py
------
ClauseGuard — Main Streamlit Application
5 features: Summarization, Risk Analysis, Q&A, Auto Questions, Comparison
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.utils.data_loader import load_data
from src.models.models import train_all
from src.nlp.nlp_engine import Summarizer, QuestionSuggester, SemanticSearch, DocumentComparator

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ClauseGuard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg:        #0b0b18;
    --surface:   #13131f;
    --surface2:  #1c1c2e;
    --border:    #2a2a45;
    --accent:    #6366f1;
    --accent2:   #818cf8;
    --danger:    #ef4444;
    --warn:      #f59e0b;
    --success:   #22c55e;
    --text:      #e2e2f0;
    --muted:     #7c7ca8;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

html, body, [class*="css"] { font-family: var(--sans); }

.stApp { background: var(--bg); color: var(--text); }
.stSidebar { background: var(--surface) !important; border-right: 1px solid var(--border); }

/* Header */
.cg-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.cg-logo {
    font-family: var(--mono);
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.cg-sub {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.8rem;
    padding: 8px 18px;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* Cards */
.cg-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem;
    margin: 0.6rem 0;
}
.cg-card-danger  { border-left: 4px solid var(--danger); }
.cg-card-warn    { border-left: 4px solid var(--warn); }
.cg-card-success { border-left: 4px solid var(--success); }
.cg-card-accent  { border-left: 4px solid var(--accent); }

/* Verdict */
.verdict-box {
    border-radius: 16px;
    padding: 2rem;
    margin: 1.2rem 0;
    text-align: center;
}
.verdict-label {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.verdict-sub { font-size: 0.9rem; opacity: 0.8; }

/* Clause pill */
.clause-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    letter-spacing: 0.5px;
}
.badge-danger  { background: #7f1d1d; color: #fca5a5; }
.badge-warn    { background: #78350f; color: #fcd34d; }
.badge-success { background: #14532d; color: #86efac; }

/* Risk bar */
.risk-bar-wrap { background: var(--surface2); border-radius: 99px; height: 8px; margin-top: 6px; }
.risk-bar-fill { height: 8px; border-radius: 99px; }

/* Summary bullets */
.kp-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    border-left: 3px solid var(--accent);
}

/* Question chips */
.q-chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 6px 16px;
    margin: 4px;
    font-size: 0.82rem;
    cursor: pointer;
    color: var(--accent2);
    transition: all 0.2s;
}

/* Metrics */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent2);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Compare table */
.compare-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    border-bottom: 1px solid var(--border);
    padding: 0.7rem 0;
    align-items: center;
}
.compare-header { font-family: var(--mono); font-size: 0.75rem; color: var(--muted); }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Text areas and inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Sidebar labels */
.stSidebar .stMarkdown h3 {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
}

/* Info / warning boxes */
.stAlert { border-radius: 10px !important; border: 1px solid var(--border) !important; }

/* Scrollable clause list */
.clause-scroll {
    max-height: 520px;
    overflow-y: auto;
    padding-right: 4px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_models():
    df = load_data(use_api=False)
    classifier, scorer, verdict_engine = train_all(df)
    return classifier, scorer, verdict_engine

@st.cache_resource(show_spinner=False)
def get_nlp():
    return Summarizer(), QuestionSuggester(), SemanticSearch(), DocumentComparator()


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def analyze_document(text, doc_type, classifier, scorer, verdict_engine):
    """Run all ML models on a document. Returns list of clause dicts + verdict."""
    from src.nlp.nlp_engine import split_clauses
    clauses = split_clauses(text)

    results = []
    for clause in clauses:
        try:
            category, confidence = classifier.predict(clause)
            risk = scorer.predict(clause, doc_type)
            results.append({
                "clause": clause,
                "category": category,
                "confidence": confidence,
                "risk_score": risk,
            })
        except Exception:
            continue

    verdict = verdict_engine.verdict(results)
    return results, verdict


def risk_color(score):
    if score >= 7.5:
        return "#ef4444", "badge-danger", "🔴"
    if score >= 4.5:
        return "#f59e0b", "badge-warn", "🟡"
    return "#22c55e", "badge-success", "🟢"


def risk_bar(score):
    color, _, _ = risk_color(score)
    pct = int(score / 10 * 100)
    return f"""
    <div class="risk-bar-wrap">
        <div class="risk-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>
    """


SAMPLE_RISKY = """We may sell your personal data to third-party advertisers without your consent.
This service can terminate your account at any time without notice or reason.
All disputes must be resolved through binding arbitration and you waive your right to a jury trial.
You waive your right to participate in class action lawsuits against us.
We track your location at all times, including when the app is running in the background.
We retain your data indefinitely even after you delete your account.
By uploading content, you grant us an irrevocable, royalty-free worldwide license to use it.
We may update these terms at any time without notifying you. Continued use means you accept changes."""

SAMPLE_SAFE = """You retain full ownership of all content you create and upload to this service.
We will notify you by email at least 30 days before any material changes to this policy.
You can delete your account and all associated data at any time through your account settings.
We never sell or rent your personal information to third parties or advertisers.
Location data is only collected with your explicit permission and only while the app is open.
Data deletion requests are processed within 30 days of account closure.
Disputes may be resolved in your local jurisdiction by a court of your choosing.
We take full responsibility for data breaches caused by our negligence."""


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem">
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;
                    background:linear-gradient(135deg,#6366f1,#a78bfa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ⚖️ ClauseGuard
        </div>
        <div style="font-size:0.7rem;color:#7c7ca8;margin-top:2px;letter-spacing:1px;">
            ML-POWERED T&C ANALYZER
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Document Type")
    doc_type = st.selectbox(
        "Category",
        ["Other", "Social Media", "Banking / Finance", "Health App", "E-commerce"],
        label_visibility="collapsed",
        help="Higher-risk categories get extra scrutiny"
    )

    st.divider()
    st.markdown("### Options")
    show_confidence = st.toggle("Show classifier confidence", value=True)
    show_raw_clauses = st.toggle("Show all clauses in analysis", value=True)

    st.divider()
    st.markdown("### Model Stack")
    for label, model in [
        ("Classifier", "SVM + TF-IDF"),
        ("Risk Scorer", "Linear Regression"),
        ("Verdict", "Decision Tree"),
        ("Summarizer", "TF-IDF Extractive"),
        ("Q&A", "Cosine Similarity"),
        ("Comparison", "Cosine + Score Δ"),
    ]:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#7c7ca8;margin:3px 0;">'
            f'<span style="color:#6366f1;font-family:monospace">{label}</span>'
            f'  {model}</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        '<div style="font-size:0.7rem;color:#7c7ca8;">Training data: ToS;DR · CLAUDETTE<br>'
        'No LLMs. No API calls. 100% local ML.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="cg-header">
    <div class="cg-logo">⚖️ ClauseGuard</div>
    <div class="cg-sub">Explainable ML · Terms & Conditions Risk Intelligence</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════

with st.spinner("Loading models..."):
    classifier, scorer, verdict_engine = get_models()
    summarizer, question_suggester, semantic_search, comparator = get_nlp()


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Summarize",
    "🔍 Risk Analysis",
    "💬 Ask Questions",
    "⚡ Auto Questions",
    "📊 Compare",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARIZE
# ══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### 📋 Document Summarizer")
    st.caption("Paste a T&C document — get a plain-English extractive summary.")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        t1_text = st.text_area(
            "Paste T&C document",
            height=280,
            placeholder="Paste full Terms & Conditions text here...",
            key="t1_text",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Load Risky Sample", key="t1_risky"):
                st.session_state.t1_text = SAMPLE_RISKY
                st.rerun()
        with c2:
            if st.button("Load Safe Sample", key="t1_safe"):
                st.session_state.t1_text = SAMPLE_SAFE
                st.rerun()
        with c3:
            num_sent = st.number_input("Key points", min_value=2, max_value=10, value=5)

        summarize_btn = st.button("✨ Summarize", use_container_width=True, key="t1_go")

    with col_out:
        if summarize_btn and t1_text:
            with st.spinner("Analyzing..."):
                result = summarizer.summarize(t1_text, num_sentences=num_sent)

            st.markdown("**Summary**")
            st.markdown(
                f'<div class="cg-card cg-card-accent">'
                f'<p style="line-height:1.7;font-size:0.92rem;">{result["summary"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("**Key Points Extracted**")
            for i, kp in enumerate(result["key_points"], 1):
                st.markdown(
                    f'<div class="kp-item"><span style="color:#6366f1;font-family:monospace;'
                    f'margin-right:8px;">{i:02d}</span>{kp}</div>',
                    unsafe_allow_html=True
                )
        elif summarize_btn:
            st.warning("Please paste a document first.")
        else:
            st.markdown(
                '<div class="cg-card" style="text-align:center;padding:3rem 1rem;">'
                '<div style="font-size:2rem">📄</div>'
                '<div style="color:#7c7ca8;margin-top:0.5rem;font-size:0.9rem">'
                'Your summary will appear here</div></div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════
# TAB 2 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🔍 Risk Analysis")
    st.caption("Every clause is classified, scored, and explained. Fully offline ML.")

    t2_text = st.text_area(
        "Paste T&C document",
        height=180,
        placeholder="Paste Terms & Conditions...",
        key="t2_text",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Load Risky Sample", key="t2_risky"):
            st.session_state.t2_text = SAMPLE_RISKY
            st.rerun()
    with c2:
        if st.button("Load Safe Sample", key="t2_safe"):
            st.session_state.t2_text = SAMPLE_SAFE
            st.rerun()

    analyze_btn = st.button("🔍 Analyze Risk", use_container_width=True, key="t2_go", type="primary")

    if analyze_btn and t2_text:
        with st.spinner("Running ML pipeline..."):
            results, verdict = analyze_document(t2_text, doc_type, classifier, scorer, verdict_engine)

        if not results:
            st.error("Could not extract clauses. Try adding more text.")
        else:
            # ── VERDICT ──
            v_color = verdict["color"]
            st.markdown(
                f'<div class="verdict-box" style="background:{v_color}18;border:1px solid {v_color}44;">'
                f'<div class="verdict-label" style="color:{v_color}">{verdict["label"]}</div>'
                f'<div class="verdict-sub">{verdict["summary"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # ── REASONING ──
            with st.expander("📖 Why this verdict?", expanded=True):
                for r in verdict["reasoning"]:
                    st.markdown(f"- {r}")

            # ── METRICS ──
            scores = [r["risk_score"] for r in results]
            st.markdown(
                f"""<div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(results)}</div>
                    <div class="metric-label">Clauses Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color:{'#ef4444' if np.mean(scores)>=7 else '#f59e0b' if np.mean(scores)>=4 else '#22c55e'}">
                        {np.mean(scores):.1f}
                    </div>
                    <div class="metric-label">Avg Risk Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color:#ef4444">{sum(1 for s in scores if s>=7.5)}</div>
                    <div class="metric-label">High-Risk Clauses</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color:#f59e0b">{np.max(scores):.1f}</div>
                    <div class="metric-label">Max Risk Found</div>
                </div>
                </div>""",
                unsafe_allow_html=True
            )

            # ── CLAUSE LIST ──
            if show_raw_clauses:
                st.markdown("#### Clause-by-Clause Breakdown")
                st.caption("Sorted by risk score (highest first)")

                results_sorted = sorted(results, key=lambda x: x["risk_score"], reverse=True)

                for r in results_sorted:
                    color, badge_cls, emoji = risk_color(r["risk_score"])
                    with st.expander(
                        f"{emoji} [{r['category']}]  ·  Risk {r['risk_score']:.1f}/10"
                        f"  —  {r['clause'][:55]}...",
                    ):
                        st.markdown(
                            f'<div class="cg-card">'
                            f'<p style="font-size:0.92rem;line-height:1.6">{r["clause"]}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        cc1, cc2 = st.columns(2)
                        cc1.metric("Category", r["category"])
                        cc2.metric("Risk Score", f"{r['risk_score']:.1f} / 10")
                        if show_confidence:
                            st.caption(f"Classifier confidence: {r['confidence']}%")
                        st.markdown(risk_bar(r["risk_score"]), unsafe_allow_html=True)

            # ── CHART ──
            st.markdown("#### Risk Score Distribution")
            chart_df = pd.DataFrame({
                "Clause": [f"C{i+1}" for i in range(len(scores))],
                "Risk Score": scores,
            })
            st.bar_chart(chart_df.set_index("Clause"), color="#6366f1")

    elif analyze_btn:
        st.warning("Please paste a document.")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — ASK QUESTIONS (Q&A)
# ══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 💬 Ask Questions About a Document")
    st.caption("Semantic clause retrieval — finds the most relevant clauses for your question.")

    t3_text = st.text_area(
        "Paste T&C document to search",
        height=160,
        placeholder="Paste Terms & Conditions here first...",
        key="t3_text",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Load Risky Sample", key="t3_risky"):
            st.session_state.t3_text = SAMPLE_RISKY
            st.rerun()
    with c2:
        if st.button("Load Safe Sample", key="t3_safe"):
            st.session_state.t3_text = SAMPLE_SAFE
            st.rerun()

    index_btn = st.button("📥 Index Document", use_container_width=True, key="t3_index")
    if index_btn and t3_text:
        with st.spinner("Indexing clauses..."):
            semantic_search.index(t3_text)
        st.session_state["doc_indexed"] = True
        st.session_state["indexed_text"] = t3_text
        st.success(f"Document indexed — {len(semantic_search.clauses)} clauses ready.")

    st.divider()

    question = st.text_input(
        "Your question",
        placeholder="e.g. Can they sell my data? Can I delete my account?",
        key="t3_q",
    )

    ask_btn = st.button("🔎 Find Answer", use_container_width=True, key="t3_ask", type="primary")

    if ask_btn:
        if not st.session_state.get("doc_indexed"):
            st.warning("Index a document first using the button above.")
        elif not question:
            st.warning("Type a question.")
        else:
            with st.spinner("Searching..."):
                answers = semantic_search.answer(question, top_k=3)

            st.markdown(f'**Most relevant clauses for:** *"{question}"*')
            for i, ans in enumerate(answers, 1):
                score_color = "#22c55e" if ans["score"] > 40 else "#f59e0b" if ans["score"] > 15 else "#7c7ca8"
                st.markdown(
                    f'<div class="cg-card cg-card-accent">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">'
                    f'<span style="font-family:monospace;font-size:0.75rem;color:#6366f1">RESULT {i}</span>'
                    f'<span style="font-family:monospace;font-size:0.75rem;color:{score_color}">'
                    f'Relevance: {ans["score"]:.0f}%</span></div>'
                    f'<p style="font-size:0.92rem;line-height:1.6;margin:0">{ans["clause"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Previous question history ──
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []


# ══════════════════════════════════════════════════════════════════
# TAB 4 — AUTO QUESTION SUGGESTIONS
# ══════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### ⚡ Smart Question Suggestions")
    st.caption(
        "The app reads your document and automatically suggests the most important "
        "questions you should be asking — based on detected clause categories."
    )

    t4_text = st.text_area(
        "Paste T&C document",
        height=180,
        placeholder="Paste Terms & Conditions...",
        key="t4_text",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load Risky Sample", key="t4_risky"):
            st.session_state.t4_text = SAMPLE_RISKY
            st.rerun()
    with c2:
        if st.button("Load Safe Sample", key="t4_safe"):
            st.session_state.t4_text = SAMPLE_SAFE
            st.rerun()

    suggest_btn = st.button("⚡ Generate Questions", use_container_width=True, key="t4_go", type="primary")

    if suggest_btn and t4_text:
        with st.spinner("Analyzing document..."):
            questions = question_suggester.suggest(t4_text, max_questions=9)

        st.markdown("#### Questions You Should Be Asking")
        st.caption("Click any question to copy it — then use the Ask Questions tab to get answers.")

        for i, q in enumerate(questions):
            st.markdown(
                f'<div class="kp-item" style="cursor:pointer;">'
                f'<span style="color:#6366f1;font-family:monospace;margin-right:10px;'
                f'font-size:0.75rem;">Q{i+1:02d}</span>{q}</div>',
                unsafe_allow_html=True
            )

        st.divider()
        st.info(
            "💡 Tip: Index this document in the **Ask Questions** tab, "
            "then paste any of the above questions to find relevant clauses."
        )

    elif suggest_btn:
        st.warning("Please paste a document first.")
    else:
        st.markdown(
            '<div class="cg-card" style="text-align:center;padding:3rem;">'
            '<div style="font-size:2rem">⚡</div>'
            '<div style="color:#7c7ca8;margin-top:0.5rem;font-size:0.9rem;">'
            'Paste a document and click Generate to see suggested questions</div>'
            '</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════
# TAB 5 — COMPARISON
# ══════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### 📊 Side-by-Side Document Comparison")
    st.caption(
        "Upload two T&C documents and get a head-to-head risk comparison. "
        "\"Spotify vs YouTube Music — who treats you better?\""
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("**Document A**")
        label_a = st.text_input("Label A", "Service A", key="label_a")
        text_a = st.text_area("Paste T&C A", height=200, key="text_a",
                              placeholder="Paste first document...")
        if st.button("Use Risky Sample →", key="comp_risky"):
            st.session_state.text_a = SAMPLE_RISKY
            st.session_state.label_a = "Risky Service"
            st.rerun()

    with col_b:
        st.markdown("**Document B**")
        label_b = st.text_input("Label B", "Service B", key="label_b")
        text_b = st.text_area("Paste T&C B", height=200, key="text_b",
                              placeholder="Paste second document...")
        if st.button("← Use Safe Sample", key="comp_safe"):
            st.session_state.text_b = SAMPLE_SAFE
            st.session_state.label_b = "Safe Service"
            st.rerun()

    compare_btn = st.button(
        "⚡ Compare Documents", use_container_width=True, type="primary", key="comp_go"
    )

    if compare_btn:
        if not text_a or not text_b:
            st.warning("Please paste both documents.")
        else:
            with st.spinner("Running comparison pipeline..."):
                results_a, verdict_a = analyze_document(text_a, doc_type, classifier, scorer, verdict_engine)
                results_b, verdict_b = analyze_document(text_b, doc_type, classifier, scorer, verdict_engine)
                comparison = comparator.compare(
                    results_a, results_b, label_a, label_b, text_a, text_b
                )

            st.divider()

            # ── WINNER ──
            safer = comparison["safer"]
            st.markdown(
                f'<div style="text-align:center;padding:1.5rem;background:#14532d22;'
                f'border:1px solid #22c55e44;border-radius:14px;margin-bottom:1.5rem;">'
                f'<div style="font-size:0.75rem;color:#7c7ca8;letter-spacing:2px;'
                f'text-transform:uppercase;font-family:monospace;">Safer Choice</div>'
                f'<div style="font-size:1.8rem;font-weight:700;color:#22c55e;'
                f'font-family:monospace;margin-top:6px;">✅ {safer}</div>'
                f'<div style="font-size:0.85rem;color:#7c7ca8;margin-top:4px;">'
                f'Document similarity: {comparison["doc_similarity_pct"]}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # ── SIDE BY SIDE METRICS ──
            stats_a = comparison["stats_a"]
            stats_b = comparison["stats_b"]

            col1, col_mid, col2 = st.columns([5, 2, 5])

            with col1:
                color_a = verdict_a["color"]
                st.markdown(
                    f'<div class="cg-card" style="border-left:4px solid {color_a}">'
                    f'<div style="font-family:monospace;font-size:0.75rem;color:#7c7ca8">{label_a}</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:{color_a};margin:8px 0">'
                    f'{verdict_a["label"]}</div>'
                    f'<div style="font-size:0.85rem">Avg Risk: <b>{stats_a["avg"]}</b> / 10</div>'
                    f'<div style="font-size:0.85rem">Max Risk: <b>{stats_a["max"]}</b> / 10</div>'
                    f'<div style="font-size:0.85rem">High-Risk Clauses: <b>{stats_a["count"]}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with col_mid:
                st.markdown(
                    '<div style="text-align:center;padding-top:3rem;'
                    'font-family:monospace;color:#7c7ca8;font-size:1.5rem">VS</div>',
                    unsafe_allow_html=True
                )

            with col2:
                color_b = verdict_b["color"]
                st.markdown(
                    f'<div class="cg-card" style="border-left:4px solid {color_b}">'
                    f'<div style="font-family:monospace;font-size:0.75rem;color:#7c7ca8">{label_b}</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:{color_b};margin:8px 0">'
                    f'{verdict_b["label"]}</div>'
                    f'<div style="font-size:0.85rem">Avg Risk: <b>{stats_b["avg"]}</b> / 10</div>'
                    f'<div style="font-size:0.85rem">Max Risk: <b>{stats_b["max"]}</b> / 10</div>'
                    f'<div style="font-size:0.85rem">High-Risk Clauses: <b>{stats_b["count"]}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── CATEGORY COMPARISON ──
            st.markdown("#### Category-Level Risk Breakdown")
            cat_diff = comparison["category_diff"]

            if cat_diff:
                comp_df = pd.DataFrame([
                    {
                        "Category": cat,
                        label_a: round(v["a"], 2) if v["a"] is not None else "-",
                        label_b: round(v["b"], 2) if v["b"] is not None else "-",
                    }
                    for cat, v in sorted(cat_diff.items())
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
