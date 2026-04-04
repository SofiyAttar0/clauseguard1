# ⚖️ ClauseGuard — ML-Powered Terms & Conditions Analyzer

> Explainable machine learning that reads T&C documents so you don't have to.

**No LLMs. No API calls. 100% local ML.**

---

## What It Does

| Feature | Technique | What You Get |
|---|---|---|
|  Summarization | TF-IDF Extractive | Key points in plain English |
|  Risk Analysis | SVM + TF-IDF + Decision Tree | Clause-by-clause risk scores + verdict |
|  Ask Questions | Cosine Similarity Search | Semantic Q&A over your document |
|  Auto Questions | Rule-based NLP | Smart questions the app generates for you |
|  Compare | Cosine + Score Diff | Side-by-side risk comparison |

---

## Project Structure

```
clauseguard/
├── app.py                          ← Streamlit frontend (main entry point)
├── requirements.txt                ← Python dependencies
├── README.md
│
├── src/
│   ├── models/
│   │   ├── models.py               ← SVM classifier, Risk Scorer, Verdict Engine
│   │   └── evaluate.py             ← Full evaluation suite (use in notebook)
│   ├── nlp/
│   │   └── nlp_engine.py           ← Summarizer, Q&A, Question Suggester, Comparator
│   └── utils/
│       └── data_loader.py          ← ToS;DR API + sample dataset
│
├── notebooks/
│   └── evaluation.ipynb            ← Jupyter evaluation notebook (your credibility doc)
│
├── data/                           ← CSV data saved here after first API fetch
├── models/                         ← Trained model .pkl files saved here
└── outputs/                        ← Evaluation plots saved here
```

---

## Quickstart — Run on Your Computer

### Step 1 — Clone / download the project

```bash
# If using git
git clone https://github.com/YOUR_USERNAME/clauseguard.git
cd clauseguard

# Or just navigate to the folder
cd clauseguard
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the app

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

That's it. Models train in ~5 seconds on first launch (cached after that).

---

## Run the Evaluation Notebook

```bash
pip install jupyter
cd notebooks
jupyter notebook evaluation.ipynb
```

Run all cells top to bottom. Plots save to `outputs/`.

---

## Deployment

### Option A — Streamlit Community Cloud (FREE, recommended)

1. Push your project to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/clauseguard.git
   git push -u origin main
   ```

2. Go to **https://share.streamlit.io**

3. Click **"New app"** → Select your repo → Set main file to `app.py`

4. Click **Deploy** — live URL in ~2 minutes, free forever.

> ✅ No server. No config. Just paste your GitHub URL.

---

### Option B — Render (Free tier)

1. Push to GitHub (same as above)

2. Go to **https://render.com** → New → Web Service

3. Connect your GitHub repo

4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

5. Deploy → free URL like `clauseguard.onrender.com`

---

### Option C — Railway

1. Go to **https://railway.app**
2. New Project → Deploy from GitHub repo
3. Add environment variable: `PORT=8501`
4. Set start command: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
5. Deploy

---

### Option D — Local network sharing (demo on same WiFi)

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Anyone on the same WiFi can access at `http://YOUR_IP:8501`

Find your IP:
- Windows: `ipconfig` → look for IPv4 Address
- Mac/Linux: `ifconfig` → look for inet

---

## Use the Real Dataset (ToS;DR API)

By default the app uses the built-in labeled sample data for speed.

To fetch real data from 120+ real services:

In `app.py`, change line in `get_models()`:
```python
df = load_data(use_api=True)   # change False → True
```

Or in the notebook:
```python
df = load_data(use_api=True)
```

Data is cached to `data/clauses.csv` so it only fetches once.

---

## Model Details

### Clause Classifier
- **Algorithm:** Support Vector Machine (SVM), linear kernel
- **Features:** TF-IDF with bigrams, 8000 features, sublinear TF
- **Labels:** 9 clause categories (data-sharing, tracking, etc.)
- **Evaluation:** 5-fold stratified cross-validation, F1 macro

### Risk Scorer
- **Algorithm:** Linear Regression on human risk ratings (1–10)
- **Features:** TF-IDF + keyword boosting
- **Labels:** Human ratings from ToS;DR (Good=1, Neutral=3, Bad=7, Very Bad=10)
- **Evaluation:** MAE, RMSE, R²

### Verdict Engine
- **Algorithm:** Decision Tree (max_depth=5), fully explainable rules
- **Input features:** avg_risk, max_risk, high_risk_count, critical_category, very_bad_count
- **Output:** Safe / Caution / Do Not Sign + reasoning

### Summarizer
- **Algorithm:** TF-IDF sentence scoring + legal keyword boosting
- **Type:** Extractive (no hallucination, reproducible)

### Semantic Search (Q&A)
- **Algorithm:** TF-IDF vectorization + cosine similarity
- **Type:** Dense retrieval over clause index

---

## Why This Is Stronger Than RAG

| | ClauseGuard | RAG + 3 prompt files |
|---|---|---|
| Explainability | ✅ Decision tree — rules visible | ❌ Black box |
| Evaluation metrics | ✅ F1, MAE, R², confusion matrix | ❌ None |
| Works offline | ✅ Fully local | ❌ Needs API key |
| Training data | ✅ Labeled legal corpus | ❌ None |
| Novel contribution | ✅ Trained SVM, custom scorer | ❌ Prompt engineering |
| Cost in production | ✅ Free forever | ❌ Per API call |
| "How does it work?" | ✅ Full explanation | ❌ "We use GPT" |

---

## Tech Stack

- Python 3.10+
- scikit-learn (SVM, Regression, Decision Tree, TF-IDF)
- Streamlit (frontend)
- pandas, numpy, matplotlib, seaborn
- Jupyter (evaluation notebook)
