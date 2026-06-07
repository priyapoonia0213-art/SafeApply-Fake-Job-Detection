# 🛡️ SafeApply — Fake Job Posting Detector

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=flat-square)

An end-to-end NLP + Machine Learning pipeline that detects fraudulent job postings — protecting job seekers before they apply.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Handling Class Imbalance](#-handling-class-imbalance--smote)
- [Models and Results](#-models-and-results)
- [Why XGBoost?](#-why-xgboost)
- [Streamlit Web App](#-streamlit-web-app)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)

---

## 🎯 Problem Statement

Fake job postings deceive thousands of job seekers daily — stealing personal information, charging illegal upfront fees, and wasting precious time. SafeApply uses machine learning to flag suspicious listings automatically, before a job seeker applies.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — [Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) |
| Total Rows | 17,880 job postings |
| Total Columns | 18 features |
| Real Jobs | 17,014 → 95.1% |
| Fake Jobs | 866 → 4.9% |
| Target Column | `fraudulent` (0 = Real, 1 = Fake) |
| Text Columns Used | `title`, `company_profile`, `description`, `requirements`, `benefits` |

The dataset has significant class imbalance — only 4.9% of postings are fake. This means a naive model that always predicts "Real" would get 95% accuracy while catching zero fraud. This is exactly why we use **F1-Score and Recall** as our primary metrics, and apply **SMOTE** to balance training data.

**Missing values (before cleaning):**

| Column | Missing |
|---|---|
| salary_range | 15,012 |
| department | 11,547 |
| required_education | 8,105 |
| benefits | 7,212 |
| company_profile | 3,308 |
| requirements | 2,696 |

Columns `salary_range`, `department`, and `job_id` were dropped. All remaining text columns were filled with empty strings.

---

## 🔧 Project Pipeline

```
Raw CSV  (fake_job_postings.csv)
     │
     ▼
1.  Load Dataset — kagglehub download, 17,880 rows × 18 columns
     │
     ▼
2.  Drop Columns — job_id, salary_range, department (too sparse / irrelevant)
     │
     ▼
3.  Fill Nulls — all text columns filled with ""
     │
     ▼
4.  Combine Text — title + company_profile + description +
                   requirements + benefits + employment_type +
                   required_experience + required_education +
                   industry + function + location  →  combined_text
     │
     ▼
5.  Clean Text — lowercase → email_token / number_token substitution
                 → remove special chars → stopword removal → lemmatization
     │
     ▼
6.  Feature Engineering — 11 extra numerical features + TF-IDF (20,000 features,
                          n-grams 1–3, sublinear_tf=True) → scipy hstack
     │
     ▼
7.  Train-Test Split — 80/20, stratify=y (preserves 4.9% fake ratio)
     │
     ▼
8.  SMOTE — training set only: 693 fake → 13,611 fake (balanced 50/50)
     │
     ▼
9.  Train 3 Models — Logistic Regression · Random Forest · XGBoost
     │
     ▼
10. Evaluate + Select — XGBoost chosen (best F1 + Recall balance)
     │
     ▼
11. Save — best_model.pkl · tfidf_vectorizer.pkl · model_info.json
     │
     ▼
12. Streamlit App — paste job posting → instant prediction + fraud flags
```

---

## 📈 Exploratory Data Analysis

**Class Distribution**

| Label | Count | Share |
|---|---|---|
| Real (0) | 17,014 | 95.1% |
| Fake (1) | 866 | 4.9% |

A naive "always Real" classifier scores 95.1% accuracy — but catches zero fraud. This confirms why accuracy alone is a misleading metric for this task.

**Text Length Statistics (after cleaning)**

| Stat | Value |
|---|---|
| Mean text length | 2,205 characters |
| Mean word count | 287 words |
| Min word count | 2 words |
| Max word count | 1,531 words |

Real jobs tend to be longer and more detailed — confirmed by the word count feature analysis.

**Top Industries Targeted by Fake Jobs** — Oil & Energy, Marketing, IT, Retail, and Financial Services appear most frequently in fraudulent postings.

**Word Cloud Observation** — Fake and real jobs use overlapping vocabulary (`work`, `team`, `experience`, `company`). Simple keyword filters cannot distinguish them; statistical ML patterns are necessary.

---

## ⚙️ Feature Engineering

**TF-IDF Vectorization**

| Parameter | Value | Reason |
|---|---|---|
| max_features | 20,000 | Top 20,000 most informative tokens |
| ngram_range | (1, 3) | Unigrams + bigrams + trigrams |
| min_df | 2 | Remove extremely rare tokens |
| max_df | 0.95 | Remove near-universal tokens |
| sublinear_tf | True | Log scaling to reduce very frequent words |
| Fit on | Training data only | Prevents data leakage |

**11 Engineered Numerical Features** (combined with TF-IDF via `scipy.sparse.hstack`):

| Feature | Type | Signal |
|---|---|---|
| `text_length` | Integer | Fake jobs tend to be shorter and vaguer |
| `word_count` | Integer | Real jobs use more detailed language |
| `has_company_logo` | Binary | Fake jobs less often have verified logos |
| `telecommuting` | Binary | Remote-only claims are a mild signal |
| `has_questions` | Binary | Legitimate jobs usually include screening questions |
| `has_salary` | Binary | Presence of salary/pay/compensation keywords |
| `has_urgent` | Binary | Urgent/immediately/hurry language |
| `has_remote` | Binary | Remote/work-from-home claims |
| `has_email` | Binary | Personal email address present |
| `is_short_desc` | Binary | Word count < 50 |
| `is_long_desc` | Binary | Word count > 500 |

**Total input features: 20,011** (20,000 TF-IDF + 11 engineered)

---

## ⚖️ Handling Class Imbalance — SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic fake job examples in the training set only.

```
Before SMOTE (training set):
  Real jobs :  13,611  (95.1%)
  Fake jobs :     693   (4.9%)   ← model barely sees fake examples

After SMOTE (training set):
  Real jobs :  13,611  (50%)
  Fake jobs :  13,611  (50%)    ← model learns fake patterns properly

Test set (never touched):
  Real jobs :   3,403  (95.2%)  ← real-world distribution for honest evaluation
  Fake jobs :     173   (4.8%)
```

SMOTE is applied **only on training data**. Applying it on test data would be data leakage.

---

## 🤖 Models and Results

All three models trained on SMOTE-balanced data, evaluated on the original untouched test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9595 | 0.5504 | 0.8844 | 0.6785 | 0.9842 |
| Random Forest | 0.9832 | 0.9829 | 0.6647 | 0.7931 | 0.9900 |
| **XGBoost ✅** | **0.9762** | **0.7178** | **0.8382** | **0.7733** | **0.9864** |

**XGBoost Final Classification Report (test set, threshold = 0.5):**

```
              precision    recall  f1-score   support

    Real Job       0.99      0.98      0.99      3403
    Fake Job       0.72      0.84      0.77       173

    accuracy                           0.98      3576
   macro avg       0.85      0.91      0.88      3576
weighted avg       0.98      0.98      0.98      3576
```

**XGBoost Hyperparameters (found via RandomizedSearchCV, 3-fold CV):**

```python
Best CV F1: 0.7809
Best params: {
    'n_estimators':     200,
    'max_depth':        8,
    'learning_rate':    0.1,
    'subsample':        0.7,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'scale_pos_weight': 19.6   # neg/pos ratio from original training data
}
```

**Threshold Analysis:**

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| 0.3 | 0.8728 | 0.7295 | 0.7947 |
| 0.4 | 0.8555 | 0.8000 | 0.8268 |
| 0.5 | 0.8382 | 0.8529 | 0.8455 |

---

## 🏆 Why XGBoost?

| Model | Strength | Weakness | Verdict |
|---|---|---|---|
| Logistic Regression | Highest Recall (0.8844) | Precision only 0.55 — too many false alarms | ❌ Flags too many real jobs |
| Random Forest | Highest Precision (0.9829) | Recall only 0.6647 — misses 1 in 3 fake jobs | ❌ Lets too many frauds through |
| XGBoost | Best F1 (0.7733) · Recall 0.8382 | Middle precision | ✅ Best practical balance |

For fraud detection, **Recall matters more than Precision**. Missing a fake job is more harmful than occasionally flagging a real one. XGBoost catches 83.8% of all fake jobs while keeping false alarms manageable.

---

## 🧪 Model Test Results

| Input | Prediction | Fake Probability |
|---|---|---|
| "URGENT HIRING — Data Entry, $4000–$8000/month, no interview, gmail contact" | ⚠️ FAKE JOB | 0.7609 |
| "Software Engineer, ABC Technologies, Bangalore, 2+ years Python/SQL" | ✅ GENUINE JOB | 0.0129 |

---

## 🌐 Streamlit Web App

A 4-page interactive app built with Streamlit:

| Page | Description |
|---|---|
| 🏠 Home | Common red flags, project overview |
| 🔎 Detect Fake Job | Paste any job posting → instant prediction + fraud probability gauge + feature flags |
| 📊 Model Performance | Full comparison of all 3 models — metrics table, bar charts |
| ℹ️ About Project | Full ML pipeline walkthrough, dataset info, tech stack |

---

## 📁 Project Structure

```
SafeApply-Fake-Job-Detection/
│
├── 📓 notebooks/
│   └── FinalFile.ipynb              ← Full Google Colab training notebook
│
├── 🌐 app/
│   └── app.py                       ← Streamlit web app
│
├── 🤖 models/
│   ├── best_model.pkl               ← Saved XGBoost model
│   ├── tfidf_vectorizer.pkl         ← Saved TF-IDF vectorizer (20,000 features)
│   └── model_info.json              ← Model performance summary
│
├── 📊 assets/
│   └── *.png                        ← EDA charts, confusion matrix, ROC curves
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

**Clone and install:**

```bash
git clone https://github.com/saloni-78/SafeApply-Fake-Job-Detection.git
cd SafeApply-Fake-Job-Detection
pip install -r requirements.txt
streamlit run app/app.py
```

**Train from scratch (Google Colab):**

1. Open `notebooks/FinalFile.ipynb` in Google Colab
2. Download `fake_job_postings.csv` from Kaggle and upload to your Drive
3. Run all cells — model files are saved to `models/`
4. Download `best_model.pkl`, `tfidf_vectorizer.pkl`, `model_info.json` → place in `models/`

---

## 🛠️ Tech Stack

| Category | Library |
|---|---|
| Data | pandas, numpy |
| NLP | TfidfVectorizer (scikit-learn), NLTK (lemmatization, stopwords) |
| ML | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Imbalance | imbalanced-learn (SMOTE) |
| Sparse Matrix | scipy (hstack) |
| Hyperparameter Tuning | RandomizedSearchCV (3-fold stratified CV) |
| Visualization | matplotlib, seaborn, wordcloud |
| Web App | Streamlit |
| Model Saving | joblib |
| Environment | Google Colab |

---

## 💡 Key Learnings

- **Accuracy is misleading on imbalanced data** — 95% accuracy can mean zero fraud caught. Always use F1-Score and Recall for fraud detection.
- **SMOTE on training data only** — applying it to test data causes data leakage and inflated metrics.
- **TF-IDF fit on training data only** — same reason: prevents test set contamination.
- **Bigrams carry more signal** than unigrams alone — "no experience" and "urgent hiring" are far more informative than their individual words.
- **Random Forest had the highest precision but lowest recall** — the worst possible trade-off for fraud detection, where missing a scam is more costly than a false alarm.
- **Threshold tuning matters** — XGBoost at threshold 0.4 gives better F1 (0.8268) than the default 0.5.
- **scale_pos_weight = 19.6** — the class ratio from the original training data, passed to XGBoost so it natively handles imbalance even before SMOTE.

---

## ⚠️ Limitations

- Dataset is primarily English-language Western job postings — regional or vernacular scam patterns may not be well represented.
- The model does not use `employment_type`, `industry`, or `required_experience` as label-encoded features — these could add signal in a future version.
- No SHAP explainability currently — a future addition to show which words drove each individual prediction.

---

## 👤 Authors

**Priya · Bhawana · Saloni**

Built with 🤍 using Python · Scikit-learn · XGBoost · Streamlit