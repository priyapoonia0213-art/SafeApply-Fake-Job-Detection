import streamlit as st
import joblib
import json
import re
import pandas as pd
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Global CSS — Beige theme + animations ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lato:wght@300;400;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
}
.stApp {
    background-color: #F5F0E8;
    color: #3D2B1F;
}

/* ── Remove harsh top black border — replace with soft beige ── */
[data-testid="stAppViewContainer"] > section:first-child {
    border-top: none !important;
}
header[data-testid="stHeader"] {
    background-color: #F5F0E8 !important;
    border-bottom: 1.5px solid #D9C9AE !important;
    box-shadow: none !important;
}
/* Streamlit toolbar top line */
.stApp > header {
    border-top: none !important;
    border-bottom: 1.5px solid #D9C9AE !important;
    box-shadow: 0 2px 8px rgba(90,50,20,0.06) !important;
}
/* Remove any black decorative top bar */
.stApp::before,
[data-testid="stDecoration"] {
    display: none !important;
    background: none !important;
    border: none !important;
    height: 0 !important;
}

/* ── Sidebar ── */
/* ── Sidebar collapse/expand toggle — always visible ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
    opacity: 1 !important;
    visibility: visible !important;
    background: #C8B89A !important;
    border-radius: 50% !important;
    color: #3D2B1F !important;
}

[data-testid="stSidebarCollapseButton"] button svg,
[data-testid="stSidebarCollapsedControl"] button svg {
    fill: #3D2B1F !important;
    stroke: #3D2B1F !important;
}

/* This is the key — overrides Streamlit's hover-only visibility */
[data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    transition: none !important;
}
            
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #EAE0CC 0%, #D9CDB8 100%);
    border-right: 2px solid #C8B89A;
}
[data-testid="stSidebar"] * {
    color: #3D2B1F !important;
}

/* ── Sidebar radio — bigger font, clean highlight only on selected ── */
[data-testid="stSidebar"] .stRadio label {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 7px 14px !important;
    border-radius: 10px !important;
    display: block !important;
    margin: 3px 0 !important;
    transition: background 0.2s ease !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(193,127,62,0.15) !important;
}
/* Selected radio item — only background highlight, no box */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div + label,
[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ label,
[data-testid="stSidebar"] .stRadio div[role="radio"][aria-checked="true"] {
    background: rgba(193,127,62,0.22) !important;
    border-radius: 10px !important;
}
/* Target the active radio option via streamlit's internal checked state */
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(193,127,62,0.22) !important;
    color: #5C3D2E !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border-left: 3px solid #C17F3E !important;
    padding-left: 11px !important;
}
/* Remove old nth-child override */

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #5C3D2E !important;
}
h1 { font-size: 2.4rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #C17F3E, #D4924A) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-family: 'Lato', sans-serif !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(193,127,62,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 7px 20px rgba(193,127,62,0.45) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #EDE4D3;
    border-radius: 14px;
    padding: 14px 10px;
    border: 1px solid #C8B89A;
    text-align: center;
    transition: transform 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); }
[data-testid="stMetricLabel"] { color: #7A5C3E !important; font-weight: 700; }
[data-testid="stMetricValue"] { color: #3D2B1F !important; font-size: 1.4rem !important; }

/* ── Cards ── */
.beige-card {
    background: #EDE4D3;
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid #C8B89A;
    margin-bottom: 16px;
    transition: transform 0.25s, box-shadow 0.25s;
    animation: fadeSlideUp 0.5s ease both;
}
.beige-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 28px rgba(90,50,20,0.13);
}

/* ── Result boxes ── */
.result-fake {
    background: linear-gradient(135deg, #FDDEDE, #F9C4C4);
    border: 2px solid #E05252;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    animation: bounceIn 0.6s ease;
}
.result-real {
    background: linear-gradient(135deg, #D6F0E0, #B8E8CC);
    border: 2px solid #3BAA6B;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    animation: bounceIn 0.6s ease;
}
.result-fake h2 { color: #B22222 !important; font-size: 2rem !important; }
.result-real h2 { color: #1A6B3C !important; font-size: 2rem !important; }

/* ── Divider ── */
hr { border-color: #C8B89A !important; }

/* ── Text area ── */
.stTextArea textarea {
    background: #FAF6EF !important;
    border: 1.5px solid #C8B89A !important;
    border-radius: 12px !important;
    color: #3D2B1F !important;
    font-family: 'Lato', sans-serif !important;
}
            
/* ── Text area label ── */
.stTextArea label {
    color: #5C3D2E !important;
    font-weight: 600 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Footer ── */
.footer {
    margin-top: 48px;
    padding: 18px;
    text-align: center;
    border-top: 1.5px solid #C8B89A;
    color: #7A5C3E;
    font-size: 0.88rem;
    letter-spacing: 0.5px;
    animation: fadeSlideUp 0.8s ease both;
}

/* ── Animations ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes bounceIn {
    0%   { transform: scale(0.85); opacity: 0; }
    60%  { transform: scale(1.04); opacity: 1; }
    100% { transform: scale(1); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}
@keyframes spinSlow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes floatUpDown {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-10px); }
}
.animate-fade { animation: fadeSlideUp 0.6s ease both; }
.pulse { animation: pulse 2s infinite; }

/* ── Page title area ── */
.page-header {
    animation: fadeSlideUp 0.5s ease both;
    margin-bottom: 8px;
}

/* ── Flag row ── */
.flag-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    border-radius: 10px;
    margin: 5px 0;
    background: #F0E8D8;
    font-size: 0.95rem;
    transition: background 0.2s;
}
.flag-row:hover { background: #E5DAC5; }

/* ── Step pill ── */
.step-pill {
    display: inline-block;
    background: #C17F3E;
    color: #fff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 8px;
}

/* ── Sidebar title ── */
.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #5C3D2E !important;
    margin-bottom: 4px;
}

/* ── Error / 404 / Loading page styles ── */
.error-page {
    text-align: center;
    padding: 60px 20px;
    animation: fadeSlideUp 0.6s ease both;
}
.error-icon {
    font-size: 5rem;
    animation: floatUpDown 3s ease-in-out infinite;
    display: block;
    margin-bottom: 16px;
}
.error-code {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 700;
    color: #C17F3E;
    line-height: 1;
    margin-bottom: 8px;
}
.error-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #5C3D2E;
    margin-bottom: 12px;
}
.error-subtitle {
    font-size: 1rem;
    color: #7A5C3E;
    margin-bottom: 32px;
    line-height: 1.7;
}
.loading-spinner {
    width: 52px;
    height: 52px;
    border: 5px solid #D9C9AE;
    border-top: 5px solid #C17F3E;
    border-radius: 50%;
    animation: spinSlow 1s linear infinite;
    margin: 0 auto 20px auto;
}
.error-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #C17F3E, #D4924A);
    border-radius: 4px;
    margin: 0 auto 24px auto;
}
</style>
""", unsafe_allow_html=True)

# ── Footer helper ─────────────────────────────────────────────────────────────
def show_footer():
    st.markdown("""
    <div class="footer">
        Built with ❤️ using &nbsp;<strong>Scikit-learn</strong>,&nbsp;
        <strong>XGBoost</strong>,&nbsp;<strong>NLTK</strong>&nbsp;&&nbsp;<strong>Streamlit</strong>
        &nbsp;·&nbsp; Fake Job Posting Detector &nbsp;·&nbsp; 2024
    </div>
    """, unsafe_allow_html=True)

# ── NLTK ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
load_nltk()

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    with open('models/model_info.json') as f:
        info = json.load(f)
    return model, tfidf, info

try:
    model, tfidf, model_info = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR  = str(e)

# ── Text utilities ────────────────────────────────────────────────────────────
@st.cache_resource
def get_cleaner():
    return WordNetLemmatizer(), set(stopwords.words('english'))

lemmatizer, stop_words = get_cleaner()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', ' email_token ', text)
    text = re.sub(r'\d+', ' number_token ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

def extract_features(job_text):
    return pd.DataFrame({
        'text_length':      [len(job_text)],
        'word_count':       [len(job_text.split())],
        'has_company_logo': [0],
        'telecommuting':    [1 if 'remote' in job_text.lower() else 0],
        'has_questions':    [1 if '?' in job_text else 0],
        'has_salary':       [1 if 'salary' in job_text.lower() else 0],
        'has_urgent':       [1 if 'urgent' in job_text.lower() else 0],
        'has_remote':       [1 if 'remote' in job_text.lower() else 0],
        'has_email':        [1 if '@' in job_text else 0],
        'is_short_desc':    [1 if len(job_text.split()) < 50 else 0],
        'is_long_desc':     [1 if len(job_text.split()) > 500 else 0],
    })

def predict(job_text):
    cleaned  = clean_text(job_text)
    X_text   = tfidf.transform([cleaned])
    features = extract_features(job_text)
    X_final  = sp.hstack([X_text, features])
    pred     = model.predict(X_final)[0]
    prob     = model.predict_proba(X_final)[0][1]
    return pred, prob, features

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🔍 Fake Job Detector</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#7A5C3E'>Protect yourself from fraud</small>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔎 Detect Fake Job", "📊 Model Performance", "ℹ️ About Project"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='background:#D9C9AE;border-radius:12px;padding:12px;font-size:0.82rem;color:#5C3D2E;line-height:1.6'>
    ⭐ <b>Tip:</b> Head to<br>
    <span style='color:#C17F3E;font-weight:700'>🔎 Detect Fake Job</span><br>
    to check any job posting instantly!
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Powered by XGBoost + TF-IDF")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="page-header">', unsafe_allow_html=True)
    st.title("🛡️ Fake Job Posting Detector")
    st.markdown("#### Protect job seekers from fraudulent listings using Machine Learning")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    cards = [
        ("📋 The Problem", "Millions of fake job postings deceive job seekers every year, leading to financial loss and identity theft.", "#FDDEDE", "#B22222"),
        ("🤖 Our Solution", "An XGBoost classifier trained on 17,000+ real job postings that detects fraud with high accuracy.", "#D6F0E0", "#1A6B3C"),
        ("🎯 Key Metric", "Optimised for **Recall** — catching as many fake jobs as possible is the #1 priority in fraud detection.", "#FFF3CD", "#856404"),
    ]
    for col, (title, body, bg, fg) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="beige-card" style="border-left: 4px solid {fg}; background:{bg}">
                <h3 style="color:{fg} !important; font-size:1.1rem !important; margin-bottom:8px">{title}</h3>
                <p style="color:#3D2B1F; font-size:0.93rem; margin:0">{body}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚨 Common Red Flags in Fake Job Postings")
    st.markdown("<br>", unsafe_allow_html=True)

    flags = [
        ("💸", "Unrealistic Salary", "Promises $4,000–$8,000/month for zero-experience roles."),
        ("⚡", "Urgency Language", "Words like URGENT, IMMEDIATELY, LIMITED VACANCIES."),
        ("📧", "Personal Email", "Legit companies use official domain emails, not Gmail/Yahoo."),
        ("🚫", "No Interview", "'Selected immediately after submitting details' — classic scam."),
        ("🏠", "Vague Remote Work", "Work from anywhere with weekly cash payments — suspicious."),
        ("📝", "No Real Requirements", "Only asks for 'positive attitude' and 'internet connection'."),
    ]

    col1, col2 = st.columns(2)
    for i, (emoji, title, desc) in enumerate(flags):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="beige-card">
                <div style="font-size:1.5rem; margin-bottom:6px">{emoji}</div>
                <div style="font-weight:700; color:#5C3D2E; margin-bottom:4px">{title}</div>
                <div style="color:#6B4F3A; font-size:0.9rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="beige-card pulse" style="text-align:center; background: linear-gradient(135deg,#EDE0C4,#D9C9AE); border: 2px solid #C17F3E">
        <span style="font-size:1.1rem; font-weight:700; color:#5C3D2E">
        👈 Head to <span style="color:#C17F3E">🔎 Detect Fake Job</span> in the sidebar to test any job posting instantly!
        </span>
    </div>
    """, unsafe_allow_html=True)

    show_footer()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DETECT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Detect Fake Job":
    st.markdown('<div class="page-header">', unsafe_allow_html=True)
    st.title("🔎 Detect a Fake Job Posting")
    st.markdown("Paste any job posting below — our model will analyse it in seconds.")
    st.markdown('</div>', unsafe_allow_html=True)

    if not MODEL_LOADED:
        # ── Themed error page when model is missing ──────────────────────────
        st.markdown("""
        <div class="error-page">
            <span class="error-icon">🔧</span>
            <div class="error-code">Oops!</div>
            <div class="error-divider"></div>
            <div class="error-title">Model Files Not Found</div>
            <div class="error-subtitle">
                The detector couldn't load its trained model.<br>
                Please make sure the <code style="background:#EDE4D3;padding:2px 8px;border-radius:6px;color:#C17F3E">models/</code>
                folder is in the same directory as <code style="background:#EDE4D3;padding:2px 8px;border-radius:6px;color:#C17F3E">app.py</code>
                and contains:<br><br>
                📦 <strong>best_model.pkl</strong> &nbsp;·&nbsp; 📦 <strong>tfidf_vectorizer.pkl</strong> &nbsp;·&nbsp; 📄 <strong>model_info.json</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🛠️ Technical Details"):
            st.code(f"Error: {MODEL_ERROR}", language="bash")

        st.markdown("""
        <div class="beige-card" style="max-width:520px;margin:0 auto;text-align:center;border:1.5px dashed #C8B89A">
            <div style="font-size:1.4rem;margin-bottom:8px">📂</div>
            <div style="font-weight:700;color:#5C3D2E;margin-bottom:6px">Expected Folder Structure</div>
            <pre style="background:#F5F0E8;border-radius:10px;padding:14px;text-align:left;font-size:0.85rem;color:#5C3D2E;border:1px solid #D9C9AE">
your-project/
├── app.py
└── models/
    ├── best_model.pkl
    ├── tfidf_vectorizer.pkl
    └── model_info.json</pre>
        </div>
        """, unsafe_allow_html=True)
        show_footer()
        st.stop()

    FAKE_SAMPLE = """URGENT HIRING - WORK FROM HOME
Company: Global Career Solutions
Position: Data Entry Executive
Salary: $4,000 - $8,000 per month | Location: Remote

We are looking for enthusiastic candidates. No prior experience required. Freshers welcome.

Benefits: High salary | Flexible hours | Immediate joining | Weekly payments | Work from anywhere
Requirements: Basic computer knowledge | Internet connection | Positive attitude

Selection Process: No interview required. Selected immediately after submitting details.
Send resume to: quickjoboffer2026@gmail.com
Limited vacancies. URGENT REQUIREMENT!"""

    REAL_SAMPLE = """Software Engineer
Company: ABC Technologies Pvt. Ltd. | Location: Bangalore | Full-Time

ABC Technologies is seeking a Software Engineer to join our backend development team working on scalable web applications.

Responsibilities:
- Design and develop REST APIs
- Write clean, maintainable Python code
- Collaborate across teams and participate in code reviews

Requirements:
- Bachelor's degree in Computer Science
- 2+ years experience in software development
- Strong knowledge of Python, SQL, and Git

Benefits: Health insurance | Paid leave | Professional development | Hybrid work
Apply at: careers.abctech.com"""

    st.markdown("#### 🧪 Try a sample first:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚠️ Load Fake Job Sample", use_container_width=True):
            st.session_state['job_text'] = FAKE_SAMPLE
    with col2:
        if st.button("✅ Load Real Job Sample", use_container_width=True):
            st.session_state['job_text'] = REAL_SAMPLE

    st.markdown("<br>", unsafe_allow_html=True)

    job_text = st.text_area(
        "📋 Paste full job posting here:",
        value=st.session_state.get('job_text', ''),
        height=280,
        placeholder="Paste the job title, description, requirements, benefits, and contact info here..."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyse = st.button("🔍 Analyse Job Posting", type="primary", use_container_width=True)

    if analyse:
        if not job_text.strip():
            st.warning("Please paste a job posting first.")
        else:
            # ── Loading state ────────────────────────────────────────────────
            with st.spinner(""):
                st.markdown("""
                <div style="text-align:center;padding:20px 0 10px 0;animation:fadeSlideUp 0.4s ease both">
                    <div class="loading-spinner"></div>
                    <div style="color:#7A5C3E;font-size:0.95rem;font-weight:600">Analysing job posting…</div>
                    <div style="color:#A08060;font-size:0.83rem;margin-top:4px">Running NLP pipeline & XGBoost model</div>
                </div>
                """, unsafe_allow_html=True)
                pred, prob, features = predict(job_text)

            st.markdown("---")
            st.markdown("### 📋 Analysis Result")

            col1, col2 = st.columns([1, 1])

            with col1:
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-fake">
                        <h2>⚠️ FAKE JOB</h2>
                        <p style="font-size:1.1rem;color:#7B2020;margin:0">
                            Fraud Probability: <strong>{prob:.1%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-real">
                        <h2>✅ GENUINE JOB</h2>
                        <p style="font-size:1.1rem;color:#1A5C38;margin:0">
                            Fraud Probability: <strong>{prob:.1%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                bar_color = "#E05252" if prob > 0.5 else "#3BAA6B"
                st.markdown(f"""
                <div style="margin-top:14px">
                    <div style="font-size:0.85rem;color:#7A5C3E;margin-bottom:4px">Fraud Risk Meter</div>
                    <div style="background:#D9C9AE;border-radius:10px;height:22px;overflow:hidden">
                        <div style="background:{bar_color};width:{prob*100:.1f}%;height:22px;border-radius:10px;
                                    transition:width 1s ease;display:flex;align-items:center;justify-content:center">
                            <span style="color:#fff;font-size:0.78rem;font-weight:700">{prob*100:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("**🔬 Features Detected in this Posting:**")
                feat = features.iloc[0]
                flag_map = [
                    ('has_urgent',    '⚡ Urgency language',      feat['has_urgent'],    True),
                    ('has_email',     '📧 Personal email found',  feat['has_email'],     True),
                    ('has_remote',    '🏠 Remote work claim',     feat['has_remote'],    False),
                    ('has_salary',    '💸 Salary mentioned',      feat['has_salary'],    False),
                    ('has_questions', '❓ Questions present',     feat['has_questions'], False),
                    ('is_short_desc', '📄 Very short description',feat['is_short_desc'], True),
                ]
                for key, label, val, is_red in flag_map:
                    if val:
                        dot = "🔴" if is_red else "🟢"
                    else:
                        dot = "⚪"
                    st.markdown(f"""
                    <div class="flag-row">
                        <span>{dot}</span>
                        <span style="color:#3D2B1F">{label}: <strong>{'Yes' if val else 'No'}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)

                wc = int(feat['word_count'])
                st.markdown(f"""
                <div class="flag-row" style="margin-top:8px">
                    📝 <span>Word count: <strong>{wc}</strong> words</span>
                </div>
                """, unsafe_allow_html=True)

    show_footer()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<div class="page-header">', unsafe_allow_html=True)
    st.title("📊 Model Performance")
    st.markdown("How our three models compare on the fake job dataset.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    data = {
        'Model':     ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy':  [0.9595, 0.9832, 0.9762],
        'Precision': [0.5504, 0.9829, 0.7178],
        'Recall':    [0.8844, 0.6647, 0.8382],
        'F1 Score':  [0.6785, 0.7931, 0.7733],
        'ROC-AUC':   [0.9842, 0.9900, 0.9864],
    }
    df_results = pd.DataFrame(data)

    st.markdown("### 🏆 Best Model: XGBoost")
    if MODEL_LOADED:
        c1, c2, c3, c4, c5 = st.columns(5)
        for (label, val), col in zip([
            ("Accuracy",  model_info['accuracy']),
            ("Precision", model_info['precision']),
            ("Recall",    model_info['recall']),
            ("F1 Score",  model_info['f1_score']),
            ("ROC-AUC",   model_info['roc_auc']),
        ], [c1,c2,c3,c4,c5]):
            col.metric(label, f"{val:.4f}")
    else:
        # ── Loading/error state for metrics ──────────────────────────────────
        st.markdown("""
        <div class="beige-card" style="text-align:center;border:1.5px dashed #C8B89A;padding:28px">
            <div class="loading-spinner" style="margin-bottom:14px"></div>
            <div style="color:#7A5C3E;font-weight:600;font-size:0.97rem">Live metrics unavailable — model not loaded</div>
            <div style="color:#A08060;font-size:0.85rem;margin-top:6px">Showing static comparison data below</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Full Comparison Table")
    st.dataframe(
        df_results.set_index('Model').style
            .highlight_max(axis=0, color='#C8E6C9')
            .highlight_min(axis=0, color='#FFCDD2')
            .format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### 📊 Visual Comparison")

    matplotlib.rcParams['axes.facecolor'] = '#F5F0E8'
    matplotlib.rcParams['figure.facecolor'] = '#F5F0E8'
    matplotlib.rcParams['text.color'] = '#3D2B1F'
    matplotlib.rcParams['axes.labelcolor'] = '#3D2B1F'
    matplotlib.rcParams['xtick.color'] = '#3D2B1F'
    matplotlib.rcParams['ytick.color'] = '#3D2B1F'

    metrics_to_plot = ['Recall', 'F1 Score', 'Precision', 'ROC-AUC']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    colors = ['#C17F3E', '#7A5C3E', '#D4924A']

    for ax, metric in zip(axes, metrics_to_plot):
        bars = ax.bar(df_results['Model'], df_results[metric], color=colors, edgecolor='#EDE4D3', linewidth=1.2)
        ax.set_title(metric, fontweight='bold', fontsize=11, color='#5C3D2E')
        ax.set_ylim(0, 1.15)
        ax.set_xticklabels(df_results['Model'], rotation=15, ha='right', fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#C8B89A')
        for bar, val in zip(bars, df_results[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#5C3D2E')
        ax.grid(axis='y', alpha=0.3, color='#C8B89A')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 💡 Why XGBoost?")
    col1, col2, col3 = st.columns(3)
    why_cards = [
        ("❌ Logistic Regression", "Best recall (0.88) but precision is only 55% — too many false alarms. F1 of 0.68 is too weak for deployment.", "#FDDEDE", "#B22222"),
        ("❌ Random Forest", "Best accuracy & precision but recall of only 0.66 — misses 1 in 3 fake jobs! Defeats the purpose entirely.", "#FDDEDE", "#B22222"),
        ("✅ XGBoost — Winner", "Recall 0.84 · Precision 0.72 · F1 0.77 · ROC-AUC 0.986. Best overall balance. Handles class imbalance natively.", "#D6F0E0", "#1A6B3C"),
    ]
    for col, (title, body, bg, fg) in zip([col1, col2, col3], why_cards):
        with col:
            st.markdown(f"""
            <div class="beige-card" style="background:{bg}; border-left:4px solid {fg}">
                <div style="font-weight:700;color:{fg};margin-bottom:8px">{title}</div>
                <div style="font-size:0.9rem;color:#3D2B1F">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    show_footer()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.markdown('<div class="page-header">', unsafe_allow_html=True)
    st.title("ℹ️ About This Project")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div class="beige-card" style="border-left:4px solid #C17F3E">
        <h3 style="color:#5C3D2E !important; margin-bottom:8px">📌 Project Overview</h3>
        <p style="color:#3D2B1F">
        This project detects <strong>fake job postings</strong> using NLP and Machine Learning.
        Built as a resume project to demonstrate a full end-to-end ML pipeline — from raw data to a deployed web app.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="beige-card">
            <h3 style="color:#5C3D2E !important">📂 Dataset</h3>
            <ul style="color:#3D2B1F; font-size:0.93rem">
                <li><b>Source:</b> Kaggle — Real or Fake Job Postings</li>
                <li><b>Size:</b> ~17,880 job postings</li>
                <li><b>Target:</b> fraudulent (0 = Real, 1 = Fake)</li>
                <li><b>Class Imbalance:</b> ~95% Real, ~5% Fake</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="beige-card">
            <h3 style="color:#5C3D2E !important">🛠️ Tech Stack</h3>
            <ul style="color:#3D2B1F; font-size:0.93rem">
                <li><b>ML:</b> Scikit-learn, XGBoost, NLTK</li>
                <li><b>NLP:</b> TF-IDF, Lemmatization, SMOTE</li>
                <li><b>App:</b> Streamlit, Joblib</li>
                <li><b>Data:</b> Pandas, NumPy, SciPy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔧 ML Pipeline")

    steps = [
        ("1", "Data Cleaning",        "Filled nulls, dropped irrelevant columns (job_id, salary_range, department)"),
        ("2", "Text Preprocessing",   "Lowercasing → Email/Number tokenization → Stopword removal → Lemmatization"),
        ("3", "Feature Engineering",  "11 extra features: text_length, word_count, has_urgent, has_email, has_remote, etc."),
        ("4", "TF-IDF Vectorization", "20,000 features, n-grams (1,3), sublinear TF scaling"),
        ("5", "Feature Combination",  "Sparse TF-IDF matrix + engineered features combined via scipy hstack"),
        ("6", "Class Imbalance",      "SMOTE — Synthetic Minority Oversampling Technique"),
        ("7", "Model Training",       "Logistic Regression, Random Forest, XGBoost compared"),
        ("8", "Model Selection",      "XGBoost selected for best Recall + F1 balance"),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="beige-card" style="padding:14px 18px; margin-bottom:10px">
            <span class="step-pill">{num}</span>
            <strong style="color:#5C3D2E">{title}</strong>
            <span style="color:#6B4F3A; font-size:0.9rem"> — {desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ▶️ How to Run")
    st.code("streamlit run app.py", language="bash")

    show_footer()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — 404 / UNKNOWN (fallback)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="error-page">
        <span class="error-icon">🗺️</span>
        <div class="error-code">404</div>
        <div class="error-divider"></div>
        <div class="error-title">Page Not Found</div>
        <div class="error-subtitle">
            Looks like this page doesn't exist (or got lost in the job market).<br>
            Use the sidebar to navigate back to a real destination.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div class="beige-card" style="text-align:center;border:1.5px dashed #C8B89A">
            <div style="font-size:1.1rem;font-weight:700;color:#5C3D2E;margin-bottom:12px">Where would you like to go?</div>
        </div>
        """, unsafe_allow_html=True)

    show_footer()