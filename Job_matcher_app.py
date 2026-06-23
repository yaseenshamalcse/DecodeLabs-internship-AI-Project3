import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# THEME CSS — clean dark card design
# ─────────────────────────────────────────

THEME_CSS = """
<style>
  :root {
    --bg0: #06080F;
    --bg1: #0A1020;
    --card: #0E1828;
    --muted: #8B9EB7;
    --text: #DDE6F0;
    --accent: #38BDF8;
    --accent2: #818CF8;
    --green: #34D399;
    --red: #F87171;
    --border: rgba(255,255,255,0.07);
    --shadow: 0 20px 60px rgba(0,0,0,0.5);
  }

  html, body {
    background:
      radial-gradient(800px 400px at 5% 0%, rgba(56,189,248,0.12), transparent 60%),
      radial-gradient(700px 450px at 95% 0%, rgba(129,140,248,0.12), transparent 55%),
      var(--bg0) !important;
  }
  .stApp { background: transparent; }

  .brand {
    font-size: 32px;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .brand-sub {
    color: var(--muted);
    font-size: 13px;
    margin-top: 4px;
  }
  .divider {
    height: 1px;
    background: var(--border);
    margin: 18px 0;
  }
  .card {
    background: linear-gradient(160deg, rgba(56,189,248,0.08), rgba(255,255,255,0.02));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 20px 22px;
    margin-top: 16px;
    box-shadow: var(--shadow);
  }
  .job-title {
    font-size: 17px;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.01em;
  }
  .company-name {
    font-size: 12px;
    color: var(--accent);
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
  }
  .score-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.30);
    color: var(--accent);
    font-weight: 800;
    font-size: 13px;
  }
  .score-bar-bg {
    width: 100%;
    height: 8px;
    background: rgba(148,163,184,0.15);
    border-radius: 999px;
    overflow: hidden;
    margin: 12px 0 14px;
    border: 1px solid rgba(255,255,255,0.06);
  }
  .score-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 400ms ease;
  }
  .skill-pill {
    display: inline-flex;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    margin: 3px 4px 0 0;
    user-select: none;
  }
  .pill-match {
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.28);
    color: var(--green);
  }
  .pill-missing {
    background: rgba(248,113,113,0.10);
    border: 1px solid rgba(248,113,113,0.25);
    color: var(--red);
  }
  .pill-neutral {
    background: rgba(139,158,183,0.10);
    border: 1px solid rgba(139,158,183,0.22);
    color: var(--muted);
  }
  .section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .tip-card {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 14px;
    padding: 14px 18px;
    margin-top: 14px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
  }
  @media (max-width: 768px) {
    .brand { font-size: 24px !important; }
  }
</style>
"""

# ─────────────────────────────────────────
# DATA / ML LAYER
# ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_or_create_jobs() -> pd.DataFrame:
    csv_file = "job_listings.csv"
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)

    sample_data = {
        "job_title": [
            "Senior Python Developer", "Data Engineer", "ML Engineer",
            "Full Stack Developer", "Cloud DevOps Engineer", "Data Analyst",
            "Backend Engineer", "AI Product Manager", "Cybersecurity Analyst",
            "Mobile App Developer", "NLP Engineer", "Site Reliability Engineer",
            "Business Intelligence Developer", "Robotics Engineer", "Blockchain Developer",
        ],
        "company": [
            "TechCorp", "DataFlow Inc", "NeuralNet Labs", "WebForge", "CloudBase",
            "InsightCo", "APIWorks", "AIVentures", "SecureNet", "AppCraft",
            "LinguaAI", "Uptime Solutions", "MetricsPro", "RoboSystems", "ChainTech",
        ],
        "description": [
            "Python, Django, REST APIs, PostgreSQL, Redis, Docker, Git, Agile, Testing",
            "Python, SQL, Apache Spark, Kafka, AWS, ETL, Data Warehousing, Airflow",
            "Python, TensorFlow, PyTorch, MLOps, Feature Engineering, Model Deployment, Docker",
            "JavaScript, React, Node.js, Python, SQL, REST APIs, Git, HTML, CSS, Docker",
            "AWS, Terraform, Kubernetes, Docker, CI/CD, Linux, Ansible, Monitoring, Python",
            "SQL, Python, Power BI, Tableau, Excel, Statistics, Data Visualization, Reporting",
            "Java, Spring Boot, Microservices, SQL, Docker, REST APIs, Kafka, Git",
            "Product Strategy, ML Understanding, Agile, Data Analysis, Stakeholder Management, Roadmapping",
            "Network Security, SIEM, Penetration Testing, Python, Linux, Incident Response, Compliance",
            "Flutter, Dart, iOS, Android, REST APIs, Firebase, Git, UI/UX",
            "Python, NLP, Transformers, BERT, SpaCy, NLTK, TensorFlow, Text Classification",
            "Linux, Kubernetes, Prometheus, Grafana, Python, Incident Response, SLA, Automation",
            "SQL, Power BI, Tableau, Data Modeling, ETL, Reporting, Excel, Python",
            "Python, ROS, C++, Computer Vision, Sensors, Embedded Systems, Linux, Control Systems",
            "Solidity, Ethereum, Web3.js, Smart Contracts, Python, Cryptography, DeFi, JavaScript",
        ],
    }
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_file, index=False)
    return df


@st.cache_data(show_spinner=False)
def build_vectors(df: pd.DataFrame) -> Tuple[TfidfVectorizer, object]:
    corpus = df["description"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
    )
    job_vectors = vectorizer.fit_transform(corpus)
    return vectorizer, job_vectors


def parse_resume(raw: str, min_kw: int = 3) -> Tuple[bool, list | str]:
    if not raw or not raw.strip():
        return False, "Please enter at least 3 keywords."
    parts = [p.strip().lower() for p in re.split(r"[\s,]+", raw) if p.strip()]
    if len(parts) < min_kw:
        return False, f"Only {len(parts)} keyword(s). Need at least {min_kw}."
    seen, uniq = set(), []
    for p in parts:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return True, uniq


def get_matches(resume_kw: list, df, vectorizer, job_vectors, top_n: int = 5):
    text = ", ".join(resume_kw)
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, job_vectors).flatten()
    top_idx = np.argsort(sims)[::-1][:top_n]
    results = []
    for idx in top_idx:
        score = float(sims[idx])
        if score <= 0:
            continue
        job_skills = [s.strip().lower() for s in str(df.iloc[idx]["description"]).split(",")]
        user_set = set(resume_kw)
        matched = [s for s in job_skills if s in user_set]
        missing = [s for s in job_skills if s not in user_set]
        results.append({
            "job_title": df.iloc[idx]["job_title"],
            "company": df.iloc[idx]["company"],
            "description": df.iloc[idx]["description"],
            "match": round(score * 100, 1),
            "matched_skills": matched[:8],
            "missing_skills": missing[:5],
        })
    return results


# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Resume Job Matcher",
        page_icon="📄",
        layout="centered",
    )

    st.markdown(THEME_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align:center; padding: 8px 0 4px;'>
          <div class='brand'>📄 Resume Job Matcher</div>
          <div class='brand-sub'>Paste your skills → find the best-fitting job listings using TF-IDF + cosine similarity</div>
        </div>
        <div class='divider'></div>
        """,
        unsafe_allow_html=True,
    )

    df = load_or_create_jobs()
    vectorizer, job_vectors = build_vectors(df)

    with st.form("resume_form", clear_on_submit=False):
        resume_input = st.text_area(
            "Your resume keywords / skills",
            placeholder="e.g. Python, Docker, SQL, AWS, REST APIs, Git, Linux",
            height=100,
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            top_n = st.selectbox("Show top matches", options=[3, 5, 7, 10], index=1)
        with col2:
            submit = st.form_submit_button("🔍 Find Matching Jobs", use_container_width=True)

    if not submit:
        st.markdown(
            """
            <div class='tip-card'>
              <b>💡 How it works:</b><br>
              Enter skills and keywords from your resume (comma or space separated).<br>
              The app compares them against real job descriptions using <b>TF-IDF vectorization</b> and
              <b>cosine similarity</b> to rank the best-fit roles — and shows which skills are missing.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    ok, parsed = parse_resume(resume_input, min_kw=3)
    if not ok:
        st.error(str(parsed))
        return

    with st.spinner("Matching your resume to job listings..."):
        results = get_matches(parsed, df, vectorizer, job_vectors, top_n=int(top_n))

    if not results:
        st.warning("No matches found. Try adding more relevant keywords.")
        return

    st.markdown(
        f"<div style='color:#8B9EB7; font-size:13px; margin: 12px 0 4px;'>Showing top <b>{len(results)}</b> matches for <b>{len(parsed)}</b> keywords</div>",
        unsafe_allow_html=True,
    )

    for i, job in enumerate(results, 1):
        score = job["match"]
        matched_pills = "".join(
            f"<span class='skill-pill pill-match'>✓ {s.title()}</span>"
            for s in job["matched_skills"]
        )
        missing_pills = "".join(
            f"<span class='skill-pill pill-missing'>✗ {s.title()}</span>"
            for s in job["missing_skills"]
        )

        st.markdown(
            f"""
            <div class='card'>
              <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:12px;'>
                <div>
                  <div class='job-title'>#{i} — {job['job_title']}</div>
                  <div class='company-name'>{job['company']}</div>
                </div>
                <div class='score-badge'>{score:.1f}% match</div>
              </div>

              <div class='score-bar-bg'>
                <div class='score-bar-fill' style='width:{score:.1f}%;'></div>
              </div>

              <div class='section-label'>Skills you have</div>
              <div>{matched_pills if matched_pills else "<span style='color:#8B9EB7; font-size:12px;'>None matched directly</span>"}</div>

              {'<div class="section-label" style="margin-top:10px;">Skills to add</div><div>' + missing_pills + '</div>' if missing_pills else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style='text-align:center; margin-top:30px; color:rgba(139,158,183,0.7); font-size:12px;'>
          Powered by TF-IDF & Cosine Similarity · Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()