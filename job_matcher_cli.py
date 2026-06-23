import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# 1. LOAD OR CREATE JOB LISTINGS DATASET
# ─────────────────────────────────────────

def load_or_create_jobs():
    csv_file = 'job_listings.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print("✅ Job listings loaded from 'job_listings.csv'")
        return df

    print("📄 'job_listings.csv' not found. Creating sample job listings...")
    sample_data = {
        'job_title': [
            'Senior Python Developer',
            'Data Engineer',
            'ML Engineer',
            'Full Stack Developer',
            'Cloud DevOps Engineer',
            'Data Analyst',
            'Backend Engineer',
            'AI Product Manager',
            'Cybersecurity Analyst',
            'Mobile App Developer',
            'NLP Engineer',
            'Site Reliability Engineer',
            'Business Intelligence Developer',
            'Robotics Engineer',
            'Blockchain Developer',
        ],
        'company': [
            'TechCorp', 'DataFlow Inc', 'NeuralNet Labs', 'WebForge', 'CloudBase',
            'InsightCo', 'APIWorks', 'AIVentures', 'SecureNet', 'AppCraft',
            'LinguaAI', 'Uptime Solutions', 'MetricsPro', 'RoboSystems', 'ChainTech',
        ],
        'description': [
            'Python, Django, REST APIs, PostgreSQL, Redis, Docker, Git, Agile, Testing',
            'Python, SQL, Apache Spark, Kafka, AWS, ETL, Data Warehousing, Airflow',
            'Python, TensorFlow, PyTorch, MLOps, Feature Engineering, Model Deployment, Docker',
            'JavaScript, React, Node.js, Python, SQL, REST APIs, Git, HTML, CSS, Docker',
            'AWS, Terraform, Kubernetes, Docker, CI/CD, Linux, Ansible, Monitoring, Python',
            'SQL, Python, Power BI, Tableau, Excel, Statistics, Data Visualization, Reporting',
            'Java, Spring Boot, Microservices, SQL, Docker, REST APIs, Kafka, Git',
            'Product Strategy, ML Understanding, Agile, Data Analysis, Stakeholder Management, Roadmapping',
            'Network Security, SIEM, Penetration Testing, Python, Linux, Incident Response, Compliance',
            'Flutter, Dart, iOS, Android, REST APIs, Firebase, Git, UI/UX',
            'Python, NLP, Transformers, BERT, SpaCy, NLTK, TensorFlow, Text Classification',
            'Linux, Kubernetes, Prometheus, Grafana, Python, Incident Response, SLA, Automation',
            'SQL, Power BI, Tableau, Data Modeling, ETL, Reporting, Excel, Python',
            'Python, ROS, C++, Computer Vision, Sensors, Embedded Systems, Linux, Control Systems',
            'Solidity, Ethereum, Web3.js, Smart Contracts, Python, Cryptography, DeFi, JavaScript',
        ]
    }
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_file, index=False)
    print("✅ Sample job listings created and saved as 'job_listings.csv'")
    return df

# ─────────────────────────────────────────
# 2. BUILD TF-IDF VECTORS
# ─────────────────────────────────────────

def build_vectors(df):
    corpus = df['description'].fillna('').astype(str).tolist()
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )
    job_vectors = vectorizer.fit_transform(corpus)
    return vectorizer, job_vectors

# ─────────────────────────────────────────
# 3. GET RESUME TEXT FROM USER
# ─────────────────────────────────────────

def get_resume_text():
    print("\n📝 Paste your resume keywords / skills / experience summary.")
    print("   (at least 3 keywords, separated by commas or spaces, then press Enter)\n")
    while True:
        raw = input("> ").strip()
        if not raw:
            print("⚠️  No input. Please enter something.")
            continue
        parts = [p.strip().lower() for p in re.split(r'[\s,]+', raw) if p.strip()]
        if len(parts) < 3:
            print(f"⚠️  Only {len(parts)} keyword(s). Need at least 3.")
            continue
        return parts

# ─────────────────────────────────────────
# 4. MATCH RESUME TO JOB LISTINGS
# ─────────────────────────────────────────

def match_jobs(resume_keywords, df, vectorizer, job_vectors, top_n=5):
    resume_text = ', '.join(resume_keywords)
    resume_vector = vectorizer.transform([resume_text])
    similarities = cosine_similarity(resume_vector, job_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                'rank': len(results) + 1,
                'job_title': df.iloc[idx]['job_title'],
                'company': df.iloc[idx]['company'],
                'description': df.iloc[idx]['description'],
                'match': round(similarities[idx] * 100, 1)
            })
    return results

# ─────────────────────────────────────────
# 5. HIGHLIGHT MISSING SKILLS
# ─────────────────────────────────────────

def get_missing_skills(resume_keywords, job_description):
    job_skills = {s.strip().lower() for s in job_description.split(',')}
    user_skills = set(resume_keywords)
    missing = job_skills - user_skills
    return [s.title() for s in sorted(missing)][:5]

# ─────────────────────────────────────────
# 6. DISPLAY RESULTS
# ─────────────────────────────────────────

def display_results(results, resume_keywords):
    if not results:
        print("\n❌ No matching jobs found. Try adding more relevant keywords.")
        return

    print("\n" + "=" * 60)
    print("   🎯 TOP JOB MATCHES FOR YOUR RESUME")
    print("=" * 60)

    for job in results:
        bar_len = int(job['match'] / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        missing = get_missing_skills(resume_keywords, job['description'])

        print(f"\n  #{job['rank']}  {job['job_title']}  @  {job['company']}")
        print(f"      Match: [{bar}] {job['match']}%")
        if missing:
            print(f"      Skills to add: {', '.join(missing)}")
        else:
            print("      ✅ Strong skill match!")

    print("\n" + "=" * 60)

# ─────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════╗")
    print("║    📄  RESUME JOB MATCHER            ║")
    print("║    TF-IDF × Cosine Similarity        ║")
    print("╚══════════════════════════════════════╝")

    df = load_or_create_jobs()
    vectorizer, job_vectors = build_vectors(df)
    resume_keywords = get_resume_text()

    try:
        top_n = int(input("\n  How many job matches to show? (1-10, default 5): ").strip() or 5)
        top_n = max(1, min(10, top_n))
    except ValueError:
        top_n = 5

    results = match_jobs(resume_keywords, df, vectorizer, job_vectors, top_n=top_n)
    display_results(results, resume_keywords)

    print("\n✅ Done! Tip: add missing skills to boost your match score.\n")


if __name__ == "__main__":
    main()