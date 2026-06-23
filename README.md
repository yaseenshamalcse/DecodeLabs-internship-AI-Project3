# 📄 Resume Job Matcher

> Match your resume to the best-fit job listings using **TF-IDF vectorization** and **cosine similarity** — with skill gap analysis built in.

---

## 🚀 What It Does

You paste your resume keywords → the app ranks job listings by how well they match → it shows you exactly which skills you already have and which ones you're missing.

No API. No cloud. Just smart NLP running locally.

---

## ✨ Features

- 🔍 **TF-IDF + Cosine Similarity** matching engine
- ✅ **Matched skills** highlighted in green
- ❌ **Missing skills** flagged in red (your personal learning roadmap)
- 🏢 **Company names** included in every listing
- 📊 **Match percentage** with visual score bar
- 🖥️ **CLI version** for terminal use
- 🌐 **Streamlit UI** for a polished browser experience
- 📁 **Custom CSV support** — bring your own job listings

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| scikit-learn | TF-IDF vectorizer + cosine similarity |
| pandas / numpy | Data handling |
| Streamlit | Web UI |

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/resume-job-matcher.git
cd resume-job-matcher
```


### 2. Run CLI version
```bash
python job_matcher_cli.py
```

### 3. Run Streamlit UI
```bash
streamlit run job_matcher_app.py
```
Opens at `http://localhost:8501`

---

## 💡 Example Input

```
Python, Docker, SQL, AWS, REST APIs, Git, Linux
```

## 📊 Example Output

```
#1  Data Engineer  @  DataFlow Inc
    Match: [████████████░░░░░░░░] 62.4%
    Skills to add: Kafka, Airflow, Spark

#2  Senior Python Developer  @  TechCorp
    Match: [███████████░░░░░░░░░] 58.1%
    Skills to add: Django, Redis, Testing
```


## 📂 Project Structure

```
resume-job-matcher/
│
├── job_matcher_cli.py      # Terminal version
├── job_matcher_app.py      # Streamlit UI version
└── README.md
```

---

## 🤝 Contributing

Pull requests are welcome! Ideas for improvement:
- Add resume PDF upload support
- Integrate live job listings via API (LinkedIn, Indeed)
- Add skill learning resource links

---

## 📜 License

MIT License — free to use, modify, and share.

---

*Built with Python · scikit-learn · Streamlit*
