# 🧠 Churn Prediction – MLOps Pipeline 🚀

A full-stack machine learning pipeline for predicting customer churn, built with **FastAPI, Streamlit, Docker, and CI/CD on GitHub Actions**.

---

## 🔴 Live Demo
- 📊 **Streamlit UI:** [churn-mlops-pipeline.streamlit.app](https://churn-mlops-pipeline.streamlit.app)  
- 🌐 **HTML Frontend (Render):** [churn-prediction-api-front-end.onrender.com](https://churn-prediction-api-front-end.onrender.com)  
- 🧬 **GitHub Repo:** [Churn Prediction API](https://github.com/Assyrian91/churn-prediction-api)

---

## 🧱 Project Architecture

```
churn-mlops-pipeline/
│
├── data/             # Raw and cleaned datasets
├── notebooks/        # Exploratory data analysis & experiments
├── models/           # Trained models (joblib format)
├── src/              # FastAPI backend for predictions
├── front-end/        # Streamlit app (user interface)
├── Dockerfile        # Backend Docker build
├── docker-compose.yml  # Combine frontend + backend containers
└── .github/workflows/  # CI/CD automation (build, test, deploy)
```

---

## 🐳 Run Locally with Docker

1. Clone the repo:
```bash
git clone https://github.com/Assyrian91/churn-mlops-pipeline.git
cd churn-mlops-pipeline
```

2. Build and run:
```bash
docker-compose up --build
```

3. Access:
- UI → `http://localhost:8501`  
- API → `http://localhost:8000/docs`

---

## ⚙️ CI/CD Pipeline

Automated with **GitHub Actions**, triggered on every `push`:
- ✅ Linting & testing  
- 🐳 Docker image build  
- 🚀 Deploy to Render or Streamlit Cloud

---

## 🧰 Tools & Technologies

| Area        | Stack                             |
|-------------|------------------------------------|
| ML          | scikit-learn, Pandas               |
| API         | FastAPI                            |
| UI          | Streamlit, HTML/CSS (Render)       |
| DevOps      | Docker, Docker Compose             |
| Automation  | GitHub Actions (CI/CD)             |
| Hosting     | Streamlit Cloud, Render            |

---

## 📌 About

Built by [Khoshaba Odeesho](https://github.com/Assyrian91) as a real-world ML pipeline project.  
Ready for production, extendable, and fully containerized.