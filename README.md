# Customer Churn Prediction API

## Overview

This project is an MLOps (Machine Learning Operations) pipeline for a customer churn prediction model. The project packages a pre-trained machine learning model into a FastAPI application, automates the build and deployment process using a Continuous Integration/Continuous Deployment (CI/CD) pipeline on GitHub Actions, and deploys the application to a cloud service (Render).

The goal is to provide a robust and automated way to serve a machine learning model as a RESTful API.

## Project Features

* **Machine Learning Model:** A pre-trained model for predicting customer churn.
* **FastAPI:** A modern, fast web framework for building the API endpoints.
* **Dockerfile:** A containerization strategy to package the application and its dependencies.
* **GitHub Actions:** An automated CI/CD pipeline that triggers on every push to the `main` branch.
* **Docker Hub:** A container registry to store the built Docker image.
* **Render:** A cloud service used to deploy the Docker image as a publicly accessible web service.

## CI/CD Pipeline

The CI/CD pipeline is configured in the `.github/workflows/main.yml` file. It performs the following steps automatically:

1.  **Checkout Code:** The workflow checks out the repository code.
2.  **Docker Login:** It logs into Docker Hub using a username and a personal access token stored as GitHub secrets.
3.  **Build & Push:** It builds the Docker image and pushes it to the Docker Hub repository.
4.  **Deployment:** Render automatically detects the new image on Docker Hub and deploys the latest version.

## Repository Structure
## Repository Structure


├── .github/
│   └── workflows/
│       └── main.yml         # GitHub Actions workflow
├── models/                  # Contains the pre-trained ML model
│   └── model.pkl
├── src/                     # Source code for the application
│   └── init.py
├── app.py                   # FastAPI application
├── Dockerfile               # Docker container definition
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation (this file)
└── ...

## How to Use the API

The deployed API is available at a public URL provided by Render. The main endpoint is /predict, which accepts a POST request with customer data in JSON format and returns a churn prediction.

### Example Request using curl

To make a prediction, you can send a POST request to the /predict endpoint.

```bash
# Example JSON data for prediction
DATA='{
  "customer_id": "123456",
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 24,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "DSL",
  "online_security": "Yes",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 84.85,
  "total_charges": 1990.5
}'

# Replace YOUR_RENDER_URL with the URL from your Render dashboard
curl -X POST -H "Content-Type: application/json" -d "$DATA" (https://churn-prediction-api-9lrl.onrender.com/)

🎯 Goals Achieved
	•	✅ Built and trained churn prediction ML model
	•	✅ Packaged code using modular structure
	•	✅ Created FastAPI app to serve model predictions
	•	✅ Containerized app with Docker
	•	✅ Deployed to cloud (Render) with live API

Khoshaba Odeesho
📍 Melbourne | 💼 Data Analyst
🔗 LinkedIn :http://linkedin.com/in/khoshaba-odeesho-17b5b92aa


-----
🏁 Future Improvements
	•	Add monitoring with Prometheus/Grafana
	•	Automate CI/CD with GitHub Actions
	•	Improve model performance using feature engineering
