# 🚀 Churn Prediction MLOps Pipeline

### The Challenge
In the highly competitive telecommunications market, retaining customers is crucial. This project addresses the challenge of predicting which customers are likely to churn (leave the service) by building a robust MLOps pipeline.

### The Solution: A Fully Operational MLOps Pipeline
This project is an end-to-end MLOps solution. It integrates a machine learning model, a powerful FastAPI backend, and an interactive Streamlit frontend. Everything is containerized using Docker to ensure consistency and easy deployment.

You can view and interact with the live application deployed for the world to see:
**[https://churn-mlops-pipeline.streamlit.app/](https://churn-mlops-pipeline.streamlit.app/)**

### 📁 Project Structure

The project is thoughtfully organized to manage the entire lifecycle of the machine learning model:

-   **`data/`**: Stores the raw and processed datasets.
-   **`models/`**: Houses the trained machine learning model, saved as a `.pkl` file.
-   **`src/`**: Contains the core logic.
    -   `src/predict.py`: The FastAPI application that loads the model and serves predictions.
-   **`front-end/`**: The Streamlit app that provides a beautiful user interface.
-   **`docker-compose.yml`**: The "blueprint" for running our multi-container application.
-   **`requirements.txt`**: A list of all Python libraries needed for the project.

### 🛠️ How to Run Locally

If you prefer to run this amazing project on your own machine, follow these simple steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Assyrian91/churn-mlops-pipeline.git](https://github.com/Assyrian91/churn-mlops-pipeline.git)
    ```
2.  **Navigate into the project folder:**
    ```bash
    cd churn-mlops-pipeline
    ```
3.  **Launch the pipeline with Docker:**
    ```bash
    docker-compose up --build
    ```
    This command will build and start both the frontend and backend services.

Once the services are up and running, you can access them here:
-   **Frontend App (Streamlit):** [http://localhost:8501](http://localhost:8501)
-   **Backend API Documentation (FastAPI):** [http://localhost:8000/docs](http://localhost:8000/docs)