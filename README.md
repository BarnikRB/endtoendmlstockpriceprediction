Here's the updated README with **Dockerization** for the cloud version:  

---

# 📈 Automated Stock Forecasting Platform  

## 🚀 Overview  
An end-to-end machine learning pipeline for **automated stock price forecasting**, leveraging **Python, FastAPI, Airflow, GCP, XGBoost, Streamlit, Docker, and Postgres**. The platform supports both **local** and **cloud** deployments.  

## 🏗️ Tech Stack  
- 🐍 **Python** – Core programming language  
- ⚡ **FastAPI** – Backend API for data retrieval and predictions  
- 🌬️ **Apache Airflow** – Workflow orchestration  
- ☁️ **Google Cloud Platform (GCP)** – Cloud infrastructure  
- 📊 **XGBoost** – Time series forecasting model  
- 🖥️ **Streamlit** – Frontend for visualization  
- 🐳 **Docker** – Containerization for cloud deployment  
- 🗄️ **Postgres (Supabase) & SQLite** – Databases for storing stock data  

## 🔥 Key Features  
✅ **Automated Data Pipeline** – Ingests stock data from Yahoo Finance using Apache Airflow.  
✅ **Machine Learning Model** – Implements an optimized XGBoost model for stock price forecasting.  
✅ **FastAPI Backend** – Serves predictions and stock data via a REST API.  
✅ **Streamlit Frontend** – Interactive UI for visualizing predictions and trends.  
✅ **Dual Deployment Support** –  
   - **Local Version**: Uses **SQLite** and runs Airflow locally.  
   - **Cloud Version**: Uses **Supabase Postgres**, **Dockerized FastAPI**, and **Google Cloud Composer** for Airflow.  
✅ **Scalable Cloud Deployment** – GCP-based infrastructure for automated ML workflows.  

## 🛠️ Installation & Setup  

### 🔹 Local Version (SQLite & Local Airflow)  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/BarnikRB/endtoendmlstockpriceprediction.git
   cd automated-stock-forecasting
   ```
2. **Set up a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
4. **Start FastAPI Backend**  
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   ```
5. **Run Streamlit app**  
   ```bash
   streamlit run app.py
   ```

### ☁️ Cloud Version (Docker, Supabase, GCP & Google Cloud Composer)  
1. **Set up a Supabase Postgres database**  
   - Create an account at [Supabase](https://supabase.com/)  
   - Get your database URL and API keys  

2. **Deploy Airflow on Google Cloud Composer**  
   - Set up a GCP project  
   - Enable Composer and create an **Airflow environment**  
   - Deploy DAGs to Composer’s **Cloud Storage bucket**  

3. **Dockerize FastAPI Backend**  
   - Create a `Dockerfile` in the project root:  
     ```dockerfile
     FROM python:3.9
     WORKDIR /app
     COPY . /app
     RUN pip install --no-cache-dir -r requirements.txt
     CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
     ```
   - Build and run the Docker container:  
     ```bash
     docker build -t stock-forecast-api .
     docker run -p 8000:8000 stock-forecast-api
     ```

4. **Deploy FastAPI to Google Cloud Run**  
   - Authenticate with GCP:  
     ```bash
     gcloud auth login
     ```
   - Build and push the Docker image to **Google Container Registry (GCR)**:  
     ```bash
     gcloud builds submit --tag gcr.io/your-project-id/stock-forecast-api
     ```
   - Deploy the container to **Cloud Run**:  
     ```bash
     gcloud run deploy stock-forecast-api --image gcr.io/your-project-id/stock-forecast-api --platform managed --allow-unauthenticated
     ```

5. **Update environment variables**  
   ```bash
   export DB_URL="your_supabase_db_url"
   export GCP_PROJECT="your_gcp_project_id"
   ```

6. **Run the Streamlit web app**  
   ```bash
   streamlit run app.py
   ```

## 📡 API Endpoints  
Once FastAPI is running, you can access the following endpoints:  

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/health` | Check API status |
| `GET` | `/stocks/{ticker}` | Get historical stock data |
| `POST` | `/predict` | Get stock price predictions |

## 📸 Screenshots  
🚀 *Coming soon...*  

## 🤝 Contributing  
Contributions are welcome! Feel free to submit a pull request.  

## 📜 License  
This project is licensed under the **MIT License**.  

---

This version includes **Dockerization** for the cloud deployment. Let me know if you'd like any modifications! 🚀😊