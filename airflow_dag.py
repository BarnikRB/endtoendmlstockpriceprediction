from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import requests
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define API endpoint
API_URL = "http://your-fastapi-app:8000"  


# Define functions to interact with the API endpoints

def update_data():
    """
    Calls the /update_data endpoint to fetch and update data for all tickers.
    """
    endpoint = f"{API_URL}/update_data"
    try:
        response = requests.post(endpoint)
        response.raise_for_status()  
        logger.info(f"Data update successful: {response.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during data update: {e}")
        raise  

def retrain_models():
    """
    Calls the /retrain endpoint to retrain all models.
    """
    endpoint = f"{API_URL}/retrain"
    try:
        response = requests.post(endpoint)
        response.raise_for_status()
        logger.info(f"Model retraining successful: {response.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during model retraining: {e}")
        raise


# Define the DAG
dag = DAG(
    'stock_model_pipeline',
    default_args=default_args,
    description='Refetches data and retrains models periodically',
    schedule_interval='0 0 * * *', 
    catchup=False,  
    tags=['stock', 'model', 'retrain'],
)

# Define tasks
with dag:

    # Task 1: Update data
    update_data_task = PythonOperator(
        task_id='update_data',
        python_callable=update_data,
        dag=dag,
    )

    # Task 2: Retrain models
    retrain_models_task = PythonOperator(
        task_id='retrain_models',
        python_callable=retrain_models,
        dag=dag,
    )


    # Define task dependencies
    update_data_task >> retrain_models_task