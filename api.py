from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import logging
import os
import pickle
from datetime import datetime, timedelta, date
import sys
import sqlite3
from typing import List

from etl_pipeline.etl_class_sqlite import StockDataPipeline
from model_pipeline.model_trainer import StockModelTrainer

# Setup Logger
logger = logging.getLogger("fastapi")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


app = FastAPI()

# Database and trainer config
DATABASE_PATH = "./data/stock_data.db"
local_db_config = {"database": DATABASE_PATH}

pipeline = StockDataPipeline("local", local_db_config)
trainer = StockModelTrainer(DATABASE_PATH)


class PredictionRequest(BaseModel):
    ticker: str

class AddTickerRequest(BaseModel):
    ticker:str
    
class HistoricalDataResponse(BaseModel):
    dates: List[str]
    close: List[float]


@app.post("/predict", response_model=dict)
async def predict(request: PredictionRequest):
    ticker = request.ticker.lower()
    logger.info(f"Prediction requested for ticker: {ticker}")
    try:
        model_path = _get_latest_model_path(ticker)
        if not model_path:
           raise HTTPException(status_code=404, detail="Model not found for this ticker")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        last_data = _get_last_datapoints(ticker)
        if last_data is None:
            raise HTTPException(status_code=404, detail="No data found in database to make prediction")
        predictions = _make_autoregressive_predictions(model, last_data)
        return {"ticker": ticker, "predictions": predictions}

    except Exception as e:
        logger.error(f"Error during prediction for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/add", status_code=200)
async def add_ticker(request:AddTickerRequest):
   ticker = request.ticker
   logger.info(f"Add Ticker request for: {ticker}")
   if not pipeline.add_new_ticker(ticker):
      raise HTTPException(status_code=400, detail=f"Failed to add Ticker or Ticker already exists")
   
   success = _train_model_and_send_success(ticker)
   if success:
       return {"message": f"Ticker {ticker} added and model trained successfully"}
   else:
        raise HTTPException(status_code=500, detail=f"Failed to train model for {ticker}")
        
@app.get("/tickers", response_model=List[str])
async def get_tickers():
    """Endpoint to get all available tickers"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # tables = cursor.fetchall()
    # conn.close()
    # tickers = [table[0] for table in tables if table[0] != "sqlite_sequence"] #remove the metadata table
    tab_name = 'tickers'
    query = f"SELECT ticker, currency FROM {tab_name};"
    logging.debug(f"Executing SQL query: {query}")  # Log the query
    # self._get_database_connection()
    
    cursor.execute(query)
    tickers = cursor.fetchall()
    logging.debug(f"Tickers fetched from database: {tickers}")
    tickers = [x[0] for x in tickers]
    conn.close()
    return tickers

@app.get("/historical_data/{ticker}", response_model=HistoricalDataResponse)
async def get_historical_data(ticker: str):
    """Endpoint to fetch historical data for a ticker for the last month"""
    logger.info(f"Historical Data requested for ticker: {ticker}")
    try:
        today = datetime.today()
        one_month_ago = today - timedelta(days=30)
        query = f"SELECT Date, Close FROM \"{ticker}\" WHERE Date >= ? AND Date <= ? ORDER BY Date ASC"
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query(query, conn, parse_dates=['Date'], params=(one_month_ago, today))
        conn.close()

        if df.empty:
            logger.warning(f"No data found in database for last month for {ticker}")
            return HistoricalDataResponse(dates=[], close=[])
        
        dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
        close = df['Close'].tolist()
        return HistoricalDataResponse(dates=dates, close=close)

    except Exception as e:
         logger.error(f"Error fetching historical data from db for {ticker}: {e}")
         raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/update_data", status_code=200)
async def update_all_tickers_data():
    """Endpoint to fetch and update the historical data for all tickers"""
    logger.info("Initiating data update for all tickers")
    try:
        pipeline.fetch_data_for_all_tickers("1mo")
        
        return {"message": "Data update process completed."}

    except Exception as e:
        logger.error(f"Error during data update: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during data update: {e}")
    

@app.post("/retrain", status_code=200)
async def retrain_all_models():
    """Endpoint to retrain all existing models"""
    logger.info("Retraining all models initiated.")
    
    try:
        tickers = await get_tickers()
        
        for ticker in tickers:
            logger.info(f"Retraining model for {ticker}")
            success = _train_model_and_send_success(ticker)
            if not success:
               logger.warning(f"Failed to retrain model for {ticker}")
        
        return {"message": "All models retrained successfully."}

    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during retraining")
    
@app.post("/clean_db", status_code=200)
async def clean_database():
    """Endpoint to clean the entire database"""
    logger.info("Database cleaning initiated.")
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Delete all tables (except sqlite_sequence if needed)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
           if table[0] != "sqlite_sequence":
                cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
                logger.info(f"Dropped Table {table[0]}")
        
        #Recreate the ticker table
        pipeline.create_ticker_table()
        
        conn.commit()
        conn.close()
        
        #Clean the models folder as well
        _clean_models_folder()
        
        return {"message": "Database cleaned successfully."}
    except Exception as e:
        logger.error(f"Error during db cleaning: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during db cleaning: {e}")





def _clean_models_folder():
    """Helper method to clean the model folder"""
    models_folder = "./models"
    try:
        if os.path.exists(models_folder):
            for ticker_dir in os.listdir(models_folder):
                ticker_path = os.path.join(models_folder, ticker_dir)
                if os.path.isdir(ticker_path):
                    for file in os.listdir(ticker_path):
                        file_path = os.path.join(ticker_path,file)
                        os.remove(file_path)
                    os.rmdir(ticker_path)
        logger.info(f"Cleaned the folder {models_folder} successfully")
    except Exception as e:
        logger.error(f"Error cleaning the models folder: {e}")


def _get_latest_model_path(ticker):
    """Helper method to find the latest model for a given ticker"""
    ticker_folder = f"./models/{ticker}"
    if not os.path.exists(ticker_folder):
      return None
    model_files = [f for f in os.listdir(ticker_folder) if f.endswith('.pkl')]
    if not model_files:
       return None
    model_files.sort(reverse=True)
    return os.path.join(ticker_folder, model_files[0])

def _get_last_datapoints(ticker):
    """Helper method to fetch the last data points for the ticker"""
    try:
        query = f"SELECT Date, Close FROM \"{ticker}\" ORDER BY Date DESC LIMIT 7"
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query(query, conn, parse_dates=['Date'], index_col='Date')
        conn.close()
        if df.empty:
           logger.warning(f"No data found for ticker {ticker} to make predictions.")
           return None
        return df.sort_index()
    except Exception as e:
        logger.error(f"Error fetching last data point from db for {ticker}: {e}")
        return None
def _make_autoregressive_predictions(model, last_data):
    """Helper method to make autoregressive predictions"""
    
    predictions = []
    
    if last_data.empty:
        logger.warning("No data found in input dataframe to make autoregressive predictions")
        return predictions
    
    last_data_copy = last_data.copy()
    all_data = last_data_copy.copy()

    for i in range(7):
        next_date = all_data.index[-1] + timedelta(days=1)
        # Explicitly set each column to None
        all_data.loc[next_date, 'Close'] = None
        all_data.loc[next_date, 'day_of_week'] = None
        all_data.loc[next_date, 'month'] = None
        all_data.loc[next_date, 'day_of_year'] = None
        for lag in range(1, 8):
            all_data.loc[next_date, f'lag_{lag}'] = None
    
    #create df for all predictions
    # all_data = last_data_copy.copy()
    # for i in range(7):
    #     next_date = all_data.index[-1] + timedelta(days=1)
    #     all_data.loc[next_date] = [None]
    
    #create time features
    all_data['day_of_week'] = all_data.index.dayofweek
    all_data['month'] = all_data.index.month
    all_data['day_of_year'] = all_data.index.dayofyear
    
    # Create lag features
    for lag in range(1, 8):
        all_data[f'lag_{lag}'] = all_data['Close'].shift(lag)
    
    # all_data.dropna(inplace=True)
    
    for i in range(7):
         # Select the data needed for this iteration.
        #The last row of the all_data represents the prediction that can be made
        df_for_prediction = all_data.iloc[-(7 - i):].head(1).copy()
        
        if df_for_prediction.empty:
            logger.warning("No data remains to make prediction")
            break #if no data then exit loop
            
        X = df_for_prediction.drop('Close', axis=1)
        
        #Make prediction
        prediction_np = model.predict(X)[0]
        prediction = float(prediction_np) #explicitly cast to float here
        predictions.append(prediction)
        for j in range(1,8):
            if 7-(i+j) > 0:
                 all_data.loc[all_data.index[-(7-(i+j))], f'lag_{j}']= prediction
        # print(all_data)

        #Update the dataframe in place
        all_data.loc[df_for_prediction.index, "Close"] = prediction
    
    return predictions

def _train_model_and_send_success(ticker):
    """Helper method to train model and return success or failure."""
    try:
        trainer.train(ticker)
        logger.info(f"Training model complete for: {ticker}")
        return True
    except Exception as e:
        logger.error(f"Error during model training for {ticker}: {e}")
        return False


if __name__ == "__main__":
    import uvicorn
    # pipeline.run_pipeline() #create ticker table
    uvicorn.run(app, host="0.0.0.0", port=8080)