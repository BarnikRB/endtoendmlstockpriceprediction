from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import logging
import os
import pickle
from datetime import datetime, timedelta, date
import sys
import traceback
from typing import List
from supabase import Client, create_client
from etl_pipeline.etls_class_postgres import StockDataPipeline
from model_pipeline.model_trainer_cloud import StockModelTrainer
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
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
supabase_config = {
        "url": SUPABASE_URL,  
        "key": SUPABASE_KEY # e.g., "your_anon_key"
    }
db_client : Client  = create_client(supabase_url=supabase_config["url"],supabase_key=supabase_config['key'])
pipeline = StockDataPipeline(supabase_client=db_client)

trainer = StockModelTrainer()


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
        model_path = trainer._get_latest_model_path(ticker)
        if not model_path:
           raise HTTPException(status_code=404, detail="Model not found for this ticker")
        
        
        model = trainer._download_model_from_gcs(model_filename=model_path)
        
        last_data = _get_last_datapoints(ticker)
        if last_data is None:
            raise HTTPException(status_code=404, detail="No data found in database to make prediction")
        predictions = _make_autoregressive_predictions(model, last_data)
        return {"ticker": ticker, "predictions": predictions}

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Error during prediction for {ticker}: {e}\n{tb_str}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/add", status_code=200)
async def add_ticker(request:AddTickerRequest):
   ticker = request.ticker
   logger.info(f"Add Ticker request for: {ticker}")
   if not pipeline.add_new_ticker(ticker):
      raise HTTPException(status_code=400, detail=f"Failed to add Ticker or Ticker already exists")
   df = pipeline.return_ticker_data(ticker=ticker)
   success = _train_model_and_send_success(ticker,df)
   if success:
       return {"message": f"Ticker {ticker} added and model trained successfully"}
   else:
        raise HTTPException(status_code=500, detail=f"Failed to train model for {ticker}")
        
@app.get("/tickers", response_model=List[str])
async def get_tickers():
    """Endpoint to get all available tickers"""
   

    tickers = pipeline._get_tickers_from_db()
    logging.debug(f"Tickers fetched from database: {tickers}")
    
    return tickers

@app.get("/historical_data/{ticker}", response_model=HistoricalDataResponse)
async def get_historical_data(ticker: str):
    """Endpoint to fetch historical data for a ticker for the last month"""
    logger.info(f"Historical Data requested for ticker: {ticker}")
    try:
        today = datetime.today()
        one_month_ago = today - timedelta(days=30)
    
        df = pipeline.return_ticker_data(ticker=ticker)
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= one_month_ago) & (df['date'] <= today)]

        if df.empty:
            logger.warning(f"No data found in database for last month for {ticker}")
            return HistoricalDataResponse(dates=[], close=[])
        
        dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
        close = df['close'].tolist()
        return HistoricalDataResponse(dates=dates, close=close)

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Error during prediction for {ticker}: {e}\n{tb_str}")
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
            data = pipeline.return_ticker_data(ticker)
            
            success = _train_model_and_send_success(ticker,data)
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
        # conn = sqlite3.connect(DATABASE_PATH)
        # cursor = conn.cursor()
        
        # # Delete all tables (except sqlite_sequence if needed)
        # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # tables = cursor.fetchall()
        
        # for table in tables:
        #    if table[0] != "sqlite_sequence":
        #         cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
        #         logger.info(f"Dropped Table {table[0]}")
        
        # #Recreate the ticker table
        # pipeline.create_ticker_table()
        
        # conn.commit()
        # conn.close()
        
        # #Clean the models folder as well
        pipeline.clean_database()
        _clean_models_folder()
        
        return {"message": "Database cleaned successfully."}
    except Exception as e:
        logger.error(f"Error during db cleaning: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during db cleaning: {e}")





def _clean_models_folder():
    """Helper method to clean the model folder"""
    
    try:
        trainer._clean_everything()
    except Exception as e:
        logger.error(f"Error cleaning the models folder: {e}")



def _get_last_datapoints(ticker):
    """Helper method to fetch the last data points for the ticker"""
    try:
        df = pipeline.return_ticker_data(ticker=ticker)
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
    last_data_copy['date'] = pd.to_datetime(last_data_copy['date'])
    last_data_copy = last_data_copy.set_index('date') 
   
    all_data = last_data_copy.copy()

    for i in range(7):
        # print(all_data.index[-1])
        next_date = all_data.index[-1] + timedelta(days=1)
        # Explicitly set each column to None
        all_data.loc[next_date, 'close'] = None
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
        all_data[f'lag_{lag}'] = all_data['close'].shift(lag)
    
    # all_data.dropna(inplace=True)
    
    for i in range(7):
         # Select the data needed for this iteration.
        #The last row of the all_data represents the prediction that can be made
        
        df_for_prediction = all_data.iloc[-(7 - i):].head(1).copy()
        df_cop = df_for_prediction.copy()
        if df_for_prediction.empty:
            logger.warning("No data remains to make prediction")
            break #if no data then exit loop
            
        df_for_prediction = df_for_prediction.reset_index()
        X = df_for_prediction.drop(['close','date','ticker'], axis=1)
        # print(X.info())
        
        #Make prediction
        prediction_np = model.predict(X)[0]
        prediction = float(prediction_np) #explicitly cast to float here
        predictions.append(prediction)
        for j in range(1,8):
            if 7-(i+j) > 0:
                 all_data.loc[all_data.index[-(7-(i+j))], f'lag_{j}']= prediction
        # print(all_data)

        #Update the dataframe in place
        all_data.loc[df_cop.index, "close"] = prediction
    
    return predictions

def _train_model_and_send_success(ticker,df):
    """Helper method to train model and return success or failure."""
    try:
        trainer.train(ticker=ticker,df=df)
        logger.info(f"Training model complete for: {ticker}")
        return True
    except Exception as e:
        logger.error(f"Error during model training for {ticker}: {e}")
        return False


if __name__ == "__main__":
    import uvicorn
    # pipeline.run_pipeline() #create ticker table
    uvicorn.run(app, host="0.0.0.0", port=8080)