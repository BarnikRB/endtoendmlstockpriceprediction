import pandas as pd
import sqlite3
import logging
import os
import pickle
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.cloud import storage

class StockModelTrainer:

    def __init__(self, log_level=logging.INFO, bucket_name="your-bucket-name"):
        """
        Initializes the StockModelTrainer.

        Args:
            log_level: logging level, default is INFO
            bucket_name: Google Cloud Storage bucket name
        """
        self.bucket_name = bucket_name
        self.logger = self._setup_logger(log_level)
        
        self.storage_client = storage.Client()

    def _setup_logger(self, log_level):
        """Sets up the logger."""
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _preprocess_data(self, df):
        """Preprocesses data for modeling, fills missing dates, adds time features and lag features."""
        if df is None or df.empty:
            self.logger.warning("No data provided to preprocess.")
            return None

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')  # Ensure chronological order

        # Fill missing dates
        min_date = df['date'].min()
        max_date = df['date'].max()
        all_dates = pd.DataFrame({'date': pd.date_range(min_date, max_date, freq='D')})

        df = all_dates.merge(df, on='date', how='left')
        df.fillna(method='ffill', inplace=True)  # Forward Fill
        df.fillna(method='bfill', inplace=True)  # Backward Fill
        self.logger.debug("Missing dates filled.")

        # Time-related features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        self.logger.debug("Time-related features extracted.")

        # Lag features
        for lag in range(1, 8):
            df[f'lag_{lag}'] = df['close'].shift(lag)
        
        df.dropna(inplace=True)  # Drop rows with NaN values
        self.logger.debug("Lag features created.")

        return df

    def train(self, ticker, df):
        """
        Trains an XGBoost model for the given ticker and stores it on Google Cloud Storage.
        """
        ticker = ticker.lower()
        try:
            self.logger.info(f"Starting training for ticker: {ticker}")
            
            if df is None:
                return None

            df = self._preprocess_data(df)
            if df is None or df.empty:
                return None

            # Prepare the data for the model
            X = df.drop(['close', 'date', 'ticker'], axis=1)
            y = df['close']

            # Train the model
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500,
                                     learning_rate=0.1, max_depth=5, random_state=42)
            model.fit(X, y)
            self.logger.info("XGBoost model trained.")

            # Save model to Google Cloud Storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{ticker}/{ticker}_model_{timestamp}.pkl"

            # Upload to GCS
            self._upload_model_to_gcs(model, model_filename)
            self.logger.info(f"Model uploaded to GCS bucket '{self.bucket_name}' at path: {model_filename}")
            
            self.logger.info(f"Training completed for ticker: {ticker}")
            return model
        except Exception as e:
            self.logger.error(f"An error occurred during training: {e}")
            return None

    def _upload_model_to_gcs(self, model, model_filename):
        """Uploads the model to Google Cloud Storage."""
        # Serialize model using pickle
        model_bytes = pickle.dumps(model)

        # Upload to GCS
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(model_filename)
        blob.upload_from_string(model_bytes)
        self.logger.info(f"Model uploaded to GCS: gs://{self.bucket_name}/{model_filename}")

    def _download_model_from_gcs(self, model_filename):
        """Downloads a model from Google Cloud Storage."""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(model_filename)
        model_bytes = blob.download_as_string()
        model = pickle.loads(model_bytes)
        return model

