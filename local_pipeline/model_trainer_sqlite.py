import pandas as pd
import sqlite3
import logging
import os
import pickle
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class StockModelTrainer:

    def __init__(self, database_path, log_level=logging.INFO):
        """
        Initializes the StockModelTrainer.

        Args:
            database_path (str): Path to the SQLite database file.
            log_level: logging level, default is INFO
        """
        self.database_path = database_path
        self.database_connection = None
        self.logger = self._setup_logger(log_level)
        self._create_models_folder() # Creates models folder if it doesn't exist

    def _create_models_folder(self):
        """Creates the models folder if it doesn't exist."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
            self.logger.info("Created 'models' folder.")

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

    def _connect_to_database(self):
        """Establishes a connection to the SQLite database."""
        try:
            self.database_connection = sqlite3.connect(self.database_path)
            self.logger.info("Connected to database successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise

    def _close_database_connection(self):
        """Closes the database connection."""
        if self.database_connection:
            try:
                self.database_connection.close()
                self.logger.info("Database connection closed.")
            except sqlite3.Error as e:
                self.logger.error(f"Error closing database connection: {e}")

    def _load_data_from_db(self, ticker):
        """Loads stock data for a ticker from the database."""
        if not self.database_connection:
            self._connect_to_database()
        try:
            query = f"SELECT Date, Close FROM \"{ticker}\" ORDER BY Date"
            df = pd.read_sql_query(query, self.database_connection, parse_dates=['Date'], index_col='Date')
            if df.empty:
              self.logger.warning(f"No data available for ticker: {ticker}")
              return None
            self.logger.info(f"Data loaded from database for ticker: {ticker}")
            return df
        except sqlite3.Error as e:
            self.logger.error(f"Error loading data from database for ticker {ticker}: {e}")
            return None

    def _preprocess_data(self, df):
        """Preprocesses data for modeling, fills missing dates, adds time features and lag features."""
         # 1. Fill missing dates
        if df is None or df.empty:
            self.logger.warning("No data provided to preprocess.")
            return None
        
        min_date = df.index.min()
        max_date = df.index.max()

        all_dates = pd.date_range(min_date, max_date, freq='D')
        df = df.reindex(all_dates)
        df.fillna(method='ffill', inplace=True) #Forward Fill
        df.fillna(method='bfill', inplace=True) #Backward Fill
        self.logger.debug("Missing dates filled.")
        # 2. Create time-related features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        self.logger.debug("Time related features extracted")

        # 3. Create lag features
        for lag in range(1, 8):
            df[f'lag_{lag}'] = df['Close'].shift(lag)
        df.dropna(inplace=True)  # Drop rows with NaN values from lag features
        self.logger.debug("Lag features created.")
        return df

    def train(self, ticker):
        """
        Trains an XGBoost model for the given ticker.
        """
        ticker = ticker.lower()
        
        self._connect_to_database()
        try:
            self.logger.info(f"Starting training for ticker: {ticker}")
            df = self._load_data_from_db(ticker)
            if df is None:
                return None
            df = self._preprocess_data(df)
            if df is None or df.empty:
                return None
            
            # Prepare the data for the model
            X = df.drop('Close', axis=1)
            y = df['Close']

            # Split data for training and testing
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            # self.logger.debug("Data split into train and test set")

            # Initialize and train XGBoost model
            model = xgb.XGBRegressor(objective='reg:squarederror',
                                    n_estimators=500,
                                    learning_rate=0.1,
                                    max_depth=5,
                                    random_state=42)
            model.fit(X, y)
            self.logger.info("XGBoost model trained.")

            # Evaluate the model
            # y_pred = model.predict(X_test)
            # rmse = mean_squared_error(y_test, y_pred, squared=False)
            # self.logger.info(f"Model evaluation completed. RMSE: {rmse}")

            # Save model to ticker-specific folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ticker_folder = f"./models/{ticker}"
            if not os.path.exists(ticker_folder):
                os.makedirs(ticker_folder)
                self.logger.info(f"Created folder for '{ticker}'.")
            
            model_filename = f"{ticker_folder}/{ticker}_model_{timestamp}.pkl"

            # Remove old models
            self._delete_old_models(ticker_folder)
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Model saved to: {model_filename}")
            self.logger.info(f"Training completed for ticker: {ticker}")
            return model
        except Exception as e:
            self.logger.error(f"An error occurred during training: {e}")
            return None
        finally:
            self._close_database_connection()
    
    def _delete_old_models(self, ticker_folder):
        """Deletes old models in the given folder."""
        for filename in os.listdir(ticker_folder):
            if filename.endswith(".pkl"):
                file_path = os.path.join(ticker_folder, filename)
                try:
                    os.remove(file_path)
                    self.logger.debug(f"Deleted old model: {filename}")
                except Exception as e:
                    self.logger.error(f"Error deleting old model '{filename}': {e}")


if __name__ == '__main__':
    # Example Usage
    database_path = "../data/stock_data.db"  # Replace with your database path
    # Create an instance of the StockModelTrainer class.
    trainer = StockModelTrainer(database_path, log_level=logging.DEBUG)
    # Example usage
    model = trainer.train("AAPL") # Replace with the ticker you want to train with
    if model:
      print("Model trained successfully.")
    else:
       print("Model training failed")
    model = trainer.train("AAPL") # Replace with the ticker you want to train with
    if model:
      print("Model trained successfully.")
    else:
       print("Model training failed")
    model = trainer.train("GOOG") # Replace with the ticker you want to train with
    if model:
      print("Model trained successfully.")
    else:
       print("Model training failed")