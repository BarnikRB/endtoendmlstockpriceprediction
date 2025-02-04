import yfinance as yf
import pandas as pd
import sqlite3
import logging
import os
import sys
from datetime import datetime

class StockDataPipeline:

    def __init__(self, db_type, db_config, tickers_table_name="tickers"):
        self.db_type = db_type
        self.db_config = db_config
        self.tickers_table_name = tickers_table_name
        self.database_connection = None
        self.log_level = logging.INFO if os.getenv("DEBUG") else logging.ERROR

        # Get the root logger
        logger = logging.getLogger()
        logger.setLevel(self.log_level)

        # Check if the handler is already added to the logger
        if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
             # Configure the logger to output to the notebook if there isn't a stream handler already
            handler = logging.StreamHandler(sys.stdout) # Use sys.stdout to output to notebook
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def _get_database_connection(self):
        """
         Internal method to create a connection to the SQLite database.
        """
        try:
            if self.db_type == 'local':
                db_path = self.db_config['database']
                connection = sqlite3.connect(db_path)
                logging.info("Database connection established successfully.")
                self.database_connection = connection
                return connection
            else:
                raise ValueError("Invalid db_type, must be 'local'")
        except Exception as e:
            logging.error(f"Error creating database connection: {e}")
            raise

    def _get_tickers_from_db(self):
        """
        Fetches the list of stock tickers from the database table.
        """
        try:
            query = f"SELECT ticker, currency FROM {self.tickers_table_name};"
            logging.debug(f"Executing SQL query: {query}")  # Log the query
            # self._get_database_connection()
            cursor = self.database_connection.cursor()
            cursor.execute(query)
            tickers = cursor.fetchall()
            logging.debug(f"Tickers fetched from database: {tickers}")
            # self.close_connection()
            return tickers
        except sqlite3.Error as e:
            logging.error(f"Error fetching tickers from database: {e}")
            print(f"sqlite3 Error details: {e}")
            return []

    def fetch_stock_data(self, period, ticker):
        """
         Fetches stock data from Yahoo Finance for a single ticker and time period.
        """
        if not ticker:
            logging.warning("No ticker provided.")
            return None, None
        try:
            ticker_obj = yf.Ticker(ticker.lower())
            data = ticker_obj.history(period=period)
            if not data.empty:
               data = data.reset_index()
               logging.debug(f"Successfully fetched stock data for {ticker}, period: {period}")
               logging.debug(f"columns in fetched data: {data.columns.tolist()}") # Add for debug
               return data, ticker
            else:
                logging.warning(f"No data found for ticker: {ticker}")
                return None, None
        except Exception as e:
            logging.error(f"Error fetching stock data: {e}")
            return None, None

    def preprocess_data(self, data, ticker=None):
        """
            Takes the data from `fetch_stock_data` and cleans, transforms, and prepares the fetched data for database insertion
        """
        if data is None or data.empty:
            logging.warning("No data to preprocess.")
            return None

        try:
           #Check if data is multi-indexed dataframe
            if isinstance(data.index, pd.MultiIndex):
               logging.debug("Dataframe has multi-index")
               data = data.reset_index()
               logging.debug(f"Data after reset index {data.head()}")
            else:
               logging.debug("Dataframe has single index")
               if ticker is not None:
                data['Ticker'] = ticker # Add a Ticker column if its single ticker
               else:
                  logging.warning("Ticker Name not found")
                  return None
            logging.debug(f"Columns before column check: {data.columns.tolist()}")
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                    if col not in data.columns:
                        logging.warning(f"Column {col} is missing from the data")
                        return None # Return none if any of the column is not available.

            logging.debug(f"Columns after column check: {data.columns.tolist()}")
            data.dropna(subset=required_columns, inplace=True)
            logging.debug(f"Data after dropna {data.head()}")
            data = data[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            logging.debug(f"Data after selection {data.head()}")
            logging.debug("Data preprocessed successfully.")
            return data

        except Exception as e:
             logging.error(f"Error preprocessing data: {e}")
             return None

    def _create_table(self, table_name, processed_data=None):
        """
        Internal method that creates the database table if it does not exist based on the structure of the processed_data
        If processed_data is None, it assumes the table is for the tickers and creates the respective table structure
        """
        try:
           cursor = self.database_connection.cursor()

           if processed_data is not None: # If table is not the ticker table
                columns = [
                    "Date DATETIME PRIMARY KEY",
                    "Open REAL",
                    "High REAL",
                    "Low REAL",
                    "Close REAL",
                    "Volume REAL"
                ]
                create_table_query = f"CREATE TABLE IF NOT EXISTS \"{table_name}\" ({', '.join(columns)});"
                logging.debug(f"Creating table with following structure: {create_table_query}")
                cursor.execute(create_table_query)
                self.database_connection.commit()
                logging.info(f"Table '{table_name}' created successfully.")
           else: # For ticker table
                columns = [
                    "ticker TEXT PRIMARY KEY",
                    "currency TEXT"
                ]
                create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)});"
                logging.debug(f"Creating table with following structure: {create_table_query}")
                cursor.execute(create_table_query)
                self.database_connection.commit()
                logging.info(f"Table '{table_name}' created successfully.")
           return True
        except sqlite3.Error as e:
            logging.error(f"Error creating table '{table_name}': {e}")
            print(f"sqlite3 Error details: {e}")
            return False


    def insert_data_to_db(self, table_name, processed_data):
        """
            Inserts the processed stock data into the specified SQLite database table, checking the last date.
        """
        if processed_data is None or processed_data.empty:
            logging.warning(f"No processed data to insert into '{table_name}'.")
            return False
        try:
            cursor = self.database_connection.cursor()
            # 1. Get the last date
            cursor.execute(f"SELECT MAX(Date) FROM \"{table_name}\"")
            last_date_result = cursor.fetchone()
            last_date = last_date_result[0] if last_date_result and last_date_result[0] else None # assign none if table is empty


            if last_date:
            # 2. Filter New Data
                logging.debug(f"Last date in table {table_name}: {last_date}")
                processed_data['Date'] = pd.to_datetime(processed_data['Date'])
                last_date = datetime.fromisoformat(last_date)
                filtered_data = processed_data[processed_data['Date'] > last_date]
                if filtered_data.empty:
                    logging.info(f"No new data to insert into '{table_name}'.")
                    return True

                processed_data= filtered_data
            # 3. Insert New Data
            processed_data.rename(columns={'Date':'date'}, inplace=True)
            processed_data['date'] = processed_data['date'].apply(lambda x: x.isoformat())
            placeholders = ', '.join(['?' for _ in processed_data.columns])
            for row in processed_data.itertuples(index=False):
                sql = f"INSERT OR IGNORE INTO \"{table_name}\" VALUES ({placeholders})"
                cursor.execute(sql,tuple(row))
            self.database_connection.commit()
            logging.info(f"Data inserted successfully into '{table_name}'.")
            return True

        except sqlite3.Error as e:
            logging.error(f"Error inserting data into '{table_name}': {e}")
            return False

    def insert_ticker(self,ticker,currency):
        try:
          df_ticker = pd.DataFrame({"ticker": [ticker], "currency": [currency]})
          df_ticker.to_sql(self.tickers_table_name,self.database_connection, if_exists="append",index=False)
          return True
        except sqlite3.Error as e:
           logging.error(f"Error inserting data into : {e}")
           return False

    def add_new_ticker(self, ticker):
        """
        Adds a new ticker to the database, checking if it already exists.
        If not, it creates a new table for the ticker and fills with the last one year of data.
        Also, validate ticker and gets ticker currency
        """
        self._get_database_connection()
        ticker = ticker.lower()
        
        try:
            ticker_info = yf.Ticker(ticker)
            if ticker_info.info is None or 'currency' not in ticker_info.info: #validate ticker and check currency
                logging.error(f"Failed to get valid information for ticker '{ticker}'. Please verify the ticker symbol.")
                return False
            currency = ticker_info.info['currency']
            cursor = self.database_connection.cursor()
            # Check if ticker already exists
            query = f"SELECT COUNT(*) FROM {self.tickers_table_name} WHERE ticker = ?"
            cursor.execute(query, (ticker,))
            result = cursor.fetchone()[0]

            if result > 0:
                logging.info(f"Ticker '{ticker}' already exists in the database.")
                return False

            # Add new ticker and currency
            if not self.insert_ticker(ticker, currency):
                logging.error(f"Failed to insert into ticker table for ticker '{ticker}'.")
                return False
            logging.info(f"Added ticker {ticker} with currency {currency} in table {self.tickers_table_name}")

            # Fetch historical data (one year)
            data, ticker_name = self.fetch_stock_data(period="1y", ticker=ticker)
            if data is None:
                logging.error(f"Failed to fetch initial data for ticker '{ticker}'.")
                return False

            processed_data = self.preprocess_data(data, ticker_name)
            if processed_data is None:
                logging.error(f"Failed to preprocess data for ticker '{ticker}'.")
                return False
            # Create new table for ticker
            if not self._create_table(ticker, processed_data):
                logging.error(f"Failed to create the table for ticker '{ticker}'.")
                return False

            # Insert Data
            if not self.insert_data_to_db(ticker, processed_data.drop('Ticker', axis=1)):
                logging.error(f"Failed to insert data into table for ticker '{ticker}'.")
                return False
            self.close_connection()

            logging.info(f"Ticker '{ticker}' added successfully.")
            return True

        except sqlite3.Error as e:
              logging.error(f"Error adding new ticker '{ticker}': {e}")
              return False
        except Exception as e:
            logging.error(f"Error adding new ticker '{ticker}': {e}")
            return False
        
        
        
    def fetch_data_for_all_tickers(self, period):
        """
          Fetches data for all the tickers available in the database for the specified period.
        """
        self._get_database_connection()
        tickers = [t[0] for t in self._get_tickers_from_db()] # Use only tickers no currency
        if not tickers:
           logging.warning("No tickers found in the database to fetch data for")
           return None
        all_processed_data = []
        for ticker in tickers:
            data, _ = self.fetch_stock_data(period=period, ticker=ticker)
            if data is None:
                logging.error(f"Error fetching data for ticker: {ticker}, period: {period}")
                continue # skip to next ticker
            processed_data = self.preprocess_data(data, ticker)
            if processed_data is None:
                logging.error(f"Error preprocessing data for ticker: {ticker}, period: {period}")
                continue # skip to next ticker
            all_processed_data.append(processed_data)

        if not all_processed_data:
            return
        for processed_data in all_processed_data:
             ticker = processed_data['Ticker'].iloc[0]
             if not self.insert_data_to_db(ticker, processed_data.drop('Ticker', axis=1)): # Drop Ticker column as table is already for that ticker
                 logging.error(f"Error inserting data to table '{ticker}'")
        logging.info(f"Data for all tickers inserted for period : {period}")
        self.close_connection()

    def clean_database(self, table_name=None):
        """
          Removes all data from the database if no table name is provided.
          Alternatively, removes all the data from the specified table.
        """
        self._get_database_connection()
        try:
            cursor = self.database_connection.cursor()
            if table_name is None:
                # delete data from all tables, except for the tickers table
                tickers = [t[0] for t in self._get_tickers_from_db()]
                for ticker in tickers:
                    query = f"DELETE FROM \"{ticker}\""
                    cursor.execute(query)
                query = f"DELETE FROM \"{self.tickers_table_name}\""
                cursor.execute(query)
                
                logging.info(f"All stock data deleted from the database.")
            else:
                query = f"DELETE FROM \"{table_name}\""
                cursor.execute(query)
                logging.info(f"All data deleted from the table: {table_name}")
            self.database_connection.commit()
            self.close_connection()
        except sqlite3.Error as e:
            logging.error(f"Error cleaning the database: {e}")
        return

    def print_ticker_data(self, ticker):
       """
         Prints all existing data for a ticker from the database.
       """
       ticker = ticker.lower()
       self._get_database_connection()
       try:
           cursor = self.database_connection.cursor()
           query = f"SELECT * FROM \"{ticker}\""
           print(query)
           cursor.execute(query)
           rows = cursor.fetchall()
           if not rows:
               logging.warning(f"No data found for ticker '{ticker}' in the database.")
               return

           df = pd.DataFrame(rows, columns=[description[0] for description in cursor.description])
           print(f"\nData for ticker: {ticker}\n")
           print(df)
           self.close_connection()
       except sqlite3.Error as e:
              logging.error(f"Error fetching and printing data for ticker '{ticker}': {e}")

    def print_all_tickers(self):
        """
          Prints all tickers from the database.
        """
        self._get_database_connection()
        tickers = self._get_tickers_from_db()
        if tickers:
            print("\nList of tickers in the database:\n")
            for ticker, currency in tickers:
               print(f"Ticker: {ticker}, Currency: {currency}")
            self.close_connection()
        else:
              logging.warning("No tickers found in the database.")

    def close_connection(self):
        """
        Closes the database connection.
        """
        try:
            self.database_connection.close()
            logging.info("Database connection closed successfully.")
        except Exception as e:
            logging.error(f"Error closing database connection: {e}")

    def run_pipeline(self):
        """
        Orchestrates the initial setup process of the database.
        It creates a ticker table and adds first ticker
        """
        self._get_database_connection()
        if not self._create_table(self.tickers_table_name):
                logging.error("Failed to create tickers table")
                return
        self.close_connection()
        #   if not self.add_new_ticker("AAPL"):
        #       logging.error("Failed to add a first ticker")
        #       return
        #   logging.info("Initial setup done.")


if __name__ == "__main__":
    os.environ["DEBUG"] = "1" 
    local_db_config = {
        "database": "../data/stock_data.db",
    }
    
    try:

        # Example of using local database
        pipeline = StockDataPipeline("local", local_db_config)
        # pipeline.clean_database()
        # pipeline.run_pipeline() # this will create the ticker table and add AAPL as the first ticker

        # #add new ticker
        # pipeline.add_new_ticker("GOOG")
        # pipeline.add_new_ticker('AAPL')
        # pipeline.print_ticker_data('GOOG')
        # pipeline.fetch_data_for_all_tickers(period='1mo')
        # pipeline.print_ticker_data('GOOG')
        # #fetch data for all tickers
        # pipeline.fetch_data_for_all_tickers("1mo")

        # clean database
        #pipeline.clean_database()

        #clean specific table
        #pipeline.clean_database(table_name="AAPL")

        #Print ticker data
        # pipeline.print_ticker_data()

        # Print list of tickers
        pipeline.print_all_tickers()
        pipeline.clean_database()
        pipeline.print_all_tickers()
        
        pipeline.close_connection()


    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")