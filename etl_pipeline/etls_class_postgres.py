import yfinance as yf
import pandas as pd
import logging
import os
import sys
from datetime import datetime
from supabase import create_client, Client
import traceback

class StockDataPipeline:

    def __init__(self,supabase_client, tickers_table_name="tickers", stock_table_name = "stocks"):
        
        self.tickers_table_name = tickers_table_name
        self.stock_table_name = stock_table_name
        self.supabase_client: Client = supabase_client
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
    
    
    
    
    
    def _get_tickers_from_db(self):
        """
        Fetches the list of stock tickers from the database table.
        """
        try:
            # query = f"SELECT ticker, currency FROM {self.tickers_table_name};"
            # logging.debug(f"Executing SQL query: {query}")  # Log the query

            response = self.supabase_client.table(self.tickers_table_name).select("ticker, currency").execute()

            if  len(response.data) == 0:
                logging.warning(f"No tickers in database")
                return []
            
            tickers = [row['ticker'] for row in response.data]  # Format for consistency

            logging.debug(f"Tickers fetched from database: {tickers}")
            return tickers
        except Exception as e:
            logging.error(f"Error fetching tickers from database: {e}")
            return []
    
    
    
    
    def fetch_stock_data(self, period, ticker):
        """
         Fetches stock data from Yahoo Finance for a single ticker and time period.
        """
        if not ticker:
            logging.warning("No ticker provided.")
            return pd.DataFrame()
        try:
            data = yf.download(ticker.lower(),period=period)
            # data = ticker_obj.history(period=period)
            if  len(data)>0:
               data = data.stack().reset_index()
               logging.info(data.columns)
               data = data[['Date','Ticker','Close']]
               data = data.rename(columns=str.lower)
               data['ticker'] = data["ticker"].str.lower()
               data['date'] = data['date'].apply(lambda x: x.replace(tzinfo=None).isoformat())
               data = data.dropna()
               logging.debug(f"Successfully fetched stock data for {ticker}, period: {period}")
               logging.debug(f"columns in fetched data: {data.columns.tolist()}") # Add for debug
               return data
            else:
                logging.warning(f"No data found for ticker: {ticker}")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching stock data: {e}")
            return pd.DataFrame()
        




    def insert_stock_data_to_db(self,ticker_symbol, processed_data: pd.DataFrame):
        """
            Inserts the processed stock data into the specified Supabase database table, checking the last date.
        """
        table_name = self.stock_table_name
        logging.info(type(processed_data))
        if len(processed_data) == 0:
            logging.warning(f"No processed data to insert into '{table_name}'.")
            return False

        try:
            # 1. Get the last date (Supabase does not have MAX aggregate function for datetime)
            response = self.supabase_client.table(table_name).select("date", "ticker").eq("ticker", ticker_symbol).order("date", desc=True).limit(1).execute()

            

            last_date = response.data[0]['date'] if len(response.data) > 0  else None

            if last_date:
                # 2. Filter New Data
                logging.info(f"Last date in table {table_name}: {last_date}")
                processed_data['date'] = pd.to_datetime(processed_data['date'])
                last_date = pd.to_datetime(last_date).tz_localize(None)
                
                filtered_data = processed_data[pd.to_datetime(processed_data['date']) > last_date]
                
                if len(filtered_data) == 0:
                    logging.info(f"No new data to insert into '{table_name}'.")
                    return True
                processed_data = filtered_data
                processed_data['date'] = processed_data['date'].apply(lambda x: x.isoformat() )

            # 3. Insert New Data

            
            
            logging.info(type(processed_data))
            data_to_insert = processed_data.to_dict(orient='records')

            response = self.supabase_client.table(table_name).insert(data_to_insert).execute()  #Insert list of dict

            # if not response.data:
            #     logging.error(f"Error inserting data into '{table_name}'")
            #     return False

            logging.info(f"Data inserted successfully into '{table_name}'.")
            return True

        except Exception as e:
            error_message = traceback.format_exc()
            logging.error(f"Error inserting data into '{table_name}': {error_message}")
            return False

    def insert_ticker(self,ticker,currency,long_name):
        try:
          # Use the Supabase client to insert the ticker data
          data = {"ticker": ticker, "currency": currency,'name':long_name}
          response = self.supabase_client.table(self.tickers_table_name).insert(data).execute() # insert a single dict
          if  not len(response.data) > 0 :
              logging.error(f"Error inserting ticker {ticker} to Supabase")
              return False
          logging.info(f"Successfully inserted ticker {ticker} with currency {currency}")
          return True
        except Exception as e:
           logging.error(f"Error inserting data: {e}")
           return False
   
   
   
   
    def add_new_ticker(self, ticker):
        """
        Adds a new ticker to the database, checking if it already exists.
        If not, it creates a new table for the ticker and fills with the last one year of data.
        Also, validate ticker and gets ticker currency
        """
        
        ticker = ticker.lower()
        
        try:
            ticker_info = yf.Ticker(ticker)
            if ticker_info.info is None or 'currency' not in ticker_info.info: #validate ticker and check currency
                logging.error(f"Failed to get valid information for ticker '{ticker}'. Please verify the ticker symbol.")
                return False
            currency = ticker_info.info['currency']
            name = ticker_info.info['longName']

            # Check if ticker already exists

            response = self.supabase_client.table(self.tickers_table_name).select("*", count="exact").eq("ticker", ticker).execute()

            # if not response.data:
            #     logging.error(f"Error checking if ticker exists")
            #     return False
            
            count = len(response.data)  #Use len(data) instead of response count

            if count > 0:
                logging.info(f"Ticker '{ticker}' already exists in the database.")
                return False

            # Add new ticker and currency
            if not self.insert_ticker(ticker, currency,name):
                logging.error(f"Failed to insert into ticker table for ticker '{ticker}'.")
                return False
            logging.info(f"Added ticker {ticker} with currency {currency} in table {self.tickers_table_name}")

            # Fetch historical data (one year)
            data: pd.DataFrame  = self.fetch_stock_data(period="1y", ticker=ticker)
            if len(data) == 0:
                logging.error(f"Failed to fetch initial data for ticker '{ticker}'.")
                return False

            processed_data = data.copy()
            logging.info(type(processed_data))
            # if processed_data is None:
            #     logging.error(f"Failed to preprocess data for ticker '{ticker}'.")
            #     return False
           
            # Insert Data
            if not self.insert_stock_data_to_db(ticker, processed_data):
                logging.error(f"Failed to insert data into table for ticker '{ticker}'.")
                return False

            logging.info(f"Ticker '{ticker}' added successfully.")
            return True

        except Exception as e:
            logging.error(f"Error adding new ticker '{ticker}': {e}")
            return False
    




    def fetch_data_for_all_tickers(self, period):
        """
          Fetches data for all the tickers available in the database for the specified period.
        """
        
        tickers = [t for t in self._get_tickers_from_db()] # Use only tickers no currency
        if not tickers:
           logging.warning("No tickers found in the database to fetch data for")
           return None
        all_processed_data = []
        for ticker in tickers:
            data = self.fetch_stock_data(period=period, ticker=ticker)
            if data is None:
                logging.error(f"Error fetching data for ticker: {ticker}, period: {period}")
                continue # skip to next ticker
            processed_data = data
            if processed_data is None:
                logging.error(f"Error preprocessing data for ticker: {ticker}, period: {period}")
                continue # skip to next ticker
            if not self.insert_stock_data_to_db(ticker, processed_data):
                logging.error(f"Failed to insert data into table for ticker '{ticker}'.")
                continue
            
            all_processed_data.append(processed_data)

    def clean_database(self):
        
        try:
            response  = self.supabase_client.table(self.stock_table_name).delete().neq("date", datetime(1700, 1, 1)).execute()
            
            logging.info(f"{len(response.data)} rows wiped out from {self.stock_table_name}")


            response_ticker  = self.supabase_client.table(self.tickers_table_name).delete().neq('ticker', '').execute()
            
            logging.info(f"{len(response_ticker.data)} rows wiped out from {self.tickers_table_name}")
        except e:
            logging.error(f"Deletion failed due to {e}")



    def print_ticker_data(self, ticker):
       """
         Prints all existing data for a ticker from the database.
       """
       ticker = ticker.lower()
       
       try:
           # Fetch data from Supabase
           response = self.supabase_client.table(self.stock_table_name).select("date, ticker, close").eq("ticker", ticker).execute()


           
           

           data = response.data
           if  data == []:
               logging.warning(f"No data found for ticker '{ticker}' in the database.")
               return

           df = pd.DataFrame(data)
           print(f"\nData for ticker: {ticker}\n")
           print(df)
       except Exception as e:
           logging.error(f"Error fetching and printing data for ticker '{ticker}': {e}")


    def return_ticker_data(self, ticker):
       """
         Prints all existing data for a ticker from the database.
       """
       ticker = ticker.lower()
       
       try:
           # Fetch data from Supabase
           response = self.supabase_client.table(self.stock_table_name).select("date, ticker, close").eq("ticker", ticker).execute()


           
           

           data = response.data
           if  data == []:
               logging.warning(f"No data found for ticker '{ticker}' in the database.")
               return

           df = pd.DataFrame(data)
           return df
       except Exception as e:
           logging.error(f"Error fetching and printing data for ticker '{ticker}': {e}")
           return pd.DataFrame()


    def print_all_tickers(self):
        """
          Prints all tickers from the database.
        """
        
        tickers = self._get_tickers_from_db()
        if tickers:
            print("\nList of tickers in the database:\n")
            for ticker in tickers:
               print(f"Ticker: {ticker}")
        else:
              logging.warning("No tickers found in the database.")


