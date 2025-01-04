import os
import pandas as pd
import yfinance as yf
from datetime import datetime

BASE_DATA_FOLDER = (
    r"C:\Users\LucaBenedetti\Documents\personal_dev\long_short_strategy\data"
)


class DataPipeline:
    def __init__(self, base_folder=BASE_DATA_FOLDER):
        """
        Initializes the data pipeline.

        Args:
            base_folder (str): Folder to store data.
            csv_name (str): Name of the CSV file to store the data.
        """
        # TODO: define attribute data and keep them there
        self.data_dir = base_folder
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_data(self, tickers, start_date, end_date):
        """
        Fetches historical stock data using yfinance.

        Args:
            tickers (list): List of stock tickers (e.g., ["AAPL", "MSFT"]).
            start_date (str or datetime): Start date for data retrieval (YYYY-MM-DD).
            end_date (str or datetime): End date for data retrieval (YYYY-MM-DD).

        Returns:
            pandas.DataFrame: A DataFrame containing the fetched data in long format, or None if an error occurs.
        """

        try:
            data = yf.download(tickers, start=start_date, end=end_date)

            if data.empty:
                print(
                    f"No data retrieved for tickers: {tickers} between {start_date} and {end_date}. Check tickers or date range."
                )
                return None

            # Convert to long format
            data_long = data.stack(level=1).reset_index()
            data_long = data_long.rename(columns={"level_1": "Ticker"})
            data_long = data_long.rename(columns={"Date": "Date"})
            return data_long

        except Exception as e:
            print(f"An error occurred during data fetching: {e}")
            return None

    def process_data(self, data_frames):
        """Clean and process the raw data"""
        raise NotImplemented


    def store_data(self, data, filename):
        """
        Saves the data to a CSV file in the specified directory.

        Args:
            data (pandas.DataFrame): The DataFrame to save.
            filename (str): The name of the CSV file (without extension).
        """
        if data is None:
            print("No data to save.")
            return

        filepath = os.path.join(self.data_dir, f"{filename}.csv")
        try:
            data.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
        except Exception as e:
            print(f"An error occurred during data saving: {e}")

    def load_data(self, filename):
        """Loads data from a CSV file.

        Args:
            filename (str): The name of the CSV file (without extension).

        Returns:
            pandas.DataFrame: The loaded DataFrame, or None if an error occurs.
        """
        filepath = os.path.join(self.data_dir, f"{filename}.csv")
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded from {filepath}")
            return data
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None
        except pd.errors.EmptyDataError:
            print(f"File is empty: {filepath}")
            return None
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return None

    def update_data(self, symbols, lookback_days=30):
        """Update data files with recent data"""
        raise NotImplemented
