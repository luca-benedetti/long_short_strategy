from scripts.data_loader import DataPipeline

import os

if __name__ == "__main__":

    pipeline = DataPipeline(
        base_folder=r"C:\Users\LucaBenedetti\Documents\personal_dev\long_short_strategy\tests\data"
    )

    # Define symbols and dates
    symbols = ["AAPL", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # Fetch and store data
    data = pipeline.fetch_data(symbols, start_date, end_date)
    pipeline.store_data(data, filename="stocksss")

    data_2 = pipeline.load_data("stocksss")
