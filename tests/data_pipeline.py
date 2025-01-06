from scripts.data_loader import DataPipeline
from scripts.symbols_constants import (
    sp500_symbols,
    nasdaq_symbols,
    ftse_symbols,
    dax_symbols,
    nikkei_symbols,
    hangseng_symbols,
)

if __name__ == "__main__":

    pipeline = DataPipeline(
        base_folder=r"C:\Users\LucaBenedetti\Documents\personal_dev\long_short_strategy\tests\data"
    )

    # Define symbols and dates
    symbols = sp500_symbols
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # Fetch and store data
    data = pipeline.fetch_data(symbols, start_date, end_date)
    pipeline.store_data(data, filename="stocksss")

    data_2 = pipeline.load_data("stocksss")

    data_wide = pipeline.flat_to_wide(data_2)
    pipeline.store_data(data_wide, filename="wide_stocks")
