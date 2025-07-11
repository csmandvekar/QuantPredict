import yfinance as yf
import os
import pandas as pd
# from datetime import datetime
import logging


# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

# Config
symbol = "^NSEI"
start_date = "2021-01-08"
end_date = "2025-05-20"
stock_name = symbol.split(".")[0]
output_dir = os.path.join(os.getcwd(), "Dataset")

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

try:
    logging.info(f"🔍 Fetching data for {symbol} from {start_date} to {end_date}")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)

    # Edge Case 1: Empty data
    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'. It may be incorrect or the date range may be invalid.")

    # Edge Case 2: MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Save file only if data is valid
    csv_path = os.path.join(output_dir, f"{stock_name}.csv")
    df.to_csv(csv_path)
    logging.info(f"✅ Data saved to: {csv_path}")

except Exception as e:
    logging.error(f"❌ Failed to fetch or save data: {e}")
