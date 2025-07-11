import yfinance as yf
import pandas as pd
import os
import logging
import datetime
from app.core.config import DATA_DIR

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

def fetch_stock_data(script: str, start_date="2021-01-01") -> pd.DataFrame:
    # Sanitize and ensure .NS suffix for Indian stocks
    symbol = script.strip().upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    print(f"DEBUG: Using symbol: {symbol!r}")

    output_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    try:
        logging.info(f"🔍 Fetching data for {symbol} from {start_date} to latest available")
        # Remove end_date: fetch up to the latest available trading day
        df = yf.download(symbol, start=start_date, progress=False)

        if df.empty:
            logging.warning(f"→ No data returned with explicit range; trying period='max'")
            df = yf.download(symbol, period="max", progress=False)

        if df.empty:
            logging.warning(f"→ Still empty, trying Ticker.history()")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="max")

        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}' in any query. Check ticker validity.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        df.to_csv(output_path, index=False)
        logging.info(f"✅ Data saved to: {output_path}")
        return df

    except Exception as e:
        logging.error(f"❌ Failed to fetch or save data for {symbol}: {e}")
        raise ValueError(f"Failed to fetch or save data for {symbol}: {e}")
