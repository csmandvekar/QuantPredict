import sys
import yfinance
import os
import getpass
import datetime
from app.services.scraper import fetch_stock_data

# To run this script, use:
#   python -m app.diagnose_yfinance
# from the backend directory (not backend/app)

def main():
    print("=== ENVIRONMENT DIAGNOSTICS ===")
    print("PYTHON:", sys.executable)
    print("YFINANCE VERSION:", yfinance.__version__)
    print("CWD:", os.getcwd())
    print("USER:", getpass.getuser())
    print("PATH:", os.environ.get('PATH'))
    print("\n=== NETWORK TEST ===")

    symbol = "INFY"
    start_date = "2021-01-01"
    print(f"Testing fetch_stock_data for {symbol} from {start_date} to 2 days prior to today")

    df = fetch_stock_data(symbol, start_date)
    print("DataFrame shape:", df.shape)
    print(df.head())
    if df.empty:
        print("\n[ERROR] fetch_stock_data returned an empty DataFrame!")
    else:
        print("\n[SUCCESS] fetch_stock_data returned data!")

if __name__ == "__main__":
    main() 