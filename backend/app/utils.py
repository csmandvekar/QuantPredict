import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from .config import settings

# Initialize Alpha Vantage client
_ts_client = TimeSeries(key=settings.ALPHA_VANTAGE_KEY, output_format="pandas")


def fetch_intraday(symbol: str, interval: str = "60min", outputsize: str = "compact") -> pd.DataFrame:
    """
    Fetch intraday data for a symbol from Alpha Vantage.
    """
    data, _ = _ts_client.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
    data = data.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })
    return data.sort_index()


def preprocess(df: pd.DataFrame, past: int = 30):
    """
    Scale and window the data into numpy array windows.
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    df = df.dropna()
    features = df[["open", "high", "low", "close"]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    windows = []
    for i in range(past, len(scaled)):
        windows.append(scaled[i-past:i])
    return np.array(windows), scaler