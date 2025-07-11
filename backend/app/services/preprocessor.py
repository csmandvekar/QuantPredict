import pandas as pd
import numpy as np
from app.services.claude_model import CNNLSTMStockPredictor

predictor = CNNLSTMStockPredictor(sequence_length=60, future_days=30)

def prepare_sequences_for_prediction(df: pd.DataFrame):
    scaled_data, df_clean = predictor.prepare_data(df)
    X, y = predictor.create_sequences(scaled_data)
    last_sequence = X[-1]  # use last available sequence
    return last_sequence, df_clean
