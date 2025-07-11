import os
import numpy as np
from app.core.config import MODEL_DIR
from app.services.claude_model import CNNLSTMStockPredictor

def load_model_for_script(script: str):
    import joblib
    model_path = os.path.join(MODEL_DIR, f"{script}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{script}_model_scaler.save")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file for {script} not found")
    predictor = CNNLSTMStockPredictor()
    # Hardcoded feature list (must match training)
    predictor.feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'ma_7', 'ma_21', 'ma_50', 'rsi', 'macd',
        'macd_signal', 'bb_upper', 'bb_lower',
        'price_change', 'volume_ma', 'hl_pct', 'co_pct'
    ]
    # Load the fitted scaler from training
    if os.path.exists(scaler_path):
        predictor.scaler = joblib.load(scaler_path)
    predictor.model = predictor.build_model((60, len(predictor.feature_columns)))
    predictor.model.load_weights(model_path)
    return predictor
