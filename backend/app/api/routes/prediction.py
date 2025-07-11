from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.config import ALLOWED_SCRIPTS
from app.services.scraper import fetch_stock_data
from app.services.preprocessor import prepare_sequences_for_prediction
from app.services.predictor import load_model_for_script
import pandas as pd
import traceback

router = APIRouter()

class PredictionRequest(BaseModel):
    script: str

@router.get("/scripts")
async def get_available_scripts():
    return {"scripts": ALLOWED_SCRIPTS}

@router.post("/predict")
async def predict_stock(request: PredictionRequest):
    # --- 1. Sanitize and validate script ---
    print(f"DEBUG: Received script from frontend: {request.script!r}")
    script = request.script.strip().upper()
    if script not in ALLOWED_SCRIPTS:
        print(f"DEBUG: {script} not in ALLOWED_SCRIPTS: {ALLOWED_SCRIPTS}")
        raise HTTPException(status_code=400, detail="Unsupported script")

    # --- 2. Prepare symbol and data_cache path ---
    import os
    import pandas as pd
    from app.core.config import DATA_DIR
    # Use the script name directly for data_cache (without .NS suffix)
    symbol = script
    print(f"DEBUG: Using symbol for data_cache: {symbol!r}")
    data_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    print(f"DEBUG: Looking for data at: {data_path}")

    # --- 3. Load the saved data for prediction ---
    if not os.path.exists(data_path):
        print(f"DEBUG: File does not exist: {data_path}")
        raise HTTPException(status_code=500, detail=f"No pre-saved data found for {symbol}. Please ensure the dataset exists at {data_path}.")
    try:
        # Skip the second row (ticker names)
        df = pd.read_csv(data_path, skiprows=[1])
        print(f"DEBUG: Loaded data shape: {df.shape}")
        # Ensure numeric columns
        for col in ['close', 'open', 'high', 'low', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"DEBUG: Exception while loading CSV: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load saved data for {symbol}: {e}")

    # --- 4. Preprocess + get last sequence ---
    try:
        last_sequence, df_clean = prepare_sequences_for_prediction(df)
    except Exception as e:
        print(f"DEBUG: Exception in prepare_sequences_for_prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in preprocessing: {e}")

    # 3. Load pretrained model
    model = load_model_for_script(script)

    # 4. Predict future prices
    future_scaled = model.predict_future(last_sequence, steps=30)
    future_prices = model.inverse_transform_predictions(future_scaled)

    # 5. Extract actual prices for past 100 days
    actual_df = df_clean[['date', 'close']].tail(100)
    actual_prices = actual_df['close'].tolist()
    actual_dates = actual_df['date'].astype(str).tolist()

    # 6. Generate future date range
    last_date = pd.to_datetime(actual_df['date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30).strftime('%Y-%m-%d').tolist()

    return {
        "script": script,
        "actual": {
            "dates": actual_dates,
            "prices": actual_prices
        },
        "predicted": {
            "dates": future_dates,
            "prices": future_prices.tolist()
        }
    }
