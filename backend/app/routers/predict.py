from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from .. import schemas, crud, auth, utils, database
from ..predictors import load_model, predict_batch
from datetime import datetime, timedelta

router = APIRouter(prefix="/predict", tags=["predict"])
CACHE_TTL = timedelta(hours=1)

@router.post("/", response_model=schemas.PredictionOut)
def make_prediction(
    req: schemas.PredictionCreate,
    background_tasks: BackgroundTasks,
    current_user=Depends(auth.get_current_user),
    db: Session = Depends(auth.get_db)
):
    # 1. Ensure market data is fresh
    latest = utils.get_latest_ts(db, req.symbol)
    now = datetime.utcnow()
    if latest is None or now - latest > CACHE_TTL:
        background_tasks.add_task(utils.refresh_symbol, req.symbol)
    # 2. Load data from DB
    df = utils.load_market_data(db, req.symbol)
    if df.empty:
        raise HTTPException(404, "No market data available yet")
    # 3. Preprocess windows
    windows, scaler = utils.preprocess(df)
    # 4. Predict
    model = load_model()
    forecast = predict_batch(model, windows, req.symbol)
    # 5. Save and return
    db_pred = crud.create_prediction(db, current_user.id, req.symbol, [float(forecast)])
    return db_pred