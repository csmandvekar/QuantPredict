from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from app.api.dependencies import Base
from datetime import datetime

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    script = Column(String(10), nullable=False)
    run_time = Column(DateTime, default=datetime.utcnow)
    predicted_range = Column(Text)  # Store JSON as string
    actual_close = Column(Float)
    future_avg_close = Column(Float)
