from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import prediction

app = FastAPI(title="Stock Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(prediction.router)
