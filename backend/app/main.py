from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import auth, predict

app = FastAPI(title="QuantPredict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(predict.router)

@app.get("/")
def health_check():
    return {"status": "ok"}