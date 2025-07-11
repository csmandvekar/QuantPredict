import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALLOWED_SCRIPTS = ["INFY", "TCS", "RELIANCE"]  # Add more as needed
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
DATA_DIR = os.path.join(BASE_DIR, "data_cache")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
