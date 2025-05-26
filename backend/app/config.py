import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    ENV: str = os.getenv("ENV", "production")
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALPHA_VANTAGE_KEY: str

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), '..', '.env')


settings = Settings()