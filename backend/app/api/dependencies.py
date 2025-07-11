from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
import os

# SQLite for demo
DB_PATH = os.path.join(os.getcwd(), "database", "stock_data.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency for route injection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
