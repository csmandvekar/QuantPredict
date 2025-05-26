from sqlalchemy.orm import Session
from . import models, schemas, auth


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(db: Session, user: schemas.UserCreate):
    hashed = auth.get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_prediction(db: Session, user_id: int, symbol: str, forecast: list):
    db_pred = models.Prediction(user_id=user_id, symbol=symbol, forecast=forecast)
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred


def get_user_predictions(db: Session, user_id: int):
    return db.query(models.Prediction).filter(models.Prediction.user_id == user_id).all()


def get_prediction(db: Session, pred_id: int, user_id: int):
    return db.query(models.Prediction).filter(models.Prediction.id == pred_id, models.Prediction.user_id == user_id).first()


def delete_prediction(db: Session, pred_id: int, user_id: int):
    pred = get_prediction(db, pred_id, user_id)
    if pred:
        db.delete(pred)
        db.commit()
    return pred