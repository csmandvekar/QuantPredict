from pydantic import BaseModel, EmailStr
from typing import List
from datetime import datetime


class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr

    class Config:
        orm_mode = True


class PredictionCreate(BaseModel):
    symbol: str
    data: List[List[float]]


class PredictionOut(BaseModel):
    id: int
    symbol: str
    created_at: datetime
    forecast: List[float]

    class Config:
        orm_mode = True