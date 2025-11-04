import os

class Config:
    SECRET_KEY = os.getenv("JWT_SECRET", "supersecretkey")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
    DB_NAME = os.getenv("DB_NAME", "golombdb")
