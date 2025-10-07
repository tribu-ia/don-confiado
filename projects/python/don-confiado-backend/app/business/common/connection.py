import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .base import Base

# Cargar variables del archivo .env
load_dotenv()

USER = os.getenv("donconfiado_db_user")
PASSWORD = os.getenv("donconfiado_db_password")
HOST = os.getenv("donconfiado_db_host")
PORT = os.getenv("donconfiado_db_port")
DBNAME = os.getenv("donconfiado_db_dbname")

DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
print(DATABASE_URL)
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)