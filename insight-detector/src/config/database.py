# src/config/database.py

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .settings import settings

# Vérifie que l'URL de la BDD est définie
if not settings.DATABASE_URL:
    raise ValueError(" DATABASE_URL n'est pas défini. Vérifie ton .env.")

# Crée automatiquement le dossier si on utilise SQLite
if settings.DATABASE_URL.startswith("sqlite:///"):
    sqlite_path = settings.DATABASE_URL.replace("sqlite:///", "")
    Path(os.path.dirname(sqlite_path)).mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

# Création du moteur SQLAlchemy
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)

# Session locale SQLAlchemy (à importer partout)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Générateur de session (utile pour FastAPI ou contextes dynamiques)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
