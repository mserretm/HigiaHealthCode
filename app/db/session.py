from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Crear l'engine de SQLAlchemy
engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)

# Crear la sessió
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Funció per obtenir una sessió de base de dades.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 