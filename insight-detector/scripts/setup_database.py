import sys
sys.path.append(".")

from src.config.database import engine
from src.data.models import Base

if __name__ == "__main__":
    print("Création de la base de données...")
    Base.metadata.create_all(bind=engine)
    print("Base initialisée !")
