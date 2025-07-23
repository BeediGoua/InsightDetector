import os
from dotenv import load_dotenv

# Charge le .env à la racine du projet, quelle que soit ta position
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

class Settings:
    ENV = os.getenv("ENV", "development")
    DATABASE_URL = os.getenv("DATABASE_URL")

    def __post_init__(self):
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL est manquant. Vérifie le fichier .env")

settings = Settings()

# Optionnel pour debug
if not settings.DATABASE_URL:
    raise EnvironmentError("DATABASE_URL non défini dans .env")
else:
    print(f"DATABASE_URL détecté : {settings.DATABASE_URL}")
