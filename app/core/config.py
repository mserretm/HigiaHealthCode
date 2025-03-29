from pydantic_settings import BaseSettings
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Higia Health Code API")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    DESCRIPTION: str = os.getenv("DESCRIPTION", "API per la classificació automàtica de codis CIE-10")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        password = quote_plus(self.POSTGRES_PASSWORD)
        return f"postgresql://{self.POSTGRES_USER}:{password}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"
    
    @property
    def DATABASE_URL(self) -> str:
        return self.SQLALCHEMY_DATABASE_URI
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
