from pydantic_settings import BaseSettings
from urllib.parse import quote_plus

class Settings(BaseSettings):
    PROJECT_NAME: str = "Higia Health Code API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API per la classificació automàtica de codis CIE-10"
    API_V1_STR: str = "/api/v1"
    
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "ia_controled_access"
    POSTGRES_PASSWORD: str = "Xarxa2025!"
    POSTGRES_DB: str = "dwh_cubes"
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        password = quote_plus(self.POSTGRES_PASSWORD)
        return f"postgresql://{self.POSTGRES_USER}:{password}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"
    
    @property
    def DATABASE_URL(self) -> str:
        return self.SQLALCHEMY_DATABASE_URI
    
    class Config:
        case_sensitive = True

settings = Settings()
