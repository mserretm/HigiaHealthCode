# app/api/main.py

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes import router
from app.core.config import settings

# Configuració del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestió del cicle de vida de l'aplicació.
    """
    logger.info("Iniciant l'aplicació...")
    yield
    logger.info("Tancant l'aplicació...")

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan
    )

    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info("Iniciant create_app: carregant recursos...")

    app.include_router(router)

    # Endpoint de benvinguda
    @app.get("/")
    async def root():
        """
        Endpoint principal que retorna informació bàsica de l'API.
        """
        return {"message": "Benvingut a l'API de classificació de codis CIE-10"}

    # Gestor d'errors global
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Error no gestionat: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Error intern del servidor"}
        )

    logger.info("create_app finalitzat. App llesta per usar-se.")
    return app

app = create_app()

# -------------------------------------------------------------------
# MAIN PER UVICORN
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
