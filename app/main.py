# app/api/main.py

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.routes import predict, train, train_in_batch, reset  
import os
from transformers import LongformerTokenizer, LongformerModel

# Configuració del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verificar_models():
    """
    Verifica si els models necessaris estan descarregats i els descarrega si cal.
    """
    try:
        # Definir la ruta local del model Longformer
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        LOCAL_LONGFORMER_PATH = os.path.join(BASE_DIR, "models", "clinical-longformer")
        
        # Verificar si el model existeix
        if not os.path.exists(LOCAL_LONGFORMER_PATH):
            logger.info("No s'ha trobat el model Clinical-Longformer. Iniciant descàrrega...")
            
            # Crear el directori si no existeix
            os.makedirs(LOCAL_LONGFORMER_PATH, exist_ok=True)
            
            # Descarregar el tokenizer i el model
            tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
            
            # Guardar el tokenizer i el model localment
            logger.info(f"Guardant el model a {LOCAL_LONGFORMER_PATH}")
            tokenizer.save_pretrained(LOCAL_LONGFORMER_PATH)
            model.save_pretrained(LOCAL_LONGFORMER_PATH)
            
            logger.info("Model descarregat i guardat correctament.")
        else:
            logger.info("Model Clinical-Longformer trobat correctament.")
            
    except Exception as e:
        logger.error(f"Error en verificar/descarregar els models: {e}")
        raise

def create_app() -> FastAPI:
    app = FastAPI(
        title="API de Models de Codificació",
        description="API per predicció i entrenament de models de codificació CIE-10. Permet realitzar prediccions, entrenar el model amb nous casos, entrenar en batch i reiniciar el model quan sigui necessari.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    logger.info("Iniciant create_app: carregant recursos...")

    # Verificar i descarregar models si cal
    verificar_models()

    # Incloure routers
    app.include_router(predict.router, prefix="/predict", tags=["Predicció"])
    app.include_router(train.router, prefix="/train", tags=["Entrenament"])
    app.include_router(train_in_batch.router, prefix="/train-batch", tags=["Entrenament Batch"])
    app.include_router(reset.router, prefix="/reset", tags=["Reinicialització"])

    # Endpoint de benvinguda
    @app.get("/")
    def home():
        return {"message": "HigiaHealthCode API", "version": "1.0.0"}

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
