# app/routes/reset.py

import logging
from fastapi import APIRouter, HTTPException
import torch
import os
from app.ml.engine import (
    model, 
    tokenizer, 
    DEVICE,
    mlb, 
    NUM_LABELS,
    save_model, 
    optimizer, 
    scheduler, 
    MODEL_PATH, 
    MODEL_DIR
)
from app.ml.training_lock import training_lock

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_reset_info():
    """
    Proporciona informació sobre com utilitzar l'endpoint de reinicialització.
    """
    return {
        "message": "Per reinicialitzar el model, feu una petició POST a aquest endpoint.",
        "instruccions": "Utilitzeu el mètode POST sense cos per reinicialitzar el model als seus pesos inicials.",
        "exemple": {
            "mètode": "POST",
            "url": "/reset/",
            "resposta_esperada": {
                "message": "Model reinicialitzat correctament. - curl -X POST http://localhost:8000/reset/"
            }
        }
    }

@router.post("/")
async def reset_model():
    """
    Endpoint per reinicialitzar el model als seus pesos inicials.
    
    Aquest endpoint:
    - Reinicialitza tots els paràmetres del model
    - Reinicia l'optimitzador
    - Reinicia l'scheduler d'aprenentatge
    - Guarda el model reinicialitzat
    
    Returns:
        dict: Missatge confirmant la reinicialització
    """
    try:
        with training_lock:  # Adquirir el lock
            logger.info("Iniciant reinicialització del model...")
            
            # Reinicialitzar el model
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            
            # Reinicialitzar l'optimitzador
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            
            # Reinicialitzar l'scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            
            # Guardar el model reinicialitzat
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                mlb=mlb,
                tokenizer=tokenizer,
                model_path=MODEL_PATH,
                model_dir=MODEL_DIR
            )
            
            logger.info("Model reinicialitzat correctament.")
            return {"message": "Model reinicialitzat correctament."}
            
    except Exception as e:
        logger.error(f"Error en reinicialitzar el model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

