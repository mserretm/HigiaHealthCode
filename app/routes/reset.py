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
from transformers import LongformerModel, LongformerConfig, LongformerTokenizer

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
    - Recarrega el Longformer i el tokenizer des de Hugging Face
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
            
            # Recargar Longformer desde el modelo local
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            LOCAL_LONGFORMER_PATH = os.path.join(BASE_DIR, "models", "clinical-longformer")
            MODEL_ID = "allenai/longformer-base-4096"
            
            try:
                # Eliminar el modelo actual si existe
                if os.path.exists(LOCAL_LONGFORMER_PATH):
                    import shutil
                    shutil.rmtree(LOCAL_LONGFORMER_PATH)
                    logger.info("Model actual eliminat correctament")
                
                # Primero cargar la configuración original
                original_config = LongformerConfig.from_pretrained(MODEL_ID)
                
                # Modificar solo los parámetros que queremos cambiar
                original_config.num_hidden_layers = 4
                original_config.attention_window = [512] * 4
                original_config.gradient_checkpointing = True
                original_config.use_cache = False
                
                # Descargar y configurar el tokenizer
                logger.info("Descarregant i configurant tokenizer...")
                tokenizer = LongformerTokenizer.from_pretrained(MODEL_ID)
                tokenizer.save_pretrained(LOCAL_LONGFORMER_PATH)
                logger.info("Tokenizer desat correctament")
                
                # Descargar y configurar el modelo
                logger.info("Descarregant model des de Hugging Face...")
                model.longformer = LongformerModel.from_pretrained(
                    MODEL_ID,
                    config=original_config,
                    ignore_mismatched_sizes=True,
                    force_download=True  # Forzar nueva descarga
                )
                
                # Guardar el modelo y la configuración en la ruta local
                model.longformer.save_pretrained(LOCAL_LONGFORMER_PATH)
                
                logger.info("Longformer descarregat i reinicialitzat correctament amb configuració reduïda")
                logger.info(f"Vocabulari size: {model.longformer.config.vocab_size}")
                logger.info(f"Capes: {model.longformer.config.num_hidden_layers}")
            except Exception as e:
                logger.error(f"Error en reinicialitzar Longformer: {str(e)}")
                raise
            
            # Reinicialitzar la resta de capes
            for module in [model.text_encoder, model.code_classifier, model.order_classifier]:
                if module is not None:
                    module.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            
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

