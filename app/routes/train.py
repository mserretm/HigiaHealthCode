# app/routes/train.py

import logging
from fastapi import APIRouter, HTTPException
from typing import List
import torch
import numpy as np
from app.schemas.requests import TrainRequest
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
from app.ml.utils import truncate_field, is_relevant, freeze_bert_layers, process_clinical_course
from app.ml.training_lock import training_lock

router = APIRouter()
logger = logging.getLogger(__name__)

# Mapatge per al camp gènere
GENERE_MAPPING = {'H': 0, 'D': 1}  # Home, Dona

# Configuració de límits de camp
FIELD_LIMITS = {
    'motiuingres': 1000,  # Augmentar de 255 a 1000
    'malaltiaactual': 10000,
    'exploracio': 5000,
    'provescomplementariesing': 10000,
    'provescomplementaries': 20000,
    'evolucio': 15000,
    'antecedents': 10000,
    'cursclinic': 100000  # Límit raonable per processar
}

@router.post("/")
async def train_case(request: TrainRequest):
    """
    Endpoint per realitzar entrenament incremental del model amb un nou cas.
    """
    try:
        with training_lock:  # Adquirir el lock
            logger.info("Iniciant entrenament incremental...")

            # 1) Congelar capes i descongelar les últimes dues
            freeze_bert_layers(model.longformer, num_unfrozen_layers=2)

            # 2) Preparar inputs
            inputs = {}
            expected_keys = {"genere", "edat", "malaltia_actual", "exploracio", "proves_complementaries_ingres", "proves_complementaries", "recomanacio_tractament", "evolucio", "motiu_ingres", "curs_clinic"}

            for key, value in request.case.items():
                if key not in expected_keys:
                    logger.warning(f"Clau no esperada: {key}. S'ignorarà aquesta entrada.")
                    continue

                if key == "genere":
                    if isinstance(value, str):
                        value = value.strip().upper()  # Normalitzar el valor de gènere
                        if value not in GENERE_MAPPING:
                            logger.warning(f"El valor de gènere '{value}' no és vàlid. Valors permesos: {list(GENERE_MAPPING.keys())}. S'ignorarà aquesta entrada.")
                            continue
                        inputs[key] = torch.tensor([GENERE_MAPPING[value]], device=DEVICE)
                elif key == "edat":
                    try:
                        inputs[key] = torch.tensor([int(value)], device=DEVICE)
                    except ValueError:
                        logger.warning(f"El valor d'edat '{value}' no és vàlid. S'ignorarà aquesta entrada.")
                        continue
                else:
                    # Processar el text segons el camp
                    if key == "curs_clinic":
                        processed_text = process_clinical_course(value, FIELD_LIMITS[key])
                    else:
                        max_length = FIELD_LIMITS.get(key, 5000)  # 5000 per defecte
                        processed_text = truncate_field(value, max_length)

                    encoded = tokenizer(
                        processed_text,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=4096
                    ).to(DEVICE)
                    inputs[key] = encoded["input_ids"]

            logger.info("Text i entrades preparades per entrenament.")

            # 3) Preparar vector d'etiquetes
            label_vector = mlb.transform([request.codes])
            labels = torch.from_numpy(np.array(label_vector)).float().to(DEVICE)

            # 4) Configurar pèrdua
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(NUM_LABELS).to(DEVICE) * 2.0)

            # 5) Entrenar (exemple 3 èpoques)
            model.train()
            optimizer.zero_grad()
            epochs = 3
            for epoch in range(epochs):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                logger.info(f"Època {epoch + 1}/{epochs}, Pèrdua: {loss.item():.4f}")

            # 6) Guardar el model actualitzat
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                mlb=mlb,
                tokenizer=tokenizer,
                model_path=MODEL_PATH,
                model_dir=MODEL_DIR
            )
            logger.info("Entrenament incremental completat.")

        return {"message": "Model actualitzat amb noves dades."}

    except HTTPException as e:
        logger.error(f"Error en les entrades d'entrenament: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error en realitzar l'entrenament: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en realitzar l'entrenament: {str(e)}")
