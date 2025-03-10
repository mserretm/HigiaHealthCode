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
    'motiuingres': 1000, 
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
            logger.info("Capes congelades i descongelades les últimes dues.")

            # 2) Preparar inputs
            logger.info("Preparant inputs...")
            inputs = {}
            case_data = request.case.model_dump()

            # Processar variables categòriques
            logger.info("Processant variables categòriques...")
            if case_data.get("genere"):
                value = case_data["genere"].strip().upper()
                if value in GENERE_MAPPING:
                    inputs["genere"] = torch.tensor([GENERE_MAPPING[value]], device=DEVICE)
                else:
                    logger.warning(f"Valor de gènere no vàlid: {value}")

            if case_data.get("edat"):
                try:
                    inputs["edat"] = torch.tensor([int(case_data["edat"])], device=DEVICE)
                except ValueError:
                    logger.warning(f"Valor d'edat no vàlid: {case_data['edat']}")

            # Processar camps de text
            logger.info("Processant camps de text...")
            text_fields = {
                "motiuingres": "motiu_ingres",
                "malaltiaactual": "malaltia_actual",
                "exploracio": "exploracio",
                "provescomplementariesing": "proves_complementaries_ingres",
                "provescomplementaries": "proves_complementaries",
                "evolucio": "evolucio",
                "antecedents": "antecedents",
                "cursclinic": "curs_clinic"
            }

            # Procesar cada campo por separado
            for field, input_key in text_fields.items():
                if case_data.get(field):
                    if field == "cursclinic":
                        processed_text = process_clinical_course(case_data[field], FIELD_LIMITS[field])
                    else:
                        processed_text = truncate_field(case_data[field], FIELD_LIMITS[field])

                    # Tokenizar cada campo individualmente
                    encoded = tokenizer(
                        processed_text,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=4096
                    ).to(DEVICE)

                    # Guardar los tensores con nombres únicos para cada campo
                    inputs[f"{input_key}_input_ids"] = encoded["input_ids"]
                    inputs[f"{input_key}_attention_mask"] = encoded["attention_mask"]
                    logger.info(f"Campo {field} procesado. Shape input_ids: {encoded['input_ids'].shape}")
                else:
                    logger.info(f"Campo {field} no presente en los datos")

            logger.info("Text i entrades preparades per entrenament.")
            logger.info(f"Claves disponibles en inputs: {list(inputs.keys())}")
            
            # Verificar cada campo procesado
            for input_key in text_fields.values():
                if f"{input_key}_input_ids" in inputs:
                    logger.info(f"Tensor {input_key}_input_ids shape: {inputs[f'{input_key}_input_ids'].shape}")
                    logger.info(f"Tensor {input_key}_attention_mask shape: {inputs[f'{input_key}_attention_mask'].shape}")

            # 3) Preparar vector d'etiquetes
            logger.info("Preparant vector d'etiquetes...")
            dx_revisat = case_data.get("dx_revisat", "")
            if isinstance(dx_revisat, str):
                codes = dx_revisat.split("|")
            else:
                codes = dx_revisat
            label_vector = mlb.transform([codes])
            labels = torch.from_numpy(np.array(label_vector)).float().to(DEVICE)

            # 4) Configurar pèrdua
            logger.info("Configurant funció de pèrdua...")
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(NUM_LABELS).to(DEVICE) * 2.0)

            # 5) Entrenar (exemple 3 èpoques)
            logger.info("Iniciant entrenament...")
            model.train()
            optimizer.zero_grad()
            epochs = 3
            for epoch in range(epochs):
                code_probs, order_probs = model(inputs)
                loss = loss_fn(code_probs, labels)  # Solo usamos la pérdida de clasificación por ahora
                loss.backward()
                optimizer.step()
                scheduler.step()
                logger.info(f"Època {epoch + 1}/{epochs}, Pèrdua: {loss.item():.4f}")

            # 6) Guardar el model actualitzat
            logger.info("Guardant el model actualitzat...")
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
