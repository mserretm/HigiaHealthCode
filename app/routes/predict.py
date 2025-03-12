# app/routes/predict.py

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Optional
import torch
import numpy as np
import pandas as pd
import os
from functools import lru_cache
import time
from concurrent.futures import TimeoutError

from app.schemas.requests import PredictRequest
from app.ml.utils import truncate_field, process_clinical_course
from app.ml.engine import model, tokenizer, DEVICE, mlb

# Configurar logging
logger = logging.getLogger(__name__)
router = APIRouter()

# Configuració de límits de camp
FIELD_LIMITS = {
    'motiuingres': 1000,
    'malaltiaactual': 10000,
    'exploracio': 5000,
    'provescomplementariesing': 10000,
    'provescomplementaries': 20000,
    'evolucio': 15000,
    'antecedents': 10000,
    'cursclinic': 100000
}

# Mapeig de gènere
GENERE_MAPPING = {
    'H': 0,  # Home
    'D': 1,  # Dona
    'HOME': 0,
    'DONA': 1,
    'M': 0,  # Masculí
    'F': 1,  # Femení
    'MASCULÍ': 0,
    'FEMENÍ': 1,
    'MASCULINO': 0,
    'FEMENINO': 1
}

# Memòria cau pel DataFrame CIE-10
@lru_cache(maxsize=1)
def load_cie10_data():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        CIE10_DATA_PATH = os.path.join(BASE_DIR, "app", "data", "CIM10MC_2024-2025_20231221.txt")
        
        colspecs = [(17, 32), (32, 287)]
        df = pd.read_fwf(
            CIE10_DATA_PATH,
            colspecs=colspecs,
            encoding='latin1',
            names=['Codi', 'Descriptiu']
        )
        
        df['Codi'] = df['Codi'].str.strip()
        df['Descriptiu'] = df['Descriptiu'].str.strip()
        df = df[df['Codi'].notna()]
        df = df[df['Codi'] != '']
        
        return df
    except Exception as e:
        logger.error(f"Error en carregar el fitxer CIE-10: {str(e)}")
        raise

# Carregar dades CIE-10 una sola vegada a l'inici
cie10_data = load_cie10_data()

async def process_prediction(inputs: Dict) -> tuple:
    """
    Processa la predicció amb un temps límit
    """
    try:
        start_time = time.time()
        with torch.no_grad():
            code_probs, order_probs = model(inputs)
            probabilities = torch.sigmoid(code_probs)
            
        if time.time() - start_time > 30:  # Temps límit de 30 segons
            raise TimeoutError("La predicció ha excedit el temps màxim permès")
            
        return probabilities
    except Exception as e:
        logger.error(f"Error en el processament de la predicció: {str(e)}")
        raise

@router.post("/")
async def predict_case(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Endpoint per realitzar prediccions de codis CIE-10.
    """
    start_time = time.time()
    try:
        logger.info(f"Rebuda petició de predicció pel cas: {request.case.cas}")
        
        # Mostrar resum del cas
        logger.info("\nResum del cas:")
        logger.info("-" * 80)
        if request.case.motiuingres:
            logger.info(f"Motiu ingrés: {request.case.motiuingres[:200]}...")
        if request.case.malaltiaactual:
            logger.info(f"Malaltia actual: {request.case.malaltiaactual[:200]}...")
        logger.info("-" * 80)
        
        # Preparar entrades amb gestió de memòria
        try:
            inputs = {}
            logger.info("Processant variables categòriques...")
            
            if request.case.genere:
                value = request.case.genere
                if value in GENERE_MAPPING:
                    inputs["genere"] = torch.tensor([GENERE_MAPPING[value]], device=DEVICE)
                    logger.info(f"Gènere processat: {value} -> {GENERE_MAPPING[value]}")

            if request.case.edat:
                inputs["edat"] = torch.tensor([int(request.case.edat)], device=DEVICE)
                logger.info(f"Edat processada: {request.case.edat}")

            # Processar camps de text amb gestió de memòria
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

            total_text_length = 0
            for field, input_key in text_fields.items():
                value = getattr(request.case, field, None)
                if value:
                    total_text_length += len(value)
                    if total_text_length > 1_000_000:  # Límit d'1M caràcters
                        raise ValueError("El text total excedeix el límit permès")
                        
                    logger.info(f"Processant camp {field} ({len(value)} caràcters)...")
                    processed_text = process_clinical_course(value, FIELD_LIMITS[field]) if field == "cursclinic" else truncate_field(value, FIELD_LIMITS[field])

                    logger.info(f"Tokenitzant {field}...")
                    encoded = tokenizer(
                        processed_text,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=4096
                    ).to(DEVICE)

                    inputs[f"{input_key}_input_ids"] = encoded["input_ids"]
                    inputs[f"{input_key}_attention_mask"] = encoded["attention_mask"]
                    logger.info(f"Camp {field} processat correctament")

            if not inputs:
                raise ValueError("No s'han proporcionat dades suficients per fer la predicció")

        except Exception as e:
            logger.error(f"Error en el preprocessament: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

        # Predicció amb temps límit
        try:
            model.eval()
            probabilities = await process_prediction(inputs)
            probs_np = probabilities.cpu().numpy()[0]
        except TimeoutError:
            raise HTTPException(status_code=408, detail="Temps d'execució excedit")
        except Exception as e:
            logger.error(f"Error en la predicció: {str(e)}")
            if "CUDA" in str(e):
                raise HTTPException(status_code=503, detail="Error en el processament GPU")
            raise HTTPException(status_code=500, detail=str(e))

        # Processar resultats
        MAX_PREDICTIONS = 15
        PROBABILITY_THRESHOLD = 0.85

        sorted_indices = np.argsort(-probs_np)
        top_indices = sorted_indices[:MAX_PREDICTIONS]
        top_probs = probs_np[top_indices]
        
        mask = top_probs > PROBABILITY_THRESHOLD
        final_indices = top_indices[mask]
        
        total_predictions = len(final_indices)
        logger.info(f"\nTotal de prediccions amb probabilitat > {PROBABILITY_THRESHOLD*100}%: {total_predictions}")
        
        if total_predictions == 0:
            logger.warning(f"No s'han trobat prediccions amb probabilitat > {PROBABILITY_THRESHOLD*100}%")
            return {
                "case_id": request.case.cas,
                "predictions": [],
                "text_length": total_text_length,
                "processing_time": time.time() - start_time
            }
        
        # Formatar resultats
        logger.info(f"\nPrediccions trobades pel cas {request.case.cas}:")
        logger.info("-" * 100)
        logger.info(f"{'Codi':<10} {'Prob':>8} {'Descripció':<80}")
        logger.info("-" * 100)
        
        predictions = []
        for idx in final_indices:
            code = mlb.classes_[idx]
            prob = float(probs_np[idx])
            
            code_info = cie10_data[cie10_data['Codi'] == code]
            descriptive = code_info.iloc[0]['Descriptiu'] if not code_info.empty else "No trobat"
            
            logger.info(f"{code:<10} {prob*100:>7.2f}% {descriptive[:80]:<80}")
            
            predictions.append({
                "code": code,
                "descriptive": descriptive,
                "probability": prob
            })
        
        logger.info("-" * 100)
        processing_time = time.time() - start_time
        logger.info(f"Predicció completada en {processing_time:.2f} segons. {len(predictions)} codis predits.")
        
        # Alliberar memòria en segon pla
        background_tasks.add_task(torch.cuda.empty_cache)
        
        return {
            "case_id": request.case.cas,
            "predictions": predictions,
            "text_length": total_text_length,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error en la predicció: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
