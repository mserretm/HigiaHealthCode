# app/routes/predict.py

import logging
from fastapi import APIRouter, HTTPException
from typing import List
import torch
import numpy as np
import pandas as pd
import os

from app.schemas.requests import PredictRequest
from app.ml.utils import truncate_field, process_clinical_course
from app.ml.engine import model, tokenizer, DEVICE, mlb

# Configurar logging
logger = logging.getLogger(__name__)
router = APIRouter()

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

# Carregar dades CIE-10
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CIE10_DATA_PATH = os.path.join(BASE_DIR, "app", "data", "CIM10MC_2024-2025_20231221.txt")

# Definir les columnes i les seves amplades segons el format fix
colspecs = [
    (17, 32),    # Codi (15 espais des de posició 17)
    (32, 287),   # Descr_codi (255 espais a continuació)
]

# Crear DataFrame des del fitxer de text amb format fix
try:
    cie10_data = pd.read_fwf(
        CIE10_DATA_PATH,
        colspecs=colspecs,
        encoding='latin1',
        names=['Codi', 'Descriptiu']
    )
    logger.info(f"Fitxer CIE-10 carregat correctament. {len(cie10_data)} registres trobats.")
except Exception as e:
    logger.error(f"Error en carregar el fitxer CIE-10: {str(e)}")
    raise

# Netejar i processar les dades
cie10_data['Codi'] = cie10_data['Codi'].str.strip()
cie10_data['Descriptiu'] = cie10_data['Descriptiu'].str.strip()
cie10_data = cie10_data[cie10_data['Codi'].notna()]
cie10_data = cie10_data[cie10_data['Codi'] != '']

@router.post("/")
async def predict_case(request: PredictRequest):
    """
    Endpoint per realitzar prediccions de codis CIE-10.
    
    Args:
        request: PredictRequest amb el cas clínic complet
        
    Returns:
        dict: Prediccions ordenades per rellevància amb descriptius
    """
    try:
        logger.info("Rebuda petició de predicció")
        
        # Preparar inputs
        inputs = {}
        
        # Procesar variables categóricas
        if request.case.genere:
            value = request.case.genere.strip().upper()
            if value in GENERE_MAPPING:
                inputs["genere"] = torch.tensor([GENERE_MAPPING[value]], device=DEVICE)

        if request.case.edat:
            try:
                inputs["edat"] = torch.tensor([int(request.case.edat)], device=DEVICE)
            except ValueError:
                logger.warning(f"Valor d'edat no vàlid: {request.case.edat}")

        # Procesar campos de texto
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
            value = getattr(request.case, field, None)
            if value:
                if field == "cursclinic":
                    processed_text = process_clinical_course(value, FIELD_LIMITS[field])
                else:
                    processed_text = truncate_field(value, FIELD_LIMITS[field])

                encoded = tokenizer(
                    processed_text,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=4096
                ).to(DEVICE)

                inputs[f"{input_key}_input_ids"] = encoded["input_ids"]
                inputs[f"{input_key}_attention_mask"] = encoded["attention_mask"]
        
        # Predicción
        model.eval()
        with torch.no_grad():
            code_probs, order_probs = model(inputs)
            probabilities = torch.sigmoid(code_probs)
        
        # Obtenir prediccions ordenades
        probs_np = probabilities.cpu().numpy()[0]
        indices = (-probs_np).argsort()
        
        # Preparar resultats amb descriptius
        predictions = []
        for idx in indices:
            code = mlb.classes_[idx]
            prob = float(probs_np[idx])
            
            if prob > 0.5:  # Només incloure prediccions amb probabilitat > 0.5
                # Buscar descriptiu
                code_info = cie10_data[cie10_data['Codi'] == code]
                descriptive = code_info.iloc[0]['Descriptiu'] if not code_info.empty else "No trobat"
                
                predictions.append({
                    "code": code,
                    "descriptive": descriptive,
                    "probability": prob
                })
        
        logger.info(f"Predicció completada amb èxit. {len(predictions)} codis predits.")
        return {
            "case_id": request.case.cas,
            "predictions": predictions,
            "text_length": len(full_text)
        }
        
    except Exception as e:
        logger.error(f"Error en la predicció: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
