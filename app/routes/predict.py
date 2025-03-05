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
        
        # Preparar text combinant tots els camps rellevants
        text_fields = []
        
        # Processar cada camp amb els seus límits
        fields_config = {
            'cas': (request.case.cas, None),  # None significa sense límit
            'edat': (request.case.edat, None),
            'genere': (request.case.genere, None),
            'servei': (request.case.servei, None),
            'motiuingres': (request.case.motiuingres, FIELD_LIMITS['motiuingres']),
            'malaltiaactual': (request.case.malaltiaactual, FIELD_LIMITS['malaltiaactual']),
            'antecedents': (request.case.antecedents, FIELD_LIMITS['antecedents']),
            'exploracio': (request.case.exploracio, FIELD_LIMITS['exploracio']),
            'provescomplementariesing': (request.case.provescomplementariesing, FIELD_LIMITS['provescomplementariesing']),
            'provescomplementaries': (request.case.provescomplementaries, FIELD_LIMITS['provescomplementaries']),
            'evolucio': (request.case.evolucio, FIELD_LIMITS['evolucio']),
            'cursclinic': (request.case.cursclinic, FIELD_LIMITS['cursclinic'])
        }
        
        for field_name, (value, limit) in fields_config.items():
            if not value:
                continue
                
            if field_name == 'cursclinic':
                processed_value = process_clinical_course(value, limit) if limit else value
            else:
                processed_value = truncate_field(value, limit) if limit else value
                
            prefix = field_name.replace('_', ' ').title()
            text_fields.append(f"{prefix}: {processed_value}")
        
        # Filtrar camps buits i unir amb [SEP]
        full_text = " [SEP] ".join([field for field in text_fields if field.split(": ")[1].strip()])
        
        # Tokenitzar
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=4096
        ).to(DEVICE)
        
        # Predicció
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)
        
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
