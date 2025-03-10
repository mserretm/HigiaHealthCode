# app/ml/engine.py

"""
Mòdul principal per la gestió del model i el seu entrenament.
Proporciona funcionalitats per carregar, entrenar i fer prediccions amb el model.
"""

import logging
import torch
import os
import shutil
from transformers import LongformerTokenizer, LongformerModel
from app.ml.model import CIE10Classifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Configurar logging
logger = logging.getLogger(__name__)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
LOCAL_LONGFORMER_PATH = os.path.join(MODEL_DIR, "clinical-longformer")
DATA_DIR = os.path.join(BASE_DIR, "data")
CIM10MC_PATH = os.path.join(DATA_DIR, "CIM10MC_2024-2025_20231221.txt")
MODEL_ID = "allenai/longformer-base-4096"

# Crear directoris necessaris
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Llegir els codis CIM10MC o utilitzar llista per defecte
if os.path.exists(CIM10MC_PATH):
    df_codis = pd.read_fwf(
        CIM10MC_PATH,
        colspecs=[(17, 32)],  # Codi (15 espais des de posició 17)
        encoding='latin1',
        names=['Codi']
    )
    
    # Netejar i obtenir codis únics
    df_codis['Codi'] = df_codis['Codi'].str.strip()
    codis_unics = sorted(df_codis['Codi'].unique().tolist())
    logger.info(f"Carregats {len(codis_unics)} codis únics del fitxer CIM10MC")
else:
    # Llista per defecte de codis més comuns
    codis_unics = [
        'A41.9', 'E11.9', 'I10', 'I21.9', 'I50.0', 
        'J18.9', 'J44.9', 'K80.2', 'N39.0', 'R50.9'
    ]
    logger.warning(f"No s'ha trobat el fitxer CIM10MC a {CIM10MC_PATH}. Utilitzant llista per defecte de {len(codis_unics)} codis.")

# Inicialitzar MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=codis_unics)
mlb.fit([[]])

NUM_LABELS = len(codis_unics)
logger.info(f"Total de codis disponibles: {NUM_LABELS}")

# Verificar si necesitamos descarregar el model
config_path = os.path.join(LOCAL_LONGFORMER_PATH, "config.json")
if not os.path.exists(LOCAL_LONGFORMER_PATH) or not os.path.exists(config_path):
    logger.info(f"Model no trobat localment. Descarregant des de Hugging Face ({MODEL_ID})...")
    
    # Si existeix el directori pero està incompleto, el eliminem
    if os.path.exists(LOCAL_LONGFORMER_PATH):
        shutil.rmtree(LOCAL_LONGFORMER_PATH)
    
    # Descarregar i guardar el model
    tokenizer = LongformerTokenizer.from_pretrained(MODEL_ID)
    base_model = LongformerModel.from_pretrained(MODEL_ID)
    
    os.makedirs(LOCAL_LONGFORMER_PATH, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_LONGFORMER_PATH)
    base_model.save_pretrained(LOCAL_LONGFORMER_PATH)
    logger.info("Model descarregat i desat localment")

# Cargar el model des del directori local
logger.info("Carregant model des del directori local...")
tokenizer = LongformerTokenizer.from_pretrained(LOCAL_LONGFORMER_PATH, local_files_only=True)
logger.info("Tokenizer carregat correctament")

# Cargar o crear el modelo
if os.path.exists(MODEL_PATH):
    logger.info(f"Carregant model entrenat des de {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CIE10Classifier(num_labels=NUM_LABELS)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    mlb = checkpoint['mlb']
else:
    logger.info("Creant nou model de classificació")
    model = CIE10Classifier(num_labels=NUM_LABELS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

model = model.to(DEVICE)

def prepare_text_inputs(case: Dict[str, Any]) -> str:
    """
    Prepara els camps de text per la tokenització.
    
    Args:
        case: Diccionari amb les dades del cas clínic
        
    Returns:
        Text preparat per tokenitzar
        
    Exemple:
        >>> case = {"motiuingres": "Dolor", "malaltiaactual": "Febre"}
        >>> prepare_text_inputs(case)
        'Dolor [SEP] Febre'
    """
    text_fields = [
        case.get('motiuingres', ''),
        case.get('malaltiaactual', ''),
        case.get('exploracio', ''),
        case.get('provescomplementariesing', ''),
        case.get('provescomplementaries', ''),
        case.get('evolucio', ''),
        case.get('antecedents', ''),
        case.get('cursclinic', '')
    ]
    
    return ' [SEP] '.join(filter(None, text_fields))

def prepare_categorical_inputs(case: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Prepara les variables categòriques pel model.
    
    Args:
        case: Diccionari amb les dades del cas
        
    Returns:
        Dict amb els tensors de les variables categòriques
        
    Exemple:
        >>> case = {"edat": 45, "genere": 1}
        >>> inputs = prepare_categorical_inputs(case)
        >>> inputs['edat'].item()
        45
    """
    categorical_fields = {
        'edat': 0,
        'genere': 0,
        'c_alta': 0,
        'periode': 0,
        'servei': 0
    }
    
    # Actualitzar amb els valors del cas
    for field in categorical_fields:
        if field in case:
            try:
                categorical_fields[field] = int(case[field])
            except (ValueError, TypeError):
                logger.warning(f"Valor invàlid per {field}: {case[field]}")
    
    # Convertir a tensors
    return {
        field: torch.tensor([value], device=DEVICE)
        for field, value in categorical_fields.items()
    }

def save_model(
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    mlb: MultiLabelBinarizer,
    tokenizer: LongformerTokenizer,
    model_path: str,
    model_dir: str
) -> None:
    """
    Guarda el model i els seus components.
    
    Args:
        model: Model a guardar
        optimizer: Optimitzador
        scheduler: Scheduler
        mlb: MultiLabelBinarizer
        tokenizer: Tokenizer
        model_path: Ruta on guardar el model
        model_dir: Directori del model
        
    Raises:
        OSError: Si hi ha problemes en crear el directori o guardar el model
    """
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'mlb': mlb,
            'tokenizer': tokenizer
        }, model_path)
        
        logger.info(f"Model guardat a {model_path}")
        
    except OSError as e:
        logger.error(f"Error en guardar el model: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error inesperat en guardar el model: {str(e)}")
        raise

def load_model(model_path: str) -> bool:
    """
    Carrega el model i els seus components.
    
    Args:
        model_path: Ruta del model a carregar
        
    Returns:
        bool: True si s'ha carregat correctament
        
    Raises:
        OSError: Si no es troba el fitxer o hi ha problemes en llegir-lo
    """
    try:
        if not os.path.exists(model_path):
            logger.warning(f"No s'ha trobat el model a {model_path}")
            return False
            
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        global mlb, tokenizer
        mlb = checkpoint['mlb']
        tokenizer = checkpoint['tokenizer']
        
        logger.info(f"Model carregat des de {model_path}")
        return True
            
    except OSError as e:
        logger.error(f"Error en carregar el model: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error inesperat en carregar el model: {str(e)}")
        return False

def ordenar_codis_per_prediccio(
    predictions: List[Tuple[str, float, float]],
    threshold: float = 0.5
) -> List[str]:
    """
    Ordena els codis segons les prediccions d'ordre.
    
    Args:
        predictions: Llista de tuples (codi, prob_codi, prob_ordre)
        threshold: Llindar de probabilitat mínim
        
    Returns:
        Llista ordenada de codis
        
    Exemple:
        >>> preds = [("A01", 0.8, 0.9), ("B02", 0.6, 0.7)]
        >>> ordenar_codis_per_prediccio(preds)
        ['A01', 'B02']
    """
    filtered_predictions = [
        (code, code_prob, order_prob) 
        for code, code_prob, order_prob in predictions 
        if code_prob > threshold
    ]
    
    sorted_predictions = sorted(
        filtered_predictions,
        key=lambda x: x[2],
        reverse=True
    )
    
    return [code for code, _, _ in sorted_predictions]

def predict_case(
    case: Dict[str, Any],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Realitza prediccions per un cas clínic.
    
    Args:
        case: Diccionari amb les dades del cas
        threshold: Llindar de probabilitat mínim
        
    Returns:
        Dict amb l'identificador del cas i les prediccions
        
    Raises:
        ValueError: Si falten dades necessàries
        RuntimeError: Si hi ha errors en la predicció
    """
    try:
        if 'cas' not in case:
            raise ValueError("Falta l'identificador del cas")
            
        # Preparar inputs
        text_input = prepare_text_inputs(case)
        if not text_input.strip():
            raise ValueError("No s'han trobat dades de text pel cas")
            
        tokenized = tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=4096
        ).to(DEVICE)
        
        categorical_inputs = prepare_categorical_inputs(case)
        inputs = {**tokenized, **categorical_inputs}
        
        # Predicció
        model.eval()
        with torch.no_grad():
            code_probs, order_probs = model(inputs)
            
        # Processar prediccions
        predictions = []
        code_probs = code_probs[0].cpu().numpy()
        order_probs = order_probs[0].cpu().numpy()
        
        for i, (code_prob, order_prob) in enumerate(zip(code_probs, order_probs)):
            if code_prob > threshold:
                code = mlb.classes_[i]
                predictions.append((code, float(code_prob), float(order_prob)))
        
        ordered_codes = ordenar_codis_per_prediccio(predictions, threshold)
        
        return {
            'cas': case['cas'],
            'prediccions': ordered_codes
        }
        
    except ValueError as e:
        logger.error(f"Error de validació: {str(e)}")
        raise
    except RuntimeError as e:
        logger.error(f"Error en la predicció: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error inesperat: {str(e)}")
        raise
