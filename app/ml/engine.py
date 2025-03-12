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
import torch.nn as nn

# Configurar logging
logger = logging.getLogger(__name__)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
LOCAL_LONGFORMER_PATH = os.path.join(MODEL_DIR, "clinical-longformer")
DATA_DIR = os.path.join(BASE_DIR, "app", "data")
CIM10MC_PATH = os.path.join(DATA_DIR, "CIM10MC_2024-2025_20231221.txt")
MODEL_ID = "allenai/longformer-base-4096"

# Crear directoris necessaris
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Llegir els codis CIM10MC o utilitzar llista per defecte
if not os.path.exists(CIM10MC_PATH):
    raise FileNotFoundError(f"No s'ha trobat el fitxer CIM10MC a {CIM10MC_PATH}. Aquest fitxer és obligatori.")

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
    # Añadir MultiLabelBinarizer a la lista de globals seguros
    torch.serialization.add_safe_globals(['sklearn.preprocessing._label.MultiLabelBinarizer'])
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        checkpoint_model_state = checkpoint['model_state_dict']
        
        # Obtener dimensiones actuales del modelo guardado
        old_num_labels = checkpoint_model_state['classifier.weight'].size(0)
        logger.info(f"Model actual té {old_num_labels} etiquetes")
        logger.info(f"Es necessiten {NUM_LABELS} etiquetes")
        
        # Crear nuevo modelo con las dimensiones actualizadas
        model = CIE10Classifier(num_labels=NUM_LABELS)
        
        # Copiar pesos existentes y expandir para nuevas etiquetas
        for name, param in checkpoint_model_state.items():
            if 'classifier' in name or 'order_classifier' in name or 'code_classifier' in name:
                if 'weight' in name:
                    # Copiar pesos existentes
                    model.state_dict()[name][:old_num_labels, :] = param
                    # Inicializar nuevos pesos
                    if NUM_LABELS > old_num_labels:
                        logger.info(f"Expandint capa {name} per {NUM_LABELS - old_num_labels} noves etiquetes")
                        nn.init.xavier_uniform_(model.state_dict()[name][old_num_labels:, :])
                elif 'bias' in name:
                    # Copiar bias existentes
                    model.state_dict()[name][:old_num_labels] = param
                    # Inicializar nuevos bias
                    if NUM_LABELS > old_num_labels:
                        model.state_dict()[name][old_num_labels:].zero_()
            else:
                # Copiar el resto de parámetros tal cual
                model.state_dict()[name].copy_(param)
        
        # Crear nuevo optimizador y scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        logger.info("Model expandit correctament amb les noves etiquetes")
            
    except Exception as e:
        logger.error(f"Error al expandir el model: {str(e)}")
        raise
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
