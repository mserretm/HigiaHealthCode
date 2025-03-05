# app/ml/engine.py

"""
Mòdul principal per la gestió del model i el seu entrenament.
"""

import logging
import torch
import os
from transformers import LongformerTokenizer
from app.ml.model import CIE10Classifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

# Configurar logging
logger = logging.getLogger(__name__)

# Configuració bàsica
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
LOCAL_LONGFORMER_PATH = os.path.join(MODEL_DIR, "clinical-longformer")
CIM10MC_PATH = os.path.join(BASE_DIR, "data", "CIM10MC_2024-2025_20231221.txt")

# Crear directori de models si no existeix
os.makedirs(MODEL_DIR, exist_ok=True)

# Llegir els codis CIM10MC
try:
    # Llegir el fitxer amb format fix
    df_codis = pd.read_fwf(
        CIM10MC_PATH,
        colspecs=[(17, 32)],  # Codi (15 espais des de posició 17)
        encoding='latin1',
        names=['Codi']
    )
    
    # Netejar i obtenir codis únics
    df_codis['Codi'] = df_codis['Codi'].str.strip()
    codis_unics = sorted(df_codis['Codi'].unique().tolist())
    
    # Inicialitzar MultiLabelBinarizer amb els codis
    mlb = MultiLabelBinarizer(classes=codis_unics)
    mlb.fit([[]])  # Ajustar amb una llista buida per inicialitzar
    
    NUM_LABELS = len(codis_unics)
    logger.info(f"Carregats {NUM_LABELS} codis únics del CIM10MC")
    
except Exception as e:
    logger.error(f"Error en carregar els codis CIM10MC: {str(e)}")
    raise

# Inicialitzar el tokenizer des de la ruta local
try:
    tokenizer = LongformerTokenizer.from_pretrained(LOCAL_LONGFORMER_PATH, local_files_only=True)
    logger.info(f"Tokenizer carregat correctament des de {LOCAL_LONGFORMER_PATH}")
except Exception as e:
    logger.error(f"Error en carregar el tokenizer local: {str(e)}")
    raise

# Inicialitzar el model amb el número correcte d'etiquetes
model = CIE10Classifier(num_labels=NUM_LABELS).to(DEVICE)

# Inicialitzar l'optimitzador i el scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

def prepare_text_inputs(case: Dict[str, Any]) -> str:
    """
    Prepara els camps de text per la tokenització.
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
    
    # Filtrar camps buits i unir amb separador
    return ' [SEP] '.join(filter(None, text_fields))

def prepare_categorical_inputs(case: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Prepara les variables categòriques per l'entrada del model.
    """
    categorical_inputs = {}
    
    # Mapeig de camps a tensors
    if 'edat' in case:
        categorical_inputs['edat'] = torch.tensor([int(case['edat'])], device=DEVICE)
    if 'genere' in case:
        categorical_inputs['genere'] = torch.tensor([int(case['genere'])], device=DEVICE)
    if 'c_alta' in case:
        categorical_inputs['c_alta'] = torch.tensor([int(case['c_alta'])], device=DEVICE)
    if 'periode' in case:
        categorical_inputs['periode'] = torch.tensor([int(case['periode'])], device=DEVICE)
    if 'servei' in case:
        categorical_inputs['servei'] = torch.tensor([int(case['servei'])], device=DEVICE)
    
    return categorical_inputs

def save_model(model, optimizer, scheduler, mlb, tokenizer, model_path, model_dir):
    """
    Guarda el model i els seus components.
    """
    try:
        # Crear directori si no existeix
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar el model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'mlb': mlb,
            'tokenizer': tokenizer
        }, model_path)
        
        logger.info(f"Model guardat correctament a {model_path}")
        
    except Exception as e:
        logger.error(f"Error en guardar el model: {str(e)}")
        raise

def load_model(model_path):
    """
    Carrega el model i els seus components.
    """
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            mlb = checkpoint['mlb']
            tokenizer = checkpoint['tokenizer']
            logger.info(f"Model carregat correctament des de {model_path}")
            return True
        else:
            logger.warning(f"No s'ha trobat el model a {model_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error en carregar el model: {str(e)}")
        return False

def ordenar_codis_per_prediccio(predictions: List[Tuple[str, float, float]]) -> List[str]:
    """
    Ordena els codis segons les prediccions d'ordre del model.
    
    Args:
        predictions: Llista de tuples (codi, probabilitat_codi, probabilitat_ordre)
        
    Returns:
        List[str]: Llista ordenada de codis segons les prediccions d'ordre
    """
    # Filtrar només els codis amb probabilitat superior al threshold
    filtered_predictions = [(code, code_prob, order_prob) 
                          for code, code_prob, order_prob in predictions 
                          if code_prob > 0.5]  # Threshold per la presència del codi
    
    # Ordenar per probabilitat d'ordre (de major a menor)
    sorted_predictions = sorted(filtered_predictions, 
                              key=lambda x: x[2], 
                              reverse=True)
    
    # Retornar només els codis en l'ordre predit
    return [code for code, _, _ in sorted_predictions]

def predict_case(case: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Realitza una predicció multi-label per un cas.
    
    Args:
        case (dict): Diccionari amb les dades del cas
        threshold (float): Llindar de probabilitat per considerar una etiqueta com a positiva
        
    Returns:
        dict: Diccionari amb el cas i les prediccions ordenades
    """
    try:
        # Preparar inputs
        text_input = prepare_text_inputs(case)
        tokenized = tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=4096
        ).to(DEVICE)
        
        # Afegir variables categòriques
        categorical_inputs = prepare_categorical_inputs(case)
        inputs = {**tokenized, **categorical_inputs}
        
        # Fer predicció
        model.eval()
        with torch.no_grad():
            code_probs, order_probs = model(inputs)
            
        # Obtenir prediccions amb probabilitats de codis i ordre
        predictions = []
        code_probs = code_probs[0].cpu().numpy()
        order_probs = order_probs[0].cpu().numpy()
        
        for i, (code_prob, order_prob) in enumerate(zip(code_probs, order_probs)):
            if code_prob > threshold:
                code = mlb.classes_[i]
                predictions.append((code, float(code_prob), float(order_prob)))
        
        # Ordenar els codis segons les prediccions d'ordre
        ordered_codes = ordenar_codis_per_prediccio(predictions)
        
        # Retornar resultats amb identificador del cas
        return {
            'cas': case['cas'],
            'prediccions': ordered_codes
        }
        
    except Exception as e:
        logger.error(f"Error en fer predicció: {str(e)}")
        raise
