# app/ml/engine.py

"""
Mòdul principal per la gestió del model i el seu entrenament.
Proporciona funcionalitats per carregar, entrenar i fer prediccions amb el model.
"""

import logging
import torch
import os
import shutil
import warnings
from transformers import LongformerTokenizer, LongformerModel
from app.ml.model import CIE10Classifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
import pickle

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", message="NVIDIA GeForce RTX.*is not compatible with the current PyTorch installation")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
def get_device():
    """
    Determina el dispositiu a utilitzar, manejando casos de incompatibilidad CUDA.
    """
    if torch.cuda.is_available():
        try:
            # Verificar compatibilidad CUDA
            cuda_capability = torch.cuda.get_device_capability()
            logger.info(f"Capacitat CUDA detectada: {cuda_capability}")
            
            # Si la capacidad es mayor que la soportada, usar CPU
            if cuda_capability[0] > 9:  # sm_90 es la máxima soportada actualmente
                logger.warning("La GPU no és compatible amb la versió actual de PyTorch. Utilitzant CPU.")
                return torch.device("cpu")
            
            return torch.device("cuda")
        except Exception as e:
            logger.warning(f"Error detectant compatibilitat CUDA: {str(e)}. Utilitzant CPU.")
            return torch.device("cpu")
    else:
        logger.info("CUDA no disponible. Utilitzant CPU.")
        return torch.device("cpu")

DEVICE = get_device()
logger.info(f"Utilitzant dispositiu: {DEVICE}")

# Variables globales
global mlb, tokenizer, model, optimizer, scheduler, predicted_codes_set

# Inicializar el conjunto de códigos predichos
predicted_codes_set = set()

# Añadir MultiLabelBinarizer a la lista de globals seguros
torch.serialization.add_safe_globals(['sklearn.preprocessing._label.MultiLabelBinarizer'])

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

# Llegir els codis CIM10MC
if not os.path.exists(CIM10MC_PATH):
    raise FileNotFoundError(f"No s'ha trobat el fitxer CIM10MC a {CIM10MC_PATH}")

logger.info(f"Llegint codis CIM10MC des de {CIM10MC_PATH}")
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
try:
    # Verificar si el directorio del modelo existe
    if not os.path.exists(LOCAL_LONGFORMER_PATH):
        raise FileNotFoundError(f"No s'ha trobat el directori del model a {LOCAL_LONGFORMER_PATH}")
    
    # Cargar el tokenizer
    logger.info("Carregant tokenizer...")
    tokenizer = LongformerTokenizer.from_pretrained(LOCAL_LONGFORMER_PATH, local_files_only=True)
    logger.info("Tokenizer carregat correctament")

    # Crear el modelo base
    logger.info("Creant model base...")
    base_model = LongformerModel.from_pretrained(LOCAL_LONGFORMER_PATH, local_files_only=True)
    logger.info("Model base creat correctament")

    # Crear el classificador
    logger.info("Creant classificador...")
    model = CIE10Classifier(num_labels=NUM_LABELS)
    logger.info("Classificador creat correctament")

    # Cargar o crear el modelo
    if os.path.exists(MODEL_PATH):
        logger.info(f"Carregant model entrenat des de {MODEL_PATH}")
        try:
            # Cargar el checkpoint con weights_only=True
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            
            # Verificar que el checkpoint tiene la estructura correcta
            if not isinstance(checkpoint, dict):
                raise ValueError("El checkpoint no té el format correcte")
            
            # Cargar el estado del modelo
            if 'model_state_dict' not in checkpoint:
                raise ValueError("El checkpoint no conté model_state_dict")
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Checkpoint carregat correctament")
            
            # Crear optimizador y scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9, last_epoch=-1)
            
            # Cargar estados del optimizador y scheduler si existen
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # Forzar que el scheduler empiece desde el principio
                scheduler.last_epoch = -1
            
            # Cargar MultiLabelBinarizer y tokenizer si existen
            if 'mlb_state' in checkpoint:
                mlb_state = checkpoint['mlb_state']
                mlb = MultiLabelBinarizer(sparse_output=mlb_state['sparse_output'])
                mlb.classes_ = np.array(mlb_state['classes_'])
                mlb.fit([[]])
            if 'tokenizer' in checkpoint:
                tokenizer = checkpoint['tokenizer']
                
        except Exception as e:
            logger.error(f"Error al cargar el checkpoint: {str(e)}")
            logger.error("Inicialitzant model nou...")
            # Si hay error, inicializar como nuevo modelo
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    else:
        logger.info("Inicialitzant model nou...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        # Guardar el modelo inicial
        logger.info("Guardant model inicial...")
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'mlb_state': {
                    'classes_': mlb.classes_.tolist(),
                    'sparse_output': mlb.sparse_output
                },
                'tokenizer': tokenizer
            }, MODEL_PATH)
            logger.info(f"Model inicial guardat a {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error al guardar el modelo inicial: {str(e)}")
            raise

    # Asegurar que el modelo está en el dispositivo correcto
    logger.info(f"Movent model a {DEVICE}...")
    model = model.to(DEVICE)
    model.train()  # Asegurar que el modelo está en modo entrenamiento
    logger.info("Model mogut correctament")

    # Verificar que el modelo y el tokenizer están disponibles
    if model is None or tokenizer is None:
        raise RuntimeError("El model o el tokenizer no s'han inicialitzat correctament")

    # Verificar que el modelo tiene todas las capas necesarias
    logger.info("Verificant capes del model...")
    required_layers = ['text_encoder', 'code_classifier', 'order_classifier', 'categorical_embeddings']
    for layer in required_layers:
        if not hasattr(model, layer):
            raise RuntimeError(f"El model no té la capa {layer}")
    logger.info("Totes les capes necessàries estan presents")

    # Verificar que el modelo tiene el número correcto de etiquetas
    if not hasattr(model, 'num_labels'):
        raise RuntimeError("El model no té l'atribut num_labels")
    if model.num_labels != NUM_LABELS:
        raise RuntimeError(f"El model té {model.num_labels} etiquetes però es necessiten {NUM_LABELS}")
    logger.info(f"Nombre d'etiquetes correcte: {model.num_labels}")

    logger.info("Model i tokenizer inicialitzats correctament")
    
except Exception as e:
    logger.error(f"Error inicialitzant el model i el tokenizer: {str(e)}")
    logger.error(f"Tipus d'error: {type(e).__name__}")
    logger.error(f"Detalls de l'error: {str(e)}")
    raise

def prepare_text_inputs(case: Dict[str, Any]) -> str:
    """
    Prepara els camps de text per la tokenització.
    
    Args:
        case: Diccionari amb les dades del cas clínic
        
    Returns:
        Text preparat per tokenitzar
    """
    try:
        # Definir els camps de text en ordre de prioritat
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
        
        # Netejar i filtrar els camps buits
        cleaned_fields = []
        for field in text_fields:
            if field is not None:
                field_str = str(field).strip()
                if field_str:
                    cleaned_fields.append(field_str)
        
        # Si no hi ha camps vàlids, retornar un text per defecte
        if not cleaned_fields:
            logger.warning("No s'han trobat camps de text vàlids")
            return "No hi ha dades disponibles per aquest cas"
        
        # Unir els camps amb el separador
        text = ' [SEP] '.join(cleaned_fields)
        
        # Verificar que el text no està buit
        if not text.strip():
            logger.warning("El text preparat està buit")
            return "No hi ha dades disponibles per aquest cas"
        
        # Verificar que el text té contingut significatiu
        if text == " [SEP] ".join([""] * len(text_fields)):
            logger.warning("El text només conté valors per defecte")
            return "No hi ha dades disponibles per aquest cas"
        
        # Log del text preparat
        logger.info(f"Text preparat per tokenitzar: {text[:200]}{'...' if len(text) > 200 else ''}")
        
        # Verificar que el text té la longitud mínima necessària
        if len(text.split()) < 2:
            logger.warning("El text és massa curt")
            return "No hi ha dades suficients per aquest cas"
        
        # Verificar que el text conté informació clínica
        clinical_keywords = ['síntoma', 'diagnóstico', 'tratamiento', 'evolución', 'exploración', 'prueba', 'test', 'resultado']
        has_clinical_info = any(keyword in text.lower() for keyword in clinical_keywords)
        if not has_clinical_info:
            logger.warning("El text no conté informació clínica significativa")
            return "No hi ha dades clíniques suficients per aquest cas"
        
        return text
        
    except Exception as e:
        logger.error(f"Error preparant els inputs de text: {str(e)}")
        logger.error(f"Tipus d'error: {type(e).__name__}")
        logger.error(f"Detalls de l'error: {str(e)}")
        raise

def prepare_categorical_inputs(case: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Prepara les variables categòriques pel model.
    
    Args:
        case: Diccionari amb les dades del cas
        
    Returns:
        Dict amb els tensors de les variables categòriques
    """
    try:
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
                    value = case[field]
                    # Convertir a int si es string
                    if isinstance(value, str):
                        value = int(float(value))
                    else:
                        value = int(value)
                    categorical_fields[field] = value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Valor invàlid per {field}: {case[field]}, usant valor per defecte. Error: {str(e)}")
        
        # Convertir a tensors amb la dimensió correcta i asegurar valores válidos
        result = {}
        for field, value in categorical_fields.items():
            # Asegurar que el valor es no negativo
            value = max(0, value)
            
            # Crear tensor y mover al dispositivo correcto
            tensor = torch.tensor([[value]], device=DEVICE, dtype=torch.long)
            result[field] = tensor
            
            logger.debug(f"Campo {field}: valor={value}, tensor shape={tensor.shape}, device={tensor.device}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error en prepare_categorical_inputs: {str(e)}")
        logger.error(f"Caso recibido: {case}")
        raise

def update_predicted_codes(codes: List[str]) -> None:
    """
    Actualiza el conjunto de códigos predichos con nuevos códigos.
    
    Args:
        codes: Lista de códigos a añadir
    """
    global predicted_codes_set
    for code in codes:
        if code in mlb.classes_:
            predicted_codes_set.add(code)
    logger.info(f"Conjunt de codis predits actualitzat. Total: {len(predicted_codes_set)}")

def is_code_in_training_history(code: str) -> bool:
    """
    Verifica si un codi ha aparegut en l'entrenament.
    
    Args:
        code: Codi CIE-10 a verificar
        
    Returns:
        bool: True si el codi ha aparegut en l'entrenament, False altrament
    """
    try:
        # Verificar si el codi està en el MultiLabelBinarizer
        if code not in mlb.classes_:
            logger.warning(f"El codi {code} no està en el catàleg de codis")
            return False
            
        # Verificar si el codi ha aparegut en l'entrenament
        code_index = mlb.classes_.tolist().index(code)
        if code_index >= mlb.classes_.shape[0]:
            logger.warning(f"El codi {code} no ha aparegut en l'entrenament")
            return False
            
        # Verificar si el codi ha estat predit en algun moment
        if code not in predicted_codes_set:
            logger.warning(f"El codi {code} no ha estat predit en cap moment")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error verificant el codi {code}: {str(e)}")
        return False

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
    """
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar tokenizer
        tokenizer.save_pretrained(LOCAL_LONGFORMER_PATH)
        logger.info("Tokenizer guardat correctament")
        
        # Guardar mlb
        mlb_path = os.path.join(MODEL_DIR, 'mlb.pkl')
        with open(mlb_path, 'wb') as f:
            pickle.dump(mlb, f)
        logger.info("MultiLabelBinarizer guardat correctament")
        
        # Guardar el conjunt de codis predits
        predicted_codes_path = os.path.join(MODEL_DIR, 'predicted_codes.pkl')
        with open(predicted_codes_path, 'wb') as f:
            pickle.dump(predicted_codes_set, f)
        logger.info("Conjunt de codis predits guardat correctament")
        
        # Guardar pesos i estat
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
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
    """
    try:
        if not os.path.exists(model_path):
            logger.warning(f"No s'ha trobat el model a {model_path}")
            return False
            
        # Cargar tokenizer
        global tokenizer
        tokenizer = LongformerTokenizer.from_pretrained(LOCAL_LONGFORMER_PATH)
        logger.info("Tokenizer carregat correctament")
        
        # Cargar mlb
        mlb_path = os.path.join(MODEL_DIR, 'mlb.pkl')
        if os.path.exists(mlb_path):
            global mlb
            with open(mlb_path, 'rb') as f:
                mlb = pickle.load(f)
            logger.info("MultiLabelBinarizer carregat correctament")
        
        # Cargar el conjunt de codis predits
        predicted_codes_path = os.path.join(MODEL_DIR, 'predicted_codes.pkl')
        if os.path.exists(predicted_codes_path):
            global predicted_codes_set
            with open(predicted_codes_path, 'rb') as f:
                predicted_codes_set = pickle.load(f)
            logger.info(f"Conjunt de codis predits carregat correctament ({len(predicted_codes_set)} codis)")
        
        # Cargar pesos i estat
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Cargar el estado del modelo
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state dict carregat correctament")
        
        # Cargar el estado del optimizador
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state dict carregat correctament")
        
        # Cargar el estado del scheduler
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scheduler.last_epoch = -1  # Forzar que empiece desde el principio
            logger.info("Scheduler state dict carregat correctament")
        
        # Asegurar que el modelo está en el dispositivo correcto
        model = model.to(DEVICE)
        logger.info(f"Model mogut a {DEVICE}")
        
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
    top_k: int = 15,
    threshold: float = 0.9
) -> Dict[str, Any]:
    """
    Realitza prediccions per un cas clínic.
    
    Args:
        case: Diccionari amb les dades del cas clínic
        top_k: Nombre de codis més probables a retornar
        threshold: Umbral mínim de probabilitat (90% per defecte)
        
    Returns:
        Dict amb les prediccions ordenades per probabilitat
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
            code_logits, order_logits = model(inputs)
            
        # Processar prediccions
        probabilities = torch.sigmoid(code_logits)[0].cpu().numpy()
        
        # Filtrar prediccions per threshold i codis vàlids
        valid_predictions = []
        for idx, prob in enumerate(probabilities):
            code = mlb.classes_[idx]
            # Verificar si el codi ha aparegut en l'entrenament i té probabilitat suficient
            if is_code_in_training_history(code) and prob > threshold:
                valid_predictions.append((idx, prob))
        
        # Ordenar per probabilitat i prendre els top_k
        valid_predictions.sort(key=lambda x: x[1], reverse=True)
        top_k_predictions = valid_predictions[:top_k]
        
        # Crear llista de prediccions amb format
        predictions = []
        for idx, prob in top_k_predictions:
            code = mlb.classes_[idx]
            predictions.append({
                'code': code,
                'probability': float(prob)
            })
        
        # Log de prediccions
        logger.info(f"Cas {case['cas']} - Prediccions amb probabilitat > {threshold:.1%}:")
        for pred in predictions:
            logger.info(f"  - {pred['code']}: {pred['probability']:.1%}")
        
        return {
            'cas': case['cas'],
            'prediccions': predictions
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

def calculate_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calcula els pesos per classe basats en la freqüència.
    
    Args:
        labels: Tensor amb les etiquetes (0s i 1s)
        
    Returns:
        Tensor amb els pesos per classe
    """
    try:
        # Calcular freqüències
        class_counts = labels.sum(dim=0)
        total_samples = labels.shape[0]
        
        # Calcular pesos inversos a la freqüència
        weights = total_samples / (class_counts + 1)  # +1 per evitar divisió per zero
        
        # Normalitzar pesos
        weights = weights / weights.mean()
        
        # Asegurar que los pesos están en el dispositivo correcto
        weights = weights.to(DEVICE)
        
        return weights
        
    except Exception as e:
        logger.error(f"Error calculant pesos per classe: {str(e)}")
        # En caso de error, usar pesos uniformes
        return torch.ones(NUM_LABELS).to(DEVICE)

class ModelEngine:
    def __init__(self):
        """
        Inicialitza el motor del model amb el tokenizer i el model Longformer.
        """
        try:
            # Verificar si el model i el tokenizer estan disponibles globalment
            if 'model' not in globals() or 'tokenizer' not in globals():
                logger.error("El model o el tokenizer no estan inicialitzats globalment")
                raise RuntimeError("El model o el tokenizer no estan inicialitzats globalment")
            
            # Carregar el tokenizer i el model
            self.tokenizer = globals()['tokenizer']
            self.model = globals()['model']
            
            # Verificar que el model i el tokenizer són vàlids
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("El model o el tokenizer són None")
            
            # Verificar que el model té els mètodes necessaris
            if not hasattr(self.model, 'forward'):
                raise RuntimeError("El model no té el mètode forward")
            
            # Verificar que el tokenizer té els mètodes necessaris
            if not hasattr(self.tokenizer, '__call__'):
                raise RuntimeError("El tokenizer no té el mètode __call__")
            
            # Verificar que el model està al dispositiu correcte
            if next(self.model.parameters()).device != DEVICE:
                self.model = self.model.to(DEVICE)
                logger.info(f"Model mogut a {DEVICE}")
            
            # Verificar que el model té les capes necessàries
            if not hasattr(self.model, 'text_encoder'):
                raise RuntimeError("El model no té la capa text_encoder")

            if not hasattr(self.model, 'order_classifier'):
                raise RuntimeError("El model no té la capa order_classifier")

            if not hasattr(self.model, 'code_classifier'):
                raise RuntimeError("El model no té la capa code_classifier")

            if not hasattr(self.model, 'categorical_embeddings'):
                raise RuntimeError("El model no té la capa categorical_embeddings")
            
            # Verificar que el model té els paràmetres necessaris
            if not hasattr(self.model, 'num_labels'):
                raise RuntimeError("El model no té l'atribut num_labels")
            
            if self.model.num_labels != NUM_LABELS:
                raise RuntimeError(f"El model té {self.model.num_labels} etiquetes però es necessiten {NUM_LABELS}")
            
            # Configurar el model per entrenament
            self.model.train()
            
            logger.info("Model i tokenizer carregats correctament")
            
        except Exception as e:
            logger.error(f"Error inicialitzant el model: {str(e)}")
            raise

    def _prepare_text(self, data: dict) -> str:
        """
        Prepara el text combinant tots els camps rellevants.
        """
        try:
            # Definir els camps a incloure en l'ordre desitjat
            fields = [
                ("Motiu d'ingrés", data.get('motiuingres')),
                ("Malaltia actual", data.get('malaltiaactual')),
                ("Exploració", data.get('exploracio')),
                ("Proves complementàries ingress", data.get('provescomplementariesing')),
                ("Proves complementàries", data.get('provescomplementaries')),
                ("Evolució", data.get('evolucio')),
                ("Antecedents", data.get('antecedents')),
                ("Curs clínic", data.get('cursclinic')),
                ("Edat", data.get('edat')),
                ("Gènere", data.get('genere')),
                ("Període", data.get('periode')),
                ("Servei", data.get('servei'))
            ]
            
            # Convertir tots els valors a string i filtrar els buits
            text_parts = []
            for label, value in fields:
                # Convertir a string i netejar
                str_value = str(value).strip() if value is not None else ""
                # Si el valor està buit, usar "No especificat"
                if not str_value:
                    str_value = "No especificat"
                text_parts.append(f"{label}: {str_value}")
            
            # Si no hi ha cap camp, retornar un text per defecte
            if not text_parts:
                logger.warning("No s'han trobat camps de text vàlids")
                return "No hi ha dades disponibles per aquest cas"
            
            # Unir tots els camps amb separador
            text = " | ".join(text_parts)
            
            # Verificar que el text no està buit
            if not text.strip():
                logger.warning("El text preparat està buit")
                return "No hi ha dades disponibles per aquest cas"
            
            # Log del text preparat
            logger.info(f"Text preparat: {text[:200]}{'...' if len(text) > 200 else ''}")
            
            # Verificar que el text té contingut significatiu
            if text == " | ".join([f"{label}: No especificat" for label, _ in fields]):
                logger.warning("El text només conté valors per defecte")
                return "No hi ha dades disponibles per aquest cas"
            
            return text
            
        except Exception as e:
            logger.error(f"Error preparant el text: {str(e)}")
            logger.error(f"Tipus d'error: {type(e).__name__}")
            logger.error(f"Detalls de l'error: {str(e)}")
            raise

    def validate_model(self, validation_data: dict) -> Dict[str, float]:
        """
        Valida el model amb un conjunt de dades.
        """
        try:
            # Verificar que el model i el tokenizer estan disponibles
            if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
                raise RuntimeError("El model o el tokenizer no estan inicialitzats")
            
            # Verificar que el model i el tokenizer són vàlids
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("El model o el tokenizer són None")
            
            # Verificar que el model té els mètodes necessaris
            if not hasattr(self.model, 'forward'):
                raise RuntimeError("El model no té el mètode forward")
            
            # Verificar que el tokenizer té els mètodes necessaris
            if not hasattr(self.tokenizer, '__call__'):
                raise RuntimeError("El tokenizer no té el mètode __call__")
            
            # Verificar que el model està al dispositiu correcte
            if next(self.model.parameters()).device != DEVICE:
                self.model = self.model.to(DEVICE)
            
            # Verificar que el model té les capes necessàries
            required_layers = ['text_encoder', 'code_classifier', 'order_classifier', 'categorical_embeddings']
            for layer in required_layers:
                if not hasattr(self.model, layer):
                    raise RuntimeError(f"El model no té la capa {layer}")
            
            # Verificar que el model té els paràmetres necessaris
            if not hasattr(self.model, 'num_labels'):
                raise RuntimeError("El model no té l'atribut num_labels")
            if self.model.num_labels != NUM_LABELS:
                raise RuntimeError(f"El model té {self.model.num_labels} etiquetes però es necessiten {NUM_LABELS}")
            
            # Verificar que el model està en mode validació
            self.model.eval()
            
            # Netejar i validar el codi CIE-10
            dx_revisat = validation_data.get('dx_revisat')
            if not dx_revisat or not isinstance(dx_revisat, str):
                raise ValueError("El cas ha de tenir un codi CIE-10 revisat")
                
            # Logs de depuració
            logger.debug(f"Codi {validation_data.get('cas', 'N/A')} - dx_revisat original: {dx_revisat}")
            
            # Netejar els codis CIE-10 (eliminar separadors buits)
            codes = [code.strip() for code in dx_revisat.split('|') if code.strip()]
            logger.debug(f"Codi {validation_data.get('cas', 'N/A')} - codis parsejats: {codes}")
            
            if not codes:
                raise ValueError("No s'han trobat codis CIE-10 vàlids")
            
            # Verificar que el MultiLabelBinarizer està disponible
            if 'mlb' not in globals():
                raise RuntimeError("El MultiLabelBinarizer no està inicialitzat")
            
            # Preparar el text combinant tots els camps
            text = self._prepare_text(validation_data)
            
            # Tokenitzar el text
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                    padding=True
                ).to(DEVICE)
            except Exception as e:
                logger.error(f"Error en la tokenització: {str(e)}")
                raise
            
            # Preparar les variables categòriques
            try:
                categorical_inputs = prepare_categorical_inputs(validation_data)
                inputs.update(categorical_inputs)
            except Exception as e:
                logger.error(f"Error preparant variables categòriques: {str(e)}")
                raise
            
            # Preparar les etiquetes amb els codis netejats
            try:
                label_vector = mlb.transform([codes])
                labels = torch.from_numpy(np.array(label_vector)).float().to(DEVICE)
                logger.info(f"Etiquetes preparades correctament. Shape: {labels.shape}")
            except Exception as e:
                logger.error(f"Error preparant etiquetes: {str(e)}")
                raise
            
            # Calcular pesos per classe
            try:
                class_weights = calculate_class_weights(labels)
                logger.info(f"Pesos per classe calculats correctament. Shape: {class_weights.shape}")
            except Exception as e:
                logger.error(f"Error calculant pesos per classe: {str(e)}")
                class_weights = torch.ones(NUM_LABELS).to(DEVICE)
            
            # Configurar pèrdua amb pesos dinàmics
            try:
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
                logger.info("Funció de pèrdua configurada correctament")
            except Exception as e:
                logger.error(f"Error configurant funció de pèrdua: {str(e)}")
                # En caso de error, usar pérdida sin pesos
                loss_fn = torch.nn.BCEWithLogitsLoss()
            
            # Validar
            try:
                # Realitzar forward pass
                code_logits, order_logits = self.model(inputs)
                
                # Calcular pèrdua
                loss = loss_fn(code_logits, labels)
                
                # Calcular prediccions
                predictions = torch.sigmoid(code_logits) > 0.5
                
                # Obtenir codis predits amb probabilitat > 80%
                predicted_codes = []
                probabilities = torch.sigmoid(code_logits)[0]
                for i, prob in enumerate(probabilities):
                    code = mlb.classes_[i]
                    # Solo incluir códigos que están en predicted_codes_set
                    if code in predicted_codes_set and prob > 0.8:
                        predicted_codes.append(code)
                
                # Obtenir els 5 codis amb més probabilitat (solo de los disponibles)
                available_indices = [i for i, code in enumerate(mlb.classes_) if code in predicted_codes_set]
                if available_indices:
                    available_probs = probabilities[available_indices]
                    top_5_indices = torch.topk(available_probs, min(5, len(available_probs))).indices
                    top_5_codes = []
                    for idx in top_5_indices:
                        code = mlb.classes_[available_indices[idx]]
                        prob = probabilities[available_indices[idx]].item()
                        top_5_codes.append(f"{code} ({prob:.2%})")
                else:
                    top_5_codes = []
                
                # Mostrar informació resumida
                logger.info(f"Cas {validation_data.get('cas', 'N/A')}")
                logger.info(f"Loss: {loss.item():.4f}")
                
                # Mostrar probabilitat del codi I69 si está disponible
                try:
                    if 'I69' in predicted_codes_set:
                        i69_index = mlb.classes_.tolist().index('I69')
                        i69_prob = probabilities[i69_index].item()
                        logger.info(f"Probabilitat del codi I69: {i69_prob:.2%}")
                except ValueError:
                    logger.info("El codi I69 no està disponible")
                
                # Mostrar prediccions amb probabilitat > 80%
                if predicted_codes:
                    logger.info(f"Prediccions (>80%): {', '.join(predicted_codes)}")
                else:
                    logger.info("No hi ha prediccions amb probabilitat > 80%")
                
                # Mostrar top 5 prediccions
                if top_5_codes:
                    logger.info("Top 5 prediccions:")
                    for code in top_5_codes:
                        logger.info(f"  - {code}")
                else:
                    logger.info("No hi ha prediccions disponibles")
                
                # Calcular mètriques
                true_positives = ((predictions == 1) & (labels == 1)).sum().item()
                false_positives = ((predictions == 1) & (labels == 0)).sum().item()
                false_negatives = ((predictions == 0) & (labels == 1)).sum().item()
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                total_labels = labels.numel()
                correct_preds = (predictions == labels).sum().item()
                accuracy = correct_preds / total_labels if total_labels > 0 else 0

                logger.debug(f"Etiquetes esperades (top activades): {[mlb.classes_[i] for i, val in enumerate(labels[0]) if val == 1.0]}")
                logger.debug(f"Prediccions activades (>0.7): {[mlb.classes_[i] for i, val in enumerate(torch.sigmoid(code_logits[0])) if val > 0.7 and mlb.classes_[i] in predicted_codes_set]}")
                logger.info(f"Precisió: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f} - Accuracy: {accuracy:.4f}")
                
                return {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'loss': loss.item()
                }
                
            except Exception as e:
                logger.error(f"Error durant la validació: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error en la validació del cas {validation_data.get('cas', 'N/A')}: {str(e)}")
            raise

    async def train_incremental(self, data: dict):
        """
        Entrena el model amb un nou cas.
        """
        try:
            # Verificar que el model i el tokenizer estan disponibles
            if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
                raise RuntimeError("El model o el tokenizer no estan inicialitzats")
            
            # Verificar que el model i el tokenizer són vàlids
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("El model o el tokenizer són None")
            
            # Verificar que el model té els mètodes necessaris
            if not hasattr(self.model, 'forward'):
                raise RuntimeError("El model no té el mètode forward")
            
            # Verificar que el tokenizer té els mètodes necessaris
            if not hasattr(self.tokenizer, '__call__'):
                raise RuntimeError("El tokenizer no té el mètode __call__")
            
            # Verificar que el model està al dispositiu correcte
            if next(self.model.parameters()).device != DEVICE:
                self.model = self.model.to(DEVICE)
            
            # Verificar que el model té les capes necessàries
            required_layers = ['text_encoder', 'code_classifier', 'order_classifier', 'categorical_embeddings']
            for layer in required_layers:
                if not hasattr(self.model, layer):
                    raise RuntimeError(f"El model no té la capa {layer}")
            
            # Verificar que el model té els paràmetres necessaris
            if not hasattr(self.model, 'num_labels'):
                raise RuntimeError("El model no té l'atribut num_labels")
            if self.model.num_labels != NUM_LABELS:
                raise RuntimeError(f"El model té {self.model.num_labels} etiquetes però es necessiten {NUM_LABELS}")
            
            # Verificar que el model està en mode entrenament
            if not self.model.training:
                self.model.train()
            
            # Reiniciar l'optimitzador i el scheduler per cada cas
            global optimizer, scheduler
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9, last_epoch=-1)
            logger.info(f"Learning rate reiniciat a 2e-5 per el cas {data['cas']}")
            
            # Netejar i validar el codi CIE-10
            dx_revisat = data.get('dx_revisat')
            if not dx_revisat or not isinstance(dx_revisat, str):
                raise ValueError("El cas ha de tenir un codi CIE-10 revisat")
                
            # Logs de depuració
            logger.debug(f"Codi {data['cas']} - dx_revisat original: {dx_revisat}")
            
            # Netejar els codis CIE-10 (eliminar separadors buits)
            codes = [code.strip() for code in dx_revisat.split('|') if code.strip()]
            logger.debug(f"Codi {data['cas']} - codis parsejats: {codes}")
            
            if not codes:
                raise ValueError("No s'han trobat codis CIE-10 vàlids")
            
            # Verificar que el MultiLabelBinarizer està disponible
            if 'mlb' not in globals():
                raise RuntimeError("El MultiLabelBinarizer no està inicialitzat")
            
            # Actualizar el conjunto de códigos predichos con los nuevos códigos
            update_predicted_codes(codes)
            
            # Preparar el text combinant tots els camps
            text = self._prepare_text(data)
            
            # Tokenitzar el text
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                    padding=True
                ).to(DEVICE)
            except Exception as e:
                logger.error(f"Error en la tokenització: {str(e)}")
                raise
            
            # Preparar les variables categòriques
            try:
                categorical_inputs = prepare_categorical_inputs(data)
                inputs.update(categorical_inputs)
            except Exception as e:
                logger.error(f"Error preparant variables categòriques: {str(e)}")
                raise
            
            # Preparar les etiquetes amb els codis netejats
            try:
                label_vector = mlb.transform([codes])
                labels = torch.from_numpy(np.array(label_vector)).float().to(DEVICE)
                logger.info(f"Etiquetes preparades correctament. Shape: {labels.shape}")
            except Exception as e:
                logger.error(f"Error preparant etiquetes: {str(e)}")
                raise
            
            # Calcular pesos per classe
            try:
                class_weights = calculate_class_weights(labels)
                logger.info(f"Pesos per classe calculats correctament. Shape: {class_weights.shape}")
            except Exception as e:
                logger.error(f"Error calculant pesos per classe: {str(e)}")
                class_weights = torch.ones(NUM_LABELS).to(DEVICE)
            
            # Configurar pèrdua amb pesos dinàmics
            try:
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
                logger.info("Funció de pèrdua configurada correctament")
            except Exception as e:
                logger.error(f"Error configurant funció de pèrdua: {str(e)}")
                # En caso de error, usar pérdida sin pesos
                loss_fn = torch.nn.BCEWithLogitsLoss()
            
            # Entrenar per 3 èpoques
            epochs = 20
            for epoch in range(epochs):
                try:
                    # Realitzar forward pass
                    code_logits, order_logits = self.model(inputs)
                    
                    # Calcular pèrdua
                    loss = loss_fn(code_logits, labels)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Actualitzar pesos
                    optimizer.step()
                    scheduler.step()
                    
                    # Netejar gradients
                    optimizer.zero_grad()
                    
                    # Mostrar informació resumida
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Cas {data['cas']} - Època {epoch+1}/{epochs} - Learning Rate: {current_lr:.8f} - Loss: {loss.item():.6f}")
                    
                except Exception as e:
                    logger.error(f"Error durant l'entrenament de l'època {epoch+1}: {str(e)}")
                    raise
            
            # Guardar model després de totes les èpoques
            save_model(
                self.model,
                optimizer,
                scheduler,
                mlb,
                self.tokenizer,
                MODEL_PATH,
                MODEL_DIR
            )
            
            # Guardar el conjunto de códigos predichos
            predicted_codes_path = os.path.join(MODEL_DIR, 'predicted_codes.pkl')
            with open(predicted_codes_path, 'wb') as f:
                pickle.dump(predicted_codes_set, f)
            logger.info(f"Conjunt de codis predits guardat correctament. Total: {len(predicted_codes_set)}")
            
        except Exception as e:
            logger.error(f"Error en l'entrenament del cas {data['cas']}: {str(e)}")
            raise

    def save_model(self):
        """
        Guarda el model entrenat.
        """
        try:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            LOCAL_LONGFORMER_PATH = os.path.join(BASE_DIR, "models", "clinical-longformer")
            
            self.model.save_pretrained(LOCAL_LONGFORMER_PATH)
            self.tokenizer.save_pretrained(LOCAL_LONGFORMER_PATH)
            
            logger.info("Model guardat correctament")
            
        except Exception as e:
            logger.error(f"Error guardant el model: {str(e)}")
            raise

    async def train_with_validation(
        self,
        train_data: List[dict],
        validation_data: List[dict],
        test_data: List[dict],
        epochs: int = 5,
        batch_size: int = 8,
        early_stopping_patience: int = 3
    ) -> Dict[str, Any]:
        """
        Entrena el model amb els tres conjunts de dades.
        """
        try:
            from app.ml.utils import EarlyStopping
            
            # Inicialitzar early stopping
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=0.001
            )
            
            # Historial de mètriques
            history = {
                'train_loss': [],
                'train_metrics': [],
                'val_loss': [],
                'val_metrics': [],
                'test_metrics': []
            }
            
            # Entrenament per èpoques
            for epoch in range(epochs):
                logger.info(f"\n{'='*50}")
                logger.info(f"ÈPOCA {epoch + 1}/{epochs}")
                logger.info(f"{'='*50}")
                
                # Entrenament amb dades T
                self.model.train()
                train_loss = 0
                train_metrics = {
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'accuracy': 0
                }
                
                # Processar en batches
                for i in range(0, len(train_data), batch_size):
                    batch = train_data[i:i + batch_size]
                    batch_loss = 0
                    
                    for case in batch:
                        # Preparar dades
                        text = self._prepare_text(case)
                        inputs = self.tokenizer(
                            text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=4096,
                            padding=True
                        ).to(DEVICE)
                        
                        categorical_inputs = prepare_categorical_inputs(case)
                        inputs.update(categorical_inputs)
                        
                        # Preparar etiquetes
                        dx_revisat = case.get('dx_revisat', '')
                        codes = [code.strip() for code in dx_revisat.split('|') if code.strip()]
                        label_vector = mlb.transform([codes])
                        labels = torch.from_numpy(np.array(label_vector)).float().to(DEVICE)
                        
                        # Forward pass
                        code_probs, _ = self.model(inputs)
                        loss = loss_fn(code_probs, labels)
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        batch_loss += loss.item()
                        
                        # Calcular mètriques
                        predictions = torch.sigmoid(code_probs) > 0.5
                        train_metrics['accuracy'] += (predictions == labels).sum().item() / labels.sum().item()
                    
                    train_loss += batch_loss / len(batch)
                
                # Promig de mètriques d'entrenament
                train_loss /= (len(train_data) // batch_size)
                train_metrics['accuracy'] /= len(train_data)
                
                # Validació amb dades V
                self.model.eval()
                val_loss = 0
                val_metrics = {
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'accuracy': 0
                }
                
                with torch.no_grad():
                    for case in validation_data:
                        # Preparar dades
                        text = self._prepare_text(case)
                        inputs = self.tokenizer(
                            text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=4096,
                            padding=True
                        ).to(DEVICE)
                        
                        categorical_inputs = prepare_categorical_inputs(case)
                        inputs.update(categorical_inputs)
                        
                        # Preparar etiquetes
                        dx_revisat = case.get('dx_revisat', '')
                        codes = [code.strip() for code in dx_revisat.split('|') if code.strip()]
                        label_vector = mlb.transform([codes])
                        labels = torch.from_numpy(np.array(label_vector)).float().to(DEVICE)
                        
                        # Forward pass
                        code_probs, _ = self.model(inputs)
                        loss = loss_fn(code_probs, labels)
                        
                        val_loss += loss.item()
                        
                        # Calcular mètriques
                        predictions = torch.sigmoid(code_probs) > 0.5
                        val_metrics['accuracy'] += (predictions == labels).sum().item() / labels.sum().item()
                
                # Promig de mètriques de validació
                val_loss /= len(validation_data)
                val_metrics['accuracy'] /= len(validation_data)
                
                # Guardar mètriques
                history['train_loss'].append(train_loss)
                history['train_metrics'].append(train_metrics)
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)
                
                # Log de mètriques
                logger.info(f"\nMÈTRIQUES DE L'ÈPOCA {epoch + 1}:")
                logger.info(f"Pèrdua d'entrenament (T): {train_loss:.4f}")
                logger.info(f"Accuracy d'entrenament (T): {train_metrics['accuracy']:.4f}")
                logger.info(f"Pèrdua de validació (V): {val_loss:.4f}")
                logger.info(f"Accuracy de validació (V): {val_metrics['accuracy']:.4f}")
                
                # Early stopping
                if early_stopping(val_loss):
                    logger.info("Early stopping activat")
                    break
                
                # Actualitzar learning rate
                scheduler.step()
            
            # Evaluació final amb dades E
            logger.info("\nEVALUACIÓ FINAL AMB EL CONJUNT DE TEST (E)")
            test_metrics = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'accuracy': 0
            }
            
            self.model.eval()
            with torch.no_grad():
                for case in test_data:
                    metrics = self.validate_model(case)
                    for key in test_metrics:
                        test_metrics[key] += metrics[key]
            
            # Promig de mètriques de test
            for key in test_metrics:
                test_metrics[key] /= len(test_data)
            
            history['test_metrics'] = test_metrics
            
            # Log de mètriques finals
            logger.info("\nMÈTRIQUES FINALS (E):")
            logger.info(f"Precisió: {test_metrics['precision']:.4f}")
            logger.info(f"Recall: {test_metrics['recall']:.4f}")
            logger.info(f"F1-score: {test_metrics['f1']:.4f}")
            logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
            
            # Guardar el model final
            self.save_model()
            
            return history
            
        except Exception as e:
            logger.error(f"Error en l'entrenament amb validació: {str(e)}")
            raise
