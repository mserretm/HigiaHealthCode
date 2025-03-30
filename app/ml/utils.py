import torch
import logging
from typing import List, Dict, Any, Optional, Union
import gc
import re
from bs4 import BeautifulSoup
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os
import pickle
from transformers import LongformerTokenizer
from app.ml.text_processor import ClinicalTextProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
FIELD_LIMITS = {
    "malaltiaactual": 5000,
    "cursclinic": 100000,
    "motiuingres": 2000,
    "antecedents": 3000,
    "exploracio": 3000,
    "proves": 3000,
    "tractament": 2000
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Utilitzant dispositiu: {DEVICE}")

def truncate_field(text: Union[str, Any], max_length: int = 5000) -> str:
    """
    Trunca el text a una longitud màxima especificada.
    
    Args:
        text: Text a truncar (pot ser qualsevol tipus)
        max_length: Longitud màxima permesa
        
    Returns:
        Text truncat a la longitud especificada
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text[:max_length]

def clean_html_text(text: Union[str, Any]) -> str:
    """
    Neteja el text HTML i el normalitza.
    
    Args:
        text: Text amb possibles tags HTML
        
    Returns:
        Text net i normalitzat
    
    Exemple:
        >>> clean_html_text("<p>Text amb <b>HTML</b></p>")
        'text amb html'
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    try:
        # Eliminar etiquetes HTML
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator=' ')
        
        # Normalitzar espais
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Convertir a minúscules i eliminar espais
        clean_text = clean_text.lower().strip()
        
        return clean_text
    except Exception as e:
        logger.warning(f"Error en netejar HTML: {str(e)}")
        return text.lower().strip()

def is_relevant(code: str) -> bool:
    """
    Verifica si un codi és rellevant (comença amb '0D').
    
    Args:
        code: Codi CIE-10 a verificar
        
    Returns:
        bool: True si el codi és rellevant
    """
    return bool(code and isinstance(code, str) and code.startswith("0D"))

def freeze_bert_layers(
    model: torch.nn.Module,
    freeze: bool = True,
    num_unfrozen_layers: int = 0
) -> None:
    """
    Congela o descongela les capes del model.
    """
    if not hasattr(model, 'encoder') or not hasattr(model.encoder, 'layer'):
        logger.warning("El model no té 'encoder.layer'; no es poden modificar les capes.")
        return

    total_layers = len(model.encoder.layer)

    if num_unfrozen_layers > total_layers:
        raise ValueError(
            f"num_unfrozen_layers={num_unfrozen_layers} excedeix el total de capes {total_layers}"
        )

    num_frozen_layers = total_layers - num_unfrozen_layers

    for layer_num, layer in enumerate(model.encoder.layer):
        for param in layer.parameters():
            param.requires_grad = not freeze if layer_num >= num_frozen_layers else False

def clear_gpu_memory() -> None:
    """
    Neteja la memòria de la GPU i registra l'ús de memòria.
    """
    if torch.cuda.is_available():
        initial_memory = get_memory_usage()
        torch.cuda.empty_cache()
        gc.collect()
        final_memory = get_memory_usage()
        
        logger.info(
            f"Memòria GPU netejada. "
            f"Abans: {initial_memory['allocated_mb']:.2f}MB, "
            f"Després: {final_memory['allocated_mb']:.2f}MB"
        )

def process_batch_with_memory_optimization(
    batch_texts: List[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int = 4096
) -> Dict[str, torch.Tensor]:
    """
    Processa un batch de textos amb optimitzacions de memòria.
    
    Args:
        batch_texts: Llista de textos a processar
        tokenizer: Tokenizer a utilitzar
        model: Model a utilitzar
        device: Dispositiu on executar
        max_length: Longitud màxima de tokenització
        
    Returns:
        Dict amb els tensors processats
        
    Raises:
        RuntimeError: Si hi ha problemes de memòria
    """
    try:
        with torch.cuda.amp.autocast():
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length
            ).to(device)
            
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            return inputs
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning("Memòria insuficient. Netejant i reintentant...")
            clear_gpu_memory()
        raise

def optimize_model_memory(model: torch.nn.Module) -> None:
    """
    Aplica optimitzacions de memòria al model.
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    if hasattr(model, 'config'):
        model.config.use_cache = False

def get_memory_usage() -> Dict[str, float]:
    """
    Retorna l'ús actual de memòria de la GPU.
    
    Returns:
        Dict amb l'ús de memòria en MB
    """
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2
        }
    return {"allocated_mb": 0, "reserved_mb": 0}

def validate_input_data(text: Optional[str], codes: Optional[List[str]]) -> bool:
    """
    Valida les dades d'entrada pel model.
    
    Args:
        text: Text a validar
        codes: Llista de codis a validar
        
    Returns:
        bool: True si les dades són vàlides
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        logger.warning("Text invàlid o buit")
        return False
        
    if not codes or not isinstance(codes, list) or not all(is_relevant(code) for code in codes):
        logger.warning("Codis invàlids o buits")
        return False
        
    return True

def process_cie10_codes(code_string: Optional[str]) -> List[str]:
    """
    Processa una cadena de codis CIE-10 separats per '|'.
    
    Args:
        code_string: Cadena de codis
        
    Returns:
        Llista de codis vàlids
        
    Exemple:
        >>> process_cie10_codes("J441|J101||J988")
        ['J441', 'J101', 'J988']
    """
    if not code_string or not isinstance(code_string, str):
        return []
    
    return [code.strip() for code in code_string.split('|') if code.strip()]

def process_clinical_course(text: Union[str, Any], max_length: int = 100000) -> str:
    """
    Processa el curs clínic per extreure la informació més rellevant.
    
    Args:
        text: Text del curs clínic
        max_length: Longitud màxima permesa
        
    Returns:
        Text processat i truncat
        
    Exemple:
        >>> text = "<p>Inici del cas</p>\\n\\nEvolució\\n\\nConclusió"
        >>> process_clinical_course(text, max_length=50)
        'inici del cas\\n\\nevolució\\n\\nconclusió'
    """
    # Netejar HTML i normalitzar
    text = clean_html_text(text)
        
    # Si el text net és més curt que max_length, retornar-lo
    if len(text) <= max_length:
        return text
    
    # Dividir en seccions
    sections = text.split("\n\n")
    relevant_sections = [sections[0]]  # Primera secció
    
    # Paraules clau per seccions importants
    keywords = [
        "diagnostic", "diagnòstic", "tractament", "complicacions",
        "evolució", "conclusió", "resultat", "pronòstic"
    ]
    
    # Afegir seccions amb paraules clau
    for section in sections[1:-1]:
        if any(keyword in section for keyword in keywords):
            relevant_sections.append(section)
    
    # Afegir última secció
    if len(sections) > 1:
        relevant_sections.append(sections[-1])
    
    # Unir i truncar si cal
    processed_text = "\n\n".join(relevant_sections)
    return processed_text[:max_length]

class EarlyStopping:
    """
    Implementa early stopping pel entrenament.
    
    Attributes:
        patience: Nombre d'èpoques a esperar
        min_delta: Canvi mínim considerat com a millora
        counter: Comptador d'èpoques sense millora
        best_loss: Millor pèrdua registrada
        early_stop: Indica si cal aturar l'entrenament
    """
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

def get_device():
    """
    Determina el dispositiu a utilitzar, manejando casos de incompatibilidad CUDA.
    """
    if torch.cuda.is_available():
        try:
            cuda_capability = torch.cuda.get_device_capability()
            logger.info(f"Capacitat CUDA detectada: {cuda_capability}")
            
            if cuda_capability[0] > 9:
                logger.warning("La GPU no és compatible amb la versió actual de PyTorch. Utilitzant CPU.")
                return torch.device("cpu")
            
            return torch.device("cuda")
        except Exception as e:
            logger.warning(f"Error detectant compatibilitat CUDA: {str(e)}. Utilitzant CPU.")
            return torch.device("cpu")
    else:
        logger.info("CUDA no disponible. Utilitzant CPU.")
        return torch.device("cpu")

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
            
            # Crear tensor y mover al dispositiu correcte
            tensor = torch.tensor([[value]], device=DEVICE, dtype=torch.long)
            result[field] = tensor
            
            logger.debug(f"Campo {field}: valor={value}, tensor shape={tensor.shape}, device={tensor.device}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error en prepare_categorical_inputs: {str(e)}")
        logger.error(f"Caso recibido: {case}")
        raise

def update_predicted_codes(codes: List[str], mlb: MultiLabelBinarizer, predicted_codes_set: set) -> None:
    """
    Actualiza el conjunt de codis predichos con nuevos códigos.
    
    Args:
        codes: Lista de códigos a añadir
        mlb: MultiLabelBinarizer
        predicted_codes_set: Conjunto de códigos predichos
    """
    for code in codes:
        if code in mlb.classes_:
            predicted_codes_set.add(code)
    logger.info(f"Conjunt de codis predits actualitzat. Total: {len(predicted_codes_set)}")

def is_code_in_training_history(code: str, mlb: MultiLabelBinarizer, predicted_codes_set: set) -> bool:
    """
    Verifica si un codi ha aparegut en l'entrenament.
    
    Args:
        code: Codi CIE-10 a verificar
        mlb: MultiLabelBinarizer
        predicted_codes_set: Conjunto de códigos predichos
        
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
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    mlb: MultiLabelBinarizer,
    tokenizer: LongformerTokenizer,
    model_path: str,
    model_dir: str,
    predicted_codes_set: set
) -> None:
    """
    Guarda el model i els seus components.
    """
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar tokenizer
        tokenizer.save_pretrained(os.path.join(model_dir, "clinical-longformer"))
        logger.info("Tokenizer guardat correctament")
        
        # Guardar mlb
        mlb_path = os.path.join(model_dir, 'mlb.pkl')
        with open(mlb_path, 'wb') as f:
            pickle.dump(mlb, f)
        logger.info("MultiLabelBinarizer guardat correctament")
        
        # Guardar el conjunt de codis predits
        predicted_codes_path = os.path.join(model_dir, 'predicted_codes.pkl')
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

def calculate_class_weights(labels: torch.Tensor, num_labels: int) -> torch.Tensor:
    """
    Calcula els pesos per classe basats en la freqüència.
    
    Args:
        labels: Tensor amb les etiquetes (0s i 1s)
        num_labels: Nombre total d'etiquetes
        
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
        
        # Asegurar que los pesos están en el dispositiu correcto
        weights = weights.to(DEVICE)
        
        return weights
        
    except Exception as e:
        logger.error(f"Error calculant pesos per classe: {str(e)}")
        # En caso de error, usar pesos uniformes
        return torch.ones(num_labels).to(DEVICE)

def calculate_kendall_tau(predicted_order: np.ndarray, true_order: np.ndarray) -> float:
    """
    Calcula la distància de Kendall-Tau entre dos ordenaments.
    
    Args:
        predicted_order: Array amb l'ordre predit
        true_order: Array amb l'ordre real
        
    Returns:
        float: Distància de Kendall-Tau normalitzada entre 0 i 1
    """
    try:
        n = len(predicted_order)
        if n <= 1:
            return 0.0
            
        # Crear matrius de comparació
        pred_matrix = np.zeros((n, n))
        true_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                pred_matrix[i,j] = predicted_order[i] < predicted_order[j]
                true_matrix[i,j] = true_order[i] < true_order[j]
        
        # Calcular discordàncies
        discordances = np.sum(pred_matrix != true_matrix)
        
        # Normalitzar per obtenir una mètrica entre 0 i 1
        max_discordances = n * (n-1) / 2
        kendall_tau = 1 - (discordances / max_discordances)
        
        return float(kendall_tau)
        
    except Exception as e:
        logger.error(f"Error calculant Kendall-Tau: {str(e)}")
        return 0.0

def prepare_text(case: dict, text_processor: ClinicalTextProcessor) -> str:
    """
    Prepara el text del cas clínic per al model.
    
    Args:
        case: Diccionari amb les dades del cas clínic
        text_processor: Instància del processador de text
        
    Returns:
        str: Text combinat del cas clínic
    """
    try:
        # Processar el cas clínic amb el processador de text
        processed_case = text_processor.process_clinical_case(case)
        
        # Construir el text combinat
        text_parts = []
        
        if processed_case.get('motiu_ingres'):
            text_parts.append(f"Motiu d'ingrés: {processed_case['motiu_ingres']}")
        
        if processed_case.get('malaltia_actual'):
            text_parts.append(f"Malaltia actual: {processed_case['malaltia_actual']}")
        
        if processed_case.get('exploracio'):
            text_parts.append(f"Exploració: {processed_case['exploracio']}")
        
        if processed_case.get('proves_complementaries_ingress'):
            text_parts.append(f"Proves complementàries ingress: {processed_case['proves_complementaries_ingress']}")
        
        if processed_case.get('proves_complementaries'):
            text_parts.append(f"Proves complementàries: {processed_case['proves_complementaries']}")
        
        if processed_case.get('evolucio_clinica'):
            text_parts.append(f"Evolució clínica: {processed_case['evolucio_clinica']}")
        
        if processed_case.get('curs_clinic'):
            text_parts.append(f"Curs clínic: {processed_case['curs_clinic']}")
        
        if processed_case.get('diagnostic_ingress'):
            text_parts.append(f"Diagnòstic ingress: {processed_case['diagnostic_ingress']}")
        
        if processed_case.get('diagnostic_alta'):
            text_parts.append(f"Diagnòstic alta: {processed_case['diagnostic_alta']}")
        
        if processed_case.get('tractament'):
            text_parts.append(f"Tractament: {processed_case['tractament']}")
        
        if processed_case.get('recomanacions_alta'):
            text_parts.append(f"Recomanacions alta: {processed_case['recomanacions_alta']}")
        
        # Unir totes les parts amb separador
        combined_text = " | ".join(text_parts)
        
        return combined_text
        
    except Exception as e:
        logger.error(f"Error preparant el text: {str(e)}")
        logger.error(f"Tipus d'error: {type(e).__name__}")
        logger.error(f"Detalls de l'error: {str(e)}")
        raise
