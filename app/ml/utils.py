import torch
import logging
from typing import List, Dict, Any, Optional, Union
import gc
import re
from bs4 import BeautifulSoup

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

def prepare_categorical_inputs(case: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Prepara les variables categòriques pel model.
    """
    try:
        categorical_fields = {
            'edat': 0,
            'genere': 0,
            'c_alta': 0,
            'periode': 0,
            'servei': 0
        }
        
        for field in categorical_fields:
            if field in case:
                try:
                    value = case[field]
                    if isinstance(value, str):
                        value = int(float(value))
                    else:
                        value = int(value)
                    categorical_fields[field] = value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Valor invàlid per {field}: {case[field]}, usant valor per defecte. Error: {str(e)}")
        
        result = {}
        for field, value in categorical_fields.items():
            value = max(0, value)
            tensor = torch.tensor([[value]], device=DEVICE, dtype=torch.long)
            result[field] = tensor
            
        return result
        
    except Exception as e:
        logger.error(f"Error en prepare_categorical_inputs: {str(e)}")
        raise
