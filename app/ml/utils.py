import torch
import logging
from typing import List, Dict, Any
import gc

def truncate_field(text: str, max_length: int = 5000) -> str:
    """
    Trunca el text a una longitud màxima especificada.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text[:max_length]

def is_relevant(code: str) -> bool:
    """
    Verifica si un codi és rellevant (comença amb '0D').
    """
    return code.startswith("0D")

def freeze_bert_layers(model: torch.nn.Module, freeze: bool = True, num_unfrozen_layers: int = 0):
    """
    Congela o descongela les capes del model.

    Args:
        model (torch.nn.Module): El model que conté les capes.
        freeze (bool): Si és True, les capes es congelaran; si és False, es descongelaran.
        num_unfrozen_layers (int): Nombre de capes superiors que no es congelaran.
        Per defecte, es congelen totes les capes.
    """
    logger = logging.getLogger(__name__)

    # Verificar si el model té l'atribut 'encoder' i 'layer'
    if not hasattr(model, 'encoder') or not hasattr(model.encoder, 'layer'):
        logger.warning("El model no té 'encoder.layer'; no es poden comptar les capes.")
        total_layers = 0
    else:
        # Comptar el nombre total de capes a l'encoder
        total_layers = len(model.encoder.layer)
        logger.info(f"Total de capes a l'encoder: {total_layers}")

    if num_unfrozen_layers > total_layers:
        raise ValueError(f"num_unfrozen_layers={num_unfrozen_layers} excedeix el nombre total de capes {total_layers}")

    num_frozen_layers = total_layers - num_unfrozen_layers

    for layer_num, layer in enumerate(model.encoder.layer):
        if layer_num < num_frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False  # Congelar la capa
            logger.debug(f"Capes 0-{layer_num} congelades.")
        else:
            for param in layer.parameters():
                param.requires_grad = not freeze  # Descongelar la capa si freeze=False
            logger.debug(f"Capes {layer_num} i superiors {'descongelades' if not freeze else 'congelades'}.")

def clear_gpu_memory():
    """
    Neteja la memòria de la GPU.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memòria GPU netejada.")

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
        device: Dispositiu on executar (CPU/GPU)
        max_length: Longitud màxima de tokenització
        
    Returns:
        Diccionari amb els tensors processats
    """
    try:
        # Utilitzar mixed precision per estalviar memòria
        with torch.cuda.amp.autocast():
            # Tokenitzar el batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length
            ).to(device)
            
            # Activar gradient checkpointing si està disponible
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            return inputs
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning("Memòria insuficient. Netejant memòria i reintentant...")
            clear_gpu_memory()
            raise
        else:
            raise

def optimize_model_memory(model: torch.nn.Module):
    """
    Aplica optimitzacions de memòria al model.
    
    Args:
        model: Model a optimitzar
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing activat per estalviar memòria.")
    
    # Desactivar atenció completa si està disponible
    if hasattr(model, 'config'):
        model.config.use_cache = False
        logger.info("Cache d'atenció desactivada per estalviar memòria.")

def get_memory_usage():
    """
    Retorna l'ús actual de memòria de la GPU.
    
    Returns:
        Diccionari amb l'ús de memòria en MB
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved
        }
    return {"allocated_mb": 0, "reserved_mb": 0}

def validate_input_data(text: str, codes: List[str]) -> bool:
    if not text or len(text.strip()) == 0:
        return False
    if not codes or not all(is_relevant(code) for code in codes):
        return False
    return True

def process_cie10_codes(code_string: str) -> list:
    """
    Processa una cadena de codis CIE-10 separats per '|'.
    
    Args:
        code_string (str): Cadena de codis separats per '|'
        
    Returns:
y        list: Llista de codis vàlids (no buits)
        
    Exemple:
        >>> process_cie10_codes("J441|J101|J988|J9600|H532|M797||||||||||")
        ['J441', 'J101', 'J988', 'J9600', 'H532', 'M797']
    """
    if not code_string:
        return []
    
    # Dividir per '|' i eliminar espais en blanc
    codes = [code.strip() for code in code_string.split('|')]
    
    # Filtrar codis buits
    valid_codes = [code for code in codes if code]
    
    return valid_codes

# Implementar early stopping
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

def process_clinical_course(text: str, max_length: int = 100000) -> str:
    """
    Processa el curs clínic per extreure la informació més rellevant.
    
    Args:
        text: Text del curs clínic
        max_length: Longitud màxima permesa
        
    Returns:
        Text processat i truncat
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    # Si el text és més curt que max_length, retornar-lo directament
    if len(text) <= max_length:
        return text
        
    # Dividir el text en seccions
    sections = text.split("\n\n")
    
    # Seleccionar les seccions més rellevants
    # 1. Primera secció (context inicial)
    # 2. Última secció (conclusió/resultat)
    # 3. Seccions del mig amb paraules clau
    relevant_sections = [sections[0]]  # Sempre incloure la primera secció
    
    # Paraules clau que indiquen seccions importants
    keywords = [
        "diagnostic", "diagnòstic", "tractament", "complicacions",
        "evolució", "conclusió", "resultat", "pronòstic"
    ]
    
    # Afegir seccions que contenen paraules clau
    for section in sections[1:-1]:  # Excloure primera i última secció
        if any(keyword in section.lower() for keyword in keywords):
            relevant_sections.append(section)
    
    # Sempre incloure l'última secció
    if len(sections) > 1:
        relevant_sections.append(sections[-1])
    
    # Unir les seccions i truncar si encara és massa llarg
    processed_text = "\n\n".join(relevant_sections)
    return processed_text[:max_length]
