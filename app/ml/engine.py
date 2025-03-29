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
from app.ml.text_processor import ClinicalTextProcessor
from app.ml.utils import (
    get_device, prepare_text_inputs, prepare_categorical_inputs,
    update_predicted_codes, is_code_in_training_history,
    save_model, calculate_class_weights, calculate_kendall_tau
)
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
import pickle

# Suprimir advertencies específiques
warnings.filterwarnings("ignore", message="NVIDIA GeForce RTX.*is not compatible with the current PyTorch installation")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir mensajes de padding de transformers
logging.getLogger("transformers").setLevel(logging.WARNING)

# Constants
DEVICE = get_device()
logger.info(f"Utilitzant dispositiu: {DEVICE}")

# Definir la funció de pèrdua
loss_fn = nn.BCEWithLogitsLoss()

# Variables globals
global mlb, tokenizer, model, optimizer, scheduler, predicted_codes_set

# Inicialitzar el conjunt de codis predits
predicted_codes_set = set()

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
    colspecs=[(17, 32)],
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

# Verificar si necessitem descarregar el model
config_path = os.path.join(LOCAL_LONGFORMER_PATH, "config.json")
if not os.path.exists(LOCAL_LONGFORMER_PATH) or not os.path.exists(config_path):
    logger.info(f"Model no trobat localment. Descarregant des de Hugging Face ({MODEL_ID})...")
    
    if os.path.exists(LOCAL_LONGFORMER_PATH):
        shutil.rmtree(LOCAL_LONGFORMER_PATH)
    
    # Descarregar amb el nou mètode recomanat
    tokenizer = LongformerTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=False,
        force_download=True
    )
    base_model = LongformerModel.from_pretrained(
        MODEL_ID,
        local_files_only=False,
        force_download=True
    )
    
    os.makedirs(LOCAL_LONGFORMER_PATH, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_LONGFORMER_PATH)
    base_model.save_pretrained(LOCAL_LONGFORMER_PATH)
    logger.info("Model descarregat i desat localment")
else:
    logger.info("Model ja existeix localment, no es descarregarà")

# Carregar el model des del directori local
logger.info("Carregant model des del directori local...")
try:
    if not os.path.exists(LOCAL_LONGFORMER_PATH):
        raise FileNotFoundError(f"No s'ha trobat el directori del model a {LOCAL_LONGFORMER_PATH}")
    
    # Carregar sempre des del directori local
    tokenizer = LongformerTokenizer.from_pretrained(
        LOCAL_LONGFORMER_PATH,
        local_files_only=True,
        force_download=False
    )
    base_model = LongformerModel.from_pretrained(
        LOCAL_LONGFORMER_PATH,
        local_files_only=True,
        force_download=False
    )
    model = CIE10Classifier(num_labels=NUM_LABELS)

    if os.path.exists(MODEL_PATH):
        logger.info(f"Carregant model entrenat des de {MODEL_PATH}")
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            
            if not isinstance(checkpoint, dict):
                raise ValueError("El checkpoint no té el format correcte")
            
            if 'model_state_dict' not in checkpoint:
                raise ValueError("El checkpoint no conté model_state_dict")
            model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9, last_epoch=-1)
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scheduler.last_epoch = -1
            
            if 'mlb_state' in checkpoint:
                mlb_state = checkpoint['mlb_state']
                mlb = MultiLabelBinarizer(sparse_output=mlb_state['sparse_output'])
                mlb.classes_ = np.array(mlb_state['classes_'])
                mlb.fit([[]])
            if 'tokenizer' in checkpoint:
                tokenizer = checkpoint['tokenizer']
                
        except Exception as e:
            logger.error(f"Error al cargar el checkpoint: {str(e)}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
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
        except Exception as e:
            logger.error(f"Error al guardar el model inicial: {str(e)}")
            raise

    model = model.to(DEVICE)
    model.train()

    if model is None or tokenizer is None:
        raise RuntimeError("El model o el tokenizer no s'han inicialitzat correctament")

    required_layers = ['text_encoder', 'code_classifier', 'order_classifier', 'categorical_embeddings']
    for layer in required_layers:
        if not hasattr(model, layer):
            raise RuntimeError(f"El model no té la capa {layer}")

    if not hasattr(model, 'num_labels'):
        raise RuntimeError("El model no té l'atribut num_labels")
    if model.num_labels != NUM_LABELS:
        raise RuntimeError(f"El model té {model.num_labels} etiquetes però es necessiten {NUM_LABELS}")

    logger.info("Model i tokenizer inicialitzats correctament")
    
except Exception as e:
    logger.error(f"Error inicialitzant el model i el tokenizer: {str(e)}")
    raise

# Inicialitzar el processador de text
text_processor = ClinicalTextProcessor()

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
            if is_code_in_training_history(code, mlb, predicted_codes_set) and prob > threshold:
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

    def _prepare_text(self, case: dict) -> str:
        """
        Prepara el text del cas clínic per al model.
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
            
            logger.info(f"Text preparat: {combined_text[:200]}...")
            return combined_text
            
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
            # Preparar inputs utilitzant el processador de text
            processed_case = text_processor.process_clinical_case(validation_data)
            
            # Construir el text combinat
            text_parts = []
            
            if processed_case.get('motiuingres'):
                text_parts.append(f"Motiu d'ingrés: {processed_case['motiuingres']}")
            
            if processed_case.get('malaltiaactual'):
                text_parts.append(f"Malaltia actual: {processed_case['malaltiaactual']}")
            
            if processed_case.get('exploracio'):
                text_parts.append(f"Exploració: {processed_case['exploracio']}")
            
            if processed_case.get('provescomplementariesing'):
                text_parts.append(f"Proves complementàries ingress: {processed_case['provescomplementariesing']}")
            
            if processed_case.get('provescomplementaries'):
                text_parts.append(f"Proves complementàries: {processed_case['provescomplementaries']}")
            
            if processed_case.get('evolucio'):
                text_parts.append(f"Evolució: {processed_case['evolucio']}")
            
            if processed_case.get('antecedents'):
                text_parts.append(f"Antecedents: {processed_case['antecedents']}")
            
            if processed_case.get('cursclinic'):
                text_parts.append(f"Curs clínic: {processed_case['cursclinic']}")
            
            # Unir totes les parts amb separador
            text = " | ".join(text_parts)
            
            # Tokenitzar el text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True
            ).to(DEVICE)
            
            # Preparar etiquetes
            dx_revisat = validation_data.get('dx_revisat', '')
            codes = [code.strip() for code in dx_revisat.split('|') if code.strip()]
            # Limitar a 15 códigos si hay más
            codes = codes[:15]
            
            # Ajustar el batch size para que coincida con el número de códigos (máximo 15)
            batch_size = min(len(codes), 15)
            inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}
            
            # Preparar variables categòriques
            categorical_inputs = prepare_categorical_inputs(validation_data)
            # Ajustar el batch size de las variables categóricas (máximo 15)
            categorical_inputs = {k: v.repeat(batch_size, 1) for k, v in categorical_inputs.items()}
            inputs.update(categorical_inputs)
            
            # Preparar etiquetes
            label_vector = mlb.transform([codes])
            labels = torch.from_numpy(np.array(label_vector)).float().to(DEVICE)
            # Ajustar el batch size de las etiquetas para que coincida con el de los inputs
            labels = labels.repeat(batch_size, 1)
            
            # Validar
            try:
                # Realitzar forward pass
                code_logits, order_logits = self.model(inputs)
                
                # Calcular pèrdua per classificació de codis
                code_loss = loss_fn(code_logits, labels)
                
                # Calcular prediccions de codis
                code_predictions = torch.sigmoid(code_logits) > 0.5
                
                # Obtenir codis predits amb probabilitat > 80%
                predicted_codes = []
                probabilities = torch.sigmoid(code_logits)[0]
                for i, prob in enumerate(probabilities):
                    code = mlb.classes_[i]
                    if code in predicted_codes_set and prob > 0.8:
                        predicted_codes.append(code)
                
                # Obtenir els 15 codis amb més probabilitat
                available_indices = [i for i, code in enumerate(mlb.classes_) if code in predicted_codes_set]
                if available_indices:
                    available_probs = probabilities[available_indices]
                    top_15_indices = torch.topk(available_probs, min(15, len(available_probs))).indices
                    top_15_codes = []
                    for idx in top_15_indices:
                        code = mlb.classes_[available_indices[idx]]
                        prob = probabilities[available_indices[idx]].item()
                        top_15_codes.append(f"{code} ({prob:.2%})")
                else:
                    top_15_codes = []
                
                # Calcular mètriques per classificació de codis
                true_positives = ((code_predictions == 1) & (labels == 1)).sum().item()
                false_positives = ((code_predictions == 1) & (labels == 0)).sum().item()
                false_negatives = ((code_predictions == 0) & (labels == 1)).sum().item()
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                total_labels = labels.numel()
                correct_preds = (code_predictions == labels).sum().item()
                accuracy = correct_preds / total_labels if total_labels > 0 else 0

                # Calcular mètriques per l'order_classifier
                order_predictions = torch.argmax(order_logits, dim=1)
                order_accuracy = (order_predictions == torch.arange(len(codes), device=DEVICE)).float().mean().item()
                
                # Calcular la pèrdua per l'order_classifier
                order_loss = torch.nn.CrossEntropyLoss()(order_logits, torch.arange(len(codes), device=DEVICE))
                
                # Calcular la distància de Kendall-Tau entre l'ordre predit i l'ordre real
                predicted_order = order_predictions.cpu().numpy()
                true_order = np.arange(len(codes))
                kendall_tau = calculate_kendall_tau(predicted_order, true_order)

                # Log compacte amb totes les mètriques
                logger.info(f"Cas {validation_data.get('cas', 'N/A')} - "
                          f"Code Loss: {code_loss.item():.4f} | "
                          f"Order Loss: {order_loss.item():.4f} | "
                          f"Prec: {precision:.4f} | Rec: {recall:.4f} | "
                          f"F1: {f1:.4f} | Acc: {accuracy:.4f} | "
                          f"Order Acc: {order_accuracy:.4f} | "
                          f"Kendall-Tau: {kendall_tau:.4f}")
                
                # Mostrar codis reals (fins a 15)
                logger.info("=== CODIS REALS ===")
                for i, code in enumerate(codes[:15]):
                    logger.info(f"  {i+1}. {code}")
                
                # Obtenir codis predits amb probabilitat > 90%
                predicted_codes = []
                probabilities = torch.sigmoid(code_logits)[0]
                for i, prob in enumerate(probabilities):
                    code = mlb.classes_[i]
                    if prob > 0.9:  # Mostrar todos los códigos con prob > 90%
                        predicted_codes.append((code, prob.item()))
                
                # Mostrar codis predits amb probabilitat > 90%
                if predicted_codes:
                    logger.info("=== CODIS PREDITS (probabilitat > 90%) ===")
                    for i, (code, prob) in enumerate(predicted_codes[:15]):
                        logger.info(f"  {i+1}. {code} ({prob:.1%})")
                else:
                    logger.info("=== NO HI HA CODIS PREDITS AMB PROBABILITAT > 90% ===")
                
                # Mostrar els 5 codis amb més probabilitat (solo de los entrenados)
                logger.info("=== TOP 5 CODIS MÉS PROBABLES (ENTRENATS) ===")
                # Obtener solo los códigos entrenados y sus probabilidades
                all_codes = []
                probabilities = torch.sigmoid(code_logits)[0]
                for i, prob in enumerate(probabilities):
                    code = mlb.classes_[i]
                    if code in predicted_codes_set:  # Solo incluir códigos entrenados
                        all_codes.append((code, prob.item()))
                
                # Ordenar por probabilidad y mostrar los top 5
                all_codes.sort(key=lambda x: x[1], reverse=True)
                for i, (code, prob) in enumerate(all_codes[:5]):
                    logger.info(f"  {i+1}. {code} ({prob:.1%})")
                
                # Log de l'ordre predit vs real
                logger.info("=== ORDRE DELS CODIS ===")
                logger.info("Ordre real (diagnòstic principal → secundaris):")
                for i, code in enumerate(codes):
                    logger.info(f"  {i+1}. {code}")
                
                logger.info("Ordre predit pel model:")
                # Asegurar que no se repitan códigos en el orden predicho
                seen_codes = set()
                for i, idx in enumerate(predicted_order):
                    code = codes[idx]
                    if code not in seen_codes:
                        seen_codes.add(code)
                        logger.info(f"  {len(seen_codes)}. {code}")
                
                # Calcular i mostrar la distància de Kendall-Tau
                if len(codes) > 1:
                    logger.info(f"Distància de Kendall-Tau: {kendall_tau:.4f}")
                    if kendall_tau > 0.8:
                        logger.info("  → L'ordre predit és molt similar al real")
                    elif kendall_tau > 0.5:
                        logger.info("  → L'ordre predit és moderadament similar al real")
                    else:
                        logger.info("  → L'ordre predit difereix significativament del real")
                else:
                    logger.info("No es pot calcular la distància de Kendall-Tau amb un sol codi")
                
                return {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'code_loss': code_loss.item(),
                    'order_loss': order_loss.item(),
                    'order_accuracy': order_accuracy,
                    'kendall_tau': kendall_tau,
                    'top_15_codes': top_15_codes,
                    'predicted_order': predicted_order.tolist(),
                    'true_order': true_order.tolist()
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
            
            # Actualizar el conjunt de códigos predichos con los nuevos códigos
            try:
                # Cargar el conjunto actual de códigos predichos
                predicted_codes_path = os.path.join(MODEL_DIR, 'predicted_codes.pkl')
                if os.path.exists(predicted_codes_path):
                    with open(predicted_codes_path, 'rb') as f:
                        predicted_codes_set = pickle.load(f)
                    logger.info(f"Conjunt de codis predits carregat. Total actual: {len(predicted_codes_set)}")
                else:
                    predicted_codes_set = set()
                    logger.info("No s'ha trobat el fitxer de codis predits. Creant nou conjunt.")
                
                # Contar cuántos códigos nuevos se van a añadir
                codis_nous = 0
                for code in codes:
                    if code not in predicted_codes_set:
                        predicted_codes_set.add(code)
                        codis_nous += 1
                
                # Solo guardar si hay códigos nuevos
                if codis_nous > 0:
                    with open(predicted_codes_path, 'wb') as f:
                        pickle.dump(predicted_codes_set, f)
                    logger.info(f"Conjunt de codis predits actualitzat. {codis_nous} codis nous afegits. Total: {len(predicted_codes_set)}")
                else:
                    logger.info("No hi ha codis nous per afegir al conjunt de codis predits")
                
            except Exception as e:
                logger.error(f"Error actualitzant el conjunt de codis predits: {str(e)}")
                raise
            
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
                
                # Ajustar el batch size para que coincida con el número de códigos (máximo 15)
                batch_size = min(len(codes), 15)
                inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}
                
            except Exception as e:
                logger.error(f"Error en la tokenització: {str(e)}")
                raise
            
            # Preparar les variables categòriques
            try:
                categorical_inputs = prepare_categorical_inputs(data)
                # Ajustar el batch size de las variables categóricas (máximo 15)
                categorical_inputs = {k: v.repeat(batch_size, 1) for k, v in categorical_inputs.items()}
                inputs.update(categorical_inputs)
            except Exception as e:
                logger.error(f"Error preparant variables categòriques: {str(e)}")
                raise
            
            # Preparar les etiquetes amb els codis netejats
            try:
                # Limitar a 15 códigos si hay más
                codes = codes[:15]
                label_vector = mlb.transform([codes])
                labels = torch.from_numpy(np.array(label_vector)).float().to(DEVICE)
                # Ajustar el batch size de las etiquetas para que coincida con el de los inputs
                labels = labels.repeat(batch_size, 1)
                logger.info(f"Cas {data['cas']} - Codis CIE-10: {' | '.join(codes)}")
            except Exception as e:
                logger.error(f"Error preparant etiquetes: {str(e)}")
                raise
            
            # Calcular pesos per classe
            try:
                class_weights = calculate_class_weights(labels, NUM_LABELS)
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
                loss_fn = torch.nn.BCEWithLogitsLoss()
            
            # Inicialitzar early stopping
            best_loss = float('inf')
            patience = 5
            patience_counter = 0
            min_delta = 0.001
            
            # Entrenar per 20 èpoques o fins que es compleixi early stopping
            max_epochs = 20
            for epoch in range(max_epochs):
                try:
                    # Realitzar forward pass
                    code_logits, order_logits = self.model(inputs)
                    
                    # Calcular pèrdua total (classificació + ordre)
                    code_loss = loss_fn(code_logits, labels)
                    order_loss = torch.nn.CrossEntropyLoss()(order_logits, torch.arange(len(codes), device=DEVICE))
                    total_loss = code_loss + order_loss
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Actualitzar pesos
                    optimizer.step()
                    scheduler.step()
                    
                    # Netejar gradients
                    optimizer.zero_grad()
                    
                    # Mostrar informació resumida
                    current_lr = optimizer.param_groups[0]['lr']
                    improvement_indicator = "[✓]" if total_loss.item() < best_loss else "[ ]"
                    logger.info(f"Època {epoch+1}/{max_epochs} - "
                              f"LR: {current_lr:.8f} - "
                              f"Total: {total_loss.item():.6f} - "
                              f"Code: {code_loss.item():.6f} - "
                              f"Order: {order_loss.item():.6f} {improvement_indicator}")
                    
                    # Verificar early stopping
                    if total_loss.item() < best_loss - min_delta:
                        best_loss = total_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping activat després de {epoch+1} èpoques sense millora")
                            break
                
                except Exception as e:
                    logger.error(f"Error durant l'entrenament de l'època {epoch+1}: {str(e)}")
                    raise
            
            # Guardar el model final
            save_model(
                self.model,
                optimizer,
                scheduler,
                mlb,
                self.tokenizer,
                MODEL_PATH,
                MODEL_DIR,
                predicted_codes_set
            )
            logger.info("Model guardat al final de l'entrenament")
            
            # Guardar el conjunt de códigos predichos
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
