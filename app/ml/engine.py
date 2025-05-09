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
    save_model, calculate_class_weights, calculate_kendall_tau,
    prepare_text
)
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
import pickle
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.train import DeepLearningTrain
from app.models.evaluate import DeepLearningEvaluate

# Suprimir advertencies específiques
warnings.filterwarnings("ignore", message="NVIDIA GeForce RTX.*is not compatible with the current PyTorch installation")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
warnings.filterwarnings("ignore", message="Input ids are automatically padded.*")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir mensajes de padding de transformers
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# Constants
DEVICE = get_device()
logger.info(f"Utilitzant dispositiu: {DEVICE}")

# Definir la funció de pèrdua
loss_fn = nn.BCEWithLogitsLoss()

# Variables globals
global mlb, tokenizer, model, optimizer, scheduler

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

    def validate_model(self, validation_data: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """Valida el model amb un cas clínic."""
        try:
            # Registrar hora de inicio
            ts_inici = datetime.now()
            
            # Cargar el conjunto de códigos entrenados
            predicted_codes_path = os.path.join(MODEL_DIR, 'predicted_codes.pkl')
            logger.info(f"Intentant carregar el conjunt de codis entrenats des de: {predicted_codes_path}")
            
            if os.path.exists(predicted_codes_path):
                with open(predicted_codes_path, 'rb') as f:
                    predicted_codes_set = pickle.load(f)
                logger.info(f"Conjunt de codis entrenats carregat correctament. Total: {len(predicted_codes_set)}")
                logger.info(f"Codis disponibles per validació: {sorted(list(predicted_codes_set))}")
            else:
                logger.error(f"No s'ha trobat el fitxer de codis entrenats a: {predicted_codes_path}")
                raise FileNotFoundError("No s'ha trobat el fitxer de codis entrenats (predicted_codes.pkl)")
            
            # Preparar el text combinant tots els camps
            text = prepare_text(validation_data, text_processor)
            
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
                
                # Calcular mètriques de classificació
                code_predictions = torch.sigmoid(code_logits)[0]
                predicted_codes = []
                for i, prob in enumerate(code_predictions):
                    code = mlb.classes_[i]
                    if code in predicted_codes_set and prob > 0.5:
                        predicted_codes.append(code)
                
                # Crear vectors binaris per les prediccions i les etiquetes reals
                y_true = set(codes)
                y_pred = set(predicted_codes)
                
                # Calcular mètriques de classificació
                true_positives = len(y_true.intersection(y_pred))
                false_positives = len(y_pred - y_true)
                false_negatives = len(y_true - y_pred)
                
                # Calcular accuracy
                total_codes = len(y_true.union(y_pred))
                accuracy = true_positives / total_codes if total_codes > 0 else 0
                
                # Calcular precision, recall i F1
                precision = true_positives / len(y_pred) if y_pred else 0
                recall = true_positives / len(y_true) if y_true else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Calcular mètriques per l'order_classifier
                order_predictions = torch.argmax(order_logits, dim=1)
                
                # Eliminar repeticions en el orden predit
                seen_indices = set()
                unique_predicted_order = []
                for idx in order_predictions:
                    if idx.item() not in seen_indices and idx.item() < len(codes):
                        seen_indices.add(idx.item())
                        unique_predicted_order.append(idx.item())
                
                # Completar el orden si es necesario
                while len(unique_predicted_order) < len(codes):
                    for i in range(len(codes)):
                        if i not in seen_indices:
                            unique_predicted_order.append(i)
                            seen_indices.add(i)
                            break
                
                # Convertir a tensor
                predicted_order = torch.tensor(unique_predicted_order, device=DEVICE)
                
                # Calcular accuracy con el orden único
                order_accuracy = (predicted_order == torch.arange(len(codes), device=DEVICE)).float().mean().item()
                
                # Calcular la pèrdua per l'order_classifier
                order_loss = torch.nn.CrossEntropyLoss()(order_logits, predicted_order)
                
                # Calcular la distància de Kendall-Tau entre l'ordre predit i l'ordre real
                predicted_order_np = predicted_order.cpu().numpy()
                true_order = np.arange(len(codes))
                kendall_tau = calculate_kendall_tau(predicted_order_np, true_order)

                # Log compacte amb totes les mètriques
                logger.info(f"=== VALIDACIÓ CAS: {validation_data.get('cas', 'N/A')} ===")
                
                # Mostrar codis reals
                logger.info(f"→ Codis reals ({len(codes)}):")
                for i, code in enumerate(codes[:15]):
                    logger.info(f"   {i+1}. {code}")
                
                # Obtenir i mostrar codis predits amb probabilitat > 90%
                predicted_codes = []
                probabilities = torch.sigmoid(code_logits)[0]
                for i, prob in enumerate(probabilities):
                    code = mlb.classes_[i]
                    if code in predicted_codes_set and prob > 0.9:
                        predicted_codes.append((code, prob.item()))
                
                if predicted_codes:
                    logger.info("→ Codis predits (>90% confiança):")
                    for i, (code, prob) in enumerate(predicted_codes[:15]):
                        logger.info(f"   {i+1}. {code} ({prob:.1%})")
                else:
                    logger.info("→ No hi ha codis predits amb confiança >90%")
                
                # Mostrar els 5 codis amb més probabilitat (solo de los entrenados)
                logger.info("→ Top 5 codis més probables (entrenats):")
                all_codes = []
                probabilities = torch.sigmoid(code_logits)[0]
                for i, prob in enumerate(probabilities):
                    code = mlb.classes_[i]
                    if code in predicted_codes_set:
                        all_codes.append((code, prob.item()))
                
                all_codes.sort(key=lambda x: x[1], reverse=True)
                for i, (code, prob) in enumerate(all_codes[:5]):
                    logger.info(f"   {i+1}. {code} ({prob:.1%})")
                
                # Mostrar ordre real i predit
                logger.info("→ Ordre real:        " + " → ".join(codes))
                logger.info("→ Ordre predit:      " + " → ".join([codes[i] for i in unique_predicted_order]))
                
                # Mostrar Kendall-Tau amb interpretació
                if len(codes) > 1:
                    kendall_tau_msg = f"→ Kendall-Tau:       {kendall_tau:.2f} → "
                    if kendall_tau > 0.8:
                        kendall_tau_msg += "L'ordre predit és molt similar al real"
                    elif kendall_tau > 0.5:
                        kendall_tau_msg += "L'ordre predit és moderadament similar al real"
                    else:
                        kendall_tau_msg += "L'ordre predit difereix significativament del real"
                    logger.info(kendall_tau_msg)
                else:
                    logger.info("→ No es pot calcular la distància de Kendall-Tau amb un sol codi")
                
                # Obtenir els 15 codis amb més probabilitat
                available_indices = [i for i, code in enumerate(mlb.classes_) if code in predicted_codes_set]
                if available_indices:
                    available_probs = probabilities[available_indices]
                    top_15_indices = torch.topk(available_probs, min(15, len(available_probs))).indices
                    top_15_codes = []
                    top_15_probs = []
                    for idx in top_15_indices:
                        code = mlb.classes_[available_indices[idx]]
                        prob = probabilities[available_indices[idx]].item()
                        top_15_codes.append(code)
                        top_15_probs.append(prob)
                else:
                    top_15_codes = []
                    top_15_probs = []
                
                # Mostrar mètriques de classificació
                logger.info("→ Mètriques de Classificació:")
                logger.info(f"   • Accuracy:  {accuracy:.2f}")
                logger.info(f"   • Precision: {precision:.2f}")
                logger.info(f"   • Recall:    {recall:.2f}")
                logger.info(f"   • F1 Score:  {f1:.2f}")
                
                # Mostrar mètriques d'ordre
                logger.info("→ Mètriques d'Ordre:")
                logger.info(f"   • Order Accuracy: {order_accuracy:.2f}")
                logger.info(f"   • Kendall-Tau:    {kendall_tau:.2f}")
                
                # Mostrar pèrdues
                logger.info("→ Pèrdues:")
                logger.info(f"   • Code Loss:  {code_loss.item():.4f}")
                logger.info(f"   • Order Loss: {order_loss.item():.4f}")
                
                # Registrar hora de fin
                ts_final = datetime.now()
                
                # Guardar resultados de evaluación en la base de datos
                evaluate_record = DeepLearningEvaluate(
                    experimento_id=f"eval_{ts_inici.strftime('%Y%m%d_%H%M%S')}",
                    caso_id=validation_data.get('cas', 'N/A'),
                    
                    # Códigos y orden
                    codis_reals=codes,
                    codis_predits_top15=top_15_codes,
                    probs_predits_top15=top_15_probs,
                    codis_predits_confianza_top15=[code for code, prob in predicted_codes],
                    ordre_real=codes,
                    ordre_predit=[codes[i] for i in unique_predicted_order],
                    
                    # Métricas de clasificación
                    accuracy=accuracy,
                    precisi=precision,
                    recall=recall,
                    f1_score=f1,
                    
                    # Métricas de orden
                    order_accuracy=order_accuracy,
                    kendall_tau=kendall_tau,
                    
                    # Pérdidas
                    code_loss=code_loss.item(),
                    order_loss=order_loss.item(),
                    
                    # Contadores
                    num_codis_reals=len(codes),
                    num_codis_predits_top15=len(top_15_codes),
                    num_codis_predits_confianza_top15=len(predicted_codes),
                    
                    # Metainformación
                    ver_modelo="1.0.0",
                    set_validacion="test",
                    ts_inici=ts_inici,
                    ts_final=ts_final
                )
                
                logger.info(f"Guardant resultats de validació a la base de dades per al cas {validation_data.get('cas', 'N/A')}...")
                db.add(evaluate_record)
                db.commit()
                logger.info(f"Resultats de validació guardats correctament a la base de dades per al cas {validation_data.get('cas', 'N/A')}.")
                
                # Crear el diccionari de retorn amb totes les mètriques
                result = {
                    'metrics': {
                        'classification': {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1': float(f1)
                        },
                        'order': {
                            'accuracy': float(order_accuracy),
                            'kendall_tau': float(kendall_tau)
                        }
                    },
                    'losses': {
                        'code_loss': float(code_loss.item()),
                        'order_loss': float(order_loss.item())
                    },
                    'predictions': {
                        'top_15_codes': top_15_codes,
                        'predicted_order': [codes[i] for i in unique_predicted_order],
                        'true_order': codes
                    }
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error durant la validació: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error en la validació del cas {validation_data.get('cas', 'N/A')}: {str(e)}")
            raise

    async def train_incremental(self, data: dict, db: Session):
        """
        Entrena el model amb un nou cas.
        """
        try:
            # Registrar hora de inicio
            hora_inici = datetime.now()
            
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
            
            # Netejar i validar el codi CIE-10
            dx_revisat = data.get('dx_revisat')
            if not dx_revisat or not isinstance(dx_revisat, str):
                raise ValueError("El cas ha de tenir un codi CIE-10 revisat")
            
            # Netejar els codis CIE-10 (eliminar separadors buits)
            codes = [code.strip() for code in dx_revisat.split('|') if code.strip()]
            
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
                else:
                    predicted_codes_set = set()
                
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
                    logger.info(f"S'han afegit {codis_nous} codis nous al conjunt de codis predits")
                
            except Exception as e:
                logger.error(f"Error actualitzant el conjunt de codis predits: {str(e)}")
                raise
            
            # Preparar el text combinant tots els camps
            text = prepare_text(data, text_processor)
            
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
            except Exception as e:
                logger.error(f"Error preparant etiquetes: {str(e)}")
                raise
            
            # Calcular pesos per classe
            try:
                class_weights = calculate_class_weights(labels, NUM_LABELS)
            except Exception as e:
                logger.error(f"Error calculant pesos per classe: {str(e)}")
                class_weights = torch.ones(NUM_LABELS).to(DEVICE)
            
            # Configurar pèrdua amb pesos dinàmics
            try:
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
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
            epochs_completed = 0
            final_lr = 2e-5  # Learning rate inicial
            final_total_loss = 0.0
            final_code_loss = 0.0
            final_order_loss = 0.0
            
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
                    
                    # Actualizar learning rate final y pérdidas finales
                    final_lr = optimizer.param_groups[0]['lr']
                    final_total_loss = total_loss.item()
                    final_code_loss = code_loss.item()
                    final_order_loss = order_loss.item()
                    
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
                            epochs_completed = epoch + 1
                            break
                    
                    epochs_completed = epoch + 1
                
                except Exception as e:
                    logger.error(f"Error durant l'entrenament de l'època {epoch+1}: {str(e)}")
                    raise
            
            # Registrar hora de fin y calcular duración
            hora_fi = datetime.now()
            durada = (hora_fi - hora_inici).total_seconds()
            
            # Guardar datos de entrenamiento en la base de datos
            train_record = DeepLearningTrain(
                cas=data['cas'],
                data_entrenament=hora_inici.date(),
                hora_inici=hora_inici.strftime('%H:%M:%S'),
                hora_fi=hora_fi.strftime('%H:%M:%S'),
                durada=f"{durada:.2f}",
                lr_inicial=2e-5,
                decay=0.9,
                batch_size=batch_size,
                max_epochs=max_epochs,
                early_stopping_patience=patience,
                max_len_input=4096,
                umbral_confianza_prediccio=0.5,
                epochs=epochs_completed,
                lr_final=final_lr,
                loss_total_final=final_total_loss,
                loss_code_final=final_code_loss,
                loss_order_final=final_order_loss
            )
            
            db.add(train_record)
            db.commit()
            
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
            
        except Exception as e:
            logger.error(f"Error en l'entrenament del cas {data['cas']}: {str(e)}")
            raise

    def predict(self, data: dict) -> Dict[str, Any]:
        """
        Realitza prediccions per un cas clínic sense validar els codis.
        """
        try:
            # Cargar el conjunto de códigos entrenados
            predicted_codes_path = os.path.join(MODEL_DIR, 'predicted_codes.pkl')
            logger.info(f"Intentant carregar el conjunt de codis entrenats des de: {predicted_codes_path}")
            
            if os.path.exists(predicted_codes_path):
                with open(predicted_codes_path, 'rb') as f:
                    predicted_codes_set = pickle.load(f)
                logger.info(f"Conjunt de codis entrenats carregat correctament. Total: {len(predicted_codes_set)}")
                logger.info(f"Codis disponibles per predicció: {sorted(list(predicted_codes_set))}")
            else:
                logger.error(f"No s'ha trobat el fitxer de codis entrenats a: {predicted_codes_path}")
                raise FileNotFoundError("No s'ha trobat el fitxer de codis entrenats (predicted_codes.pkl)")
            
            # Preparar el text combinant tots els camps
            text = prepare_text(data, text_processor)
            
            # Tokenitzar el text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True
            ).to(DEVICE)
            
            # Preparar entrades categòriques
            categorical_inputs = prepare_categorical_inputs(data)
            inputs.update(categorical_inputs)
            
            # Fer prediccions
            with torch.no_grad():
                code_logits, order_logits = self.model(inputs)
                
                # Obtenir prediccions de classificació
                code_probs = torch.sigmoid(code_logits)
                
                # Obtenir prediccions d'ordre
                order_probs = torch.softmax(order_logits, dim=1)
                
                # Crear un diccionari de códigos y sus probabilidades
                code_prob_dict = {}
                for i, code in enumerate(mlb.classes_):
                    if code in predicted_codes_set:
                        code_prob_dict[code] = code_probs[0, i].item()
                
                # Filtrar solo los códigos con probabilidad > 0.9
                filtered_codes = {code: prob for code, prob in code_prob_dict.items() if prob > 0.9}
                
                # Si no hi ha prediccions amb probabilitat > 0.9, retornar llista buida
                if not filtered_codes:
                    logger.info("No s'han trobat prediccions amb probabilitat superior al 90%")
                    return {
                        "cas": data["cas"],
                        "prediccions": [],
                        "status": "info",
                        "message": "No s'han trobat prediccions amb probabilitat superior al 90%"
                    }
                
                # Crear llista de prediccions amb codi i probabilitat
                predictions = []
                for code, prob in filtered_codes.items():
                    predictions.append({
                        "code": code,
                        "probability": prob
                    })
                
                # Ordenar per probabilitat
                predictions.sort(key=lambda x: x["probability"], reverse=True)
                
                return {
                    "cas": data["cas"],
                    "prediccions": predictions,
                    "status": "success",
                    "message": f"S'han trobat {len(predictions)} prediccions amb probabilitat superior al 90%"
                }
                
        except Exception as e:
            logger.error(f"Error en la predicció: {str(e)}")
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "error",
                "message": str(e)
            }
