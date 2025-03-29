"""
Model de deep learning personalitzat per la classificació múltiple de codis CIE-10.
Utilitza Longformer com a base i implementa una xarxa neuronal profunda per predir
tant la presència de codis com el seu ordre de rellevància.
"""

import torch
import torch.nn as nn
import os
import logging
from transformers import LongformerTokenizer, LongformerModel, LongformerConfig
from typing import Dict, Any, Tuple, Optional, Union, List

logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "allenai/longformer-base-4096"

class CIE10Classifier(nn.Module):
    def __init__(self, num_labels: int):
        """
        Inicialitza el model amb els paràmetres necessaris.
        """
        super().__init__()
        self.num_labels = num_labels
        
        # Inicialitzar el model base
        logger.info("Inicialitzant model base Longformer...")
        self.text_encoder = LongformerModel.from_pretrained(MODEL_ID)
        logger.info("Model base inicialitzat correctament")
        
        # Inicialitzar els classificadors
        logger.info("Inicialitzant classificadors...")
        self.code_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )
        logger.info("Classificador de codis inicialitzat")
        
        self.order_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )
        logger.info("Classificador d'ordre inicialitzat")
        
        # Definir límits per als embeddings categòrics
        self.embedding_limits = {
            'edat': 100,
            'genere': 3,
            'c_alta': 3,
            'periode': 100,
            'servei': 100
        }
        
        # Inicialitzar embeddings categòrics
        logger.info("Inicialitzant embeddings categòrics...")
        self.categorical_embeddings = nn.ModuleDict({
            'edat': nn.Embedding(self.embedding_limits['edat'], 32),
            'genere': nn.Embedding(self.embedding_limits['genere'], 16),
            'c_alta': nn.Embedding(self.embedding_limits['c_alta'], 16),
            'periode': nn.Embedding(self.embedding_limits['periode'], 32),
            'servei': nn.Embedding(self.embedding_limits['servei'], 32)
        })
        logger.info("Embeddings categòrics inicialitzats")
        
        # Inicialitzar projecció
        logger.info("Inicialitzant capa de projecció...")
        self.categorical_projection = nn.Linear(128, 768)
        logger.info("Capa de projecció inicialitzada")
        
        logger.info("Model complet inicialitzat correctament")
        
    def _validate_and_clip_categorical(self, field: str, value: torch.Tensor) -> torch.Tensor:
        """Valida y ajusta los valores categóricos dentro de los límites permitidos."""
        limit = self.embedding_limits[field]
        return torch.clamp(value, 0, limit - 1)
        
    def forward(self, inputs):
        # Processar text
        text_inputs = {k: v for k, v in inputs.items() if k not in self.categorical_embeddings.keys()}
        text_outputs = self.text_encoder(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Processar variables categòriques
        categorical_embeddings = []
        for field, embedding_layer in self.categorical_embeddings.items():
            if field in inputs:
                try:
                    # Assegurar que el tensor està en el dispositiu correcte
                    field_value = inputs[field].to(text_embeddings.device)
                    
                    # Validar i ajustar valors
                    field_value = self._validate_and_clip_categorical(field, field_value)
                    
                    # Aplicar embedding
                    field_embeddings = embedding_layer(field_value.squeeze(-1))  # [batch_size, embedding_dim]
                    categorical_embeddings.append(field_embeddings)
                except Exception as e:
                    logger.error(f"Error procesando campo {field}: {str(e)}")
                    logger.error(f"Valor del campo: {inputs[field]}")
                    logger.error(f"Forma del tensor: {inputs[field].shape}")
                    raise
        
        # Combinar embeddings
        if categorical_embeddings:
            try:
                categorical_combined = torch.cat(categorical_embeddings, dim=-1)  # [batch_size, total_categorical_dim]
                categorical_projected = self.categorical_projection(categorical_combined)  # [batch_size, hidden_size]
                combined_embeddings = text_embeddings + categorical_projected
            except Exception as e:
                logger.error(f"Error combinando embeddings: {str(e)}")
                logger.error(f"Número de embeddings categóricos: {len(categorical_embeddings)}")
                logger.error(f"Dimensiones de text_embeddings: {text_embeddings.shape}")
                raise
        else:
            combined_embeddings = text_embeddings
        
        # Generar prediccions
        try:
            code_logits = self.code_classifier(combined_embeddings)  # [batch_size, num_labels]
            order_logits = self.order_classifier(combined_embeddings)  # [batch_size, num_labels]
        except Exception as e:
            logger.error(f"Error en la clasificación: {str(e)}")
            logger.error(f"Dimensiones de combined_embeddings: {combined_embeddings.shape}")
            raise
        
        return code_logits, order_logits
