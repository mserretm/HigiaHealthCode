"""
Model de deep learning personalitzat per la classificació múltiple de codis CIE-10.
Utilitza Longformer com a base i implementa una xarxa neuronal profunda per predir
tant la presència de codis com el seu ordre de rellevància.
"""

import torch
import torch.nn as nn
import os
import logging
from transformers import LongformerTokenizer, LongformerModel
from typing import Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class CIE10Classifier(nn.Module):
    """
    Model de classificació múltiple personalitzat basat en deep learning.
    
    Atributs:
        hidden_size: Dimensió de l'embedding de Longformer
        intermediate_size: Dimensió de la capa intermèdia
        num_labels: Nombre de codis CIE-10 a predir
        categorical_dims: Dimensions de les variables categòriques
        longformer: Model base Longformer
        categorical_embeddings: Capes d'embedding per variables categòriques
        text_encoder: Codificador del text
        code_classifier: Classificador de codis
        order_classifier: Classificador d'ordre
    """
    
    def __init__(self, num_labels: int):
        """
        Inicialitza el model.
        
        Args:
            num_labels: Nombre de codis CIE-10 a predir
            
        Raises:
            OSError: Si no es troba el model Longformer local
            RuntimeError: Si hi ha errors en inicialitzar el model
        """
        super().__init__()
        
        # Cargar Longformer desde el directorio local
        local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app", "models", "clinical-longformer")
        self.longformer = LongformerModel.from_pretrained(local_path, local_files_only=True)
        
        # Configuración del modelo
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.longformer.config.hidden_size, num_labels)
        self.order_classifier = nn.Linear(self.longformer.config.hidden_size, num_labels)
        
        # Mantener todas las dimensiones originales
        self.hidden_size = 768  # Dimensión original
        self.intermediate_size = 3072  # Dimensión original
        
        # Simplificar dimensiones categóricas
        self.categorical_dims = {
            'edat': 10,      # Grupos de edad en décadas (0-9)
            'genere': 2,     # Binario (0-1)
            'c_alta': 8,     # (1-8)
            'periode': 5,    # 
            'servei': 50     # Reducido a 50 servicios más comunes
        }

        # Embeddings categóricos más simples
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, 16)  # Reducir dimensión de embedding
            for name, dim in self.categorical_dims.items()
        })
        
        # Simplificar codificador de texto
        self.text_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Simplificar clasificadores
        combined_dim = self.intermediate_size + len(self.categorical_dims) * 16
        self.code_classifier = nn.Sequential(
            nn.Linear(combined_dim, self.num_labels),
            nn.Sigmoid()
        )
        
        # Clasificador de orden
        self.order_classifier = nn.Sequential(
            nn.Linear(combined_dim, self.num_labels),
            nn.Sigmoid()
        )
        
    def process_categorical_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            categorical_features = []
            device = next(self.parameters()).device
            
            # Procesar cada variable categórica
            for name, dim in self.categorical_dims.items():
                if name in inputs:
                    # Preprocesar valor
                    if name == 'edat':
                        # Convertir edad a década
                        value = torch.clamp(inputs[name] // 10, 0, 9)
                    elif name == 'periode':
                        # Convertir año a índice (2010=0, 2011=1, etc.)
                        year = inputs[name].item()
                        if year < 2010:
                            logger.warning(f"Año {year} anterior a 2010, usando 2010 como valor mínimo")
                            year = 2010
                        value = torch.tensor(min(year - 2010, 4), device=device)  # Máximo 4 (2014)
                        logger.debug(f"Año {year} convertido a índice {value.item()}")
                    elif name == 'servei':
                        # Mapear servicio a un índice limitado
                        value = torch.clamp(inputs[name], 0, 49)
                    elif name == 'c_alta':
                        # Asegurar que c_alta está en el rango correcto (1-8)
                        value = torch.clamp(inputs[name], 0, 7)  # 0-7 para embedding
                    else:
                        value = inputs[name]
                    
                    feature = value.long()
                else:
                    # Valor por defecto específico para cada variable
                    default_values = {
                        'edat': 5,      # Década media
                        'genere': 0,    # Valor más común
                        'c_alta': 0,    # Ajustado a 0 para embedding
                        'periode': 4,   # Índice para 2014
                        'servei': 0     # Servicio general
                    }
                    feature = torch.tensor([default_values[name]], device=device)
                    logger.debug(f"Usando valor por defecto para {name}: {default_values[name]}")

                # Verificar que el índice está dentro del rango
                if torch.any(feature >= dim):
                    logger.warning(f"Índice fuera de rango para {name}. Valor: {feature.item()}, Máximo: {dim-1}")
                    feature = torch.clamp(feature, 0, dim-1)

                embedded = self.categorical_embeddings[name](feature)
                categorical_features.append(embedded)

            return torch.cat(categorical_features, dim=1)
            
        except Exception as e:
            logger.error(f"Error en processar variables categòriques: {str(e)}")
            raise

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Procesar texto (un solo campo o promedio de campos disponibles)
            text_features = None
            valid_fields = []
            
            for field in ["motiu_ingres", "malaltia_actual"]:  # Usar solo los campos más relevantes
                input_ids_key = f"{field}_input_ids"
                attention_mask_key = f"{field}_attention_mask"
                
                if input_ids_key in inputs and attention_mask_key in inputs:
                    outputs = self.longformer(
                        input_ids=inputs[input_ids_key],
                        attention_mask=inputs[attention_mask_key]
                    )
                    features = self.text_encoder(outputs.last_hidden_state.mean(dim=1))
                    valid_fields.append(features)
            
            if not valid_fields:
                raise ValueError("No se encontraron campos de texto válidos")
            
            text_features = torch.mean(torch.stack(valid_fields), dim=0)
            
            # Procesar variables categóricas
            categorical_features = self.process_categorical_features(inputs)
            
            # Combinar y clasificar
            combined_features = torch.cat([text_features, categorical_features], dim=1)
            
            # Predicciones de códigos y orden
            code_predictions = self.code_classifier(combined_features)
            order_predictions = self.order_classifier(combined_features)
            
            return code_predictions, order_predictions
            
        except Exception as e:
            logger.error(f"Error en forward pass: {str(e)}")
            raise 