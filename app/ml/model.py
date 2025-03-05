"""
Model de deep learning personalitzat per la classificació múltiple de codis CIE-10.
Utilitza Longformer com a tokenitzador i implementa una xarxa neuronal profunda.
"""

import torch
import torch.nn as nn
import os
import logging
from transformers import LongformerTokenizer, LongformerModel
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CIE10Classifier(nn.Module):
    """
    Model de classificació múltiple personalitzat basat en deep learning.
    Prediu tant els codis com el seu ordre de rellevància.
    """
    def __init__(self, num_labels: int):
        super().__init__()
        
        # Configuració de dimensions
        self.hidden_size = 768  # Mida de l'embedding de Longformer
        self.intermediate_size = 512
        self.num_labels = num_labels
        
        # Definir la ruta local del model Longformer
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        LOCAL_LONGFORMER_PATH = os.path.join(BASE_DIR, "models", "clinical-longformer")
        
        # Carregar el model Longformer des de la ruta local
        try:
            self.longformer = LongformerModel.from_pretrained(LOCAL_LONGFORMER_PATH, local_files_only=True)
            logger.info(f"Model Longformer carregat correctament des de {LOCAL_LONGFORMER_PATH}")
        except Exception as e:
            logger.error(f"Error en carregar el model Longformer: {str(e)}")
            raise
        
        # Dimensions per variables categòriques
        self.categorical_dims = {
            'edat': 110,  # 0-110 anys
            'genere': 2,  # 0 o 1
            'c_alta': 8,  # 1-8
            'periode': 2024,  # Any 2024
            'servei': 36**5  # Codi alfanumèric de 5 caràcters (36 possibilitats per caràcter)
        }
        # Embeddings per variables categòriques
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Sequential(
                nn.Embedding(dim, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            for name, dim in self.categorical_dims.items()
        })
        
        # Capes de processament de text
        self.text_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.LayerNorm(self.intermediate_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Capa d'atenció per combinar els diferents camps de text
        self.attention = nn.MultiheadAttention(
            embed_dim=self.intermediate_size,
            num_heads=8,
            dropout=0.2
        )
        
        # Capes de classificació per codis
        self.code_classifier = nn.Sequential(
            nn.Linear(self.intermediate_size + len(self.categorical_dims) * 32, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_labels),
            nn.Sigmoid()
        )
        
        # Capes de classificació per ordre
        self.order_classifier = nn.Sequential(
            nn.Linear(self.intermediate_size + len(self.categorical_dims) * 32, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_labels),
            nn.Softmax(dim=1)
        )
        
        # Inicialitzar els pesos
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Inicialitza els pesos del model."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def process_categorical_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Processa les variables categòriques."""
        categorical_features = []
        for name, embedding in self.categorical_embeddings.items():
            if name in inputs:
                feature = inputs[name].long()
                embedded = embedding(feature)
                categorical_features.append(embedded)
        
        return torch.cat(categorical_features, dim=1)
            
    def forward(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        """
        Forward pass del model.
        
        Args:
            inputs (dict): Diccionari amb:
                - input_ids: Tensor amb els IDs dels tokens
                - attention_mask: Tensor amb la màscara d'atenció
                - edat: Tensor amb l'edat
                - genere: Tensor amb el gènere
                - c_alta: Tensor amb el codi d'alta
                - periode: Tensor amb el període
                - servei: Tensor amb el servei
                
        Returns:
            tuple: (probabilitats de codis, probabilitats d'ordre)
        """
        try:
            # Processar el text amb Longformer
            longformer_outputs = self.longformer(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            text_embeddings = self.text_encoder(longformer_outputs.last_hidden_state)
            
            # Aplicar atenció sobre els embeddings del text
            attn_output, _ = self.attention(
                text_embeddings,
                text_embeddings,
                text_embeddings
            )
            
            # Processar variables categòriques
            categorical_embeddings = self.process_categorical_features(inputs)
            
            # Combinar totes les característiques
            combined_features = torch.cat([
                attn_output.mean(dim=1),  # Promig de l'atenció
                categorical_embeddings
            ], dim=1)
            
            # Predicció de codis i ordre
            code_probabilities = self.code_classifier(combined_features)
            order_probabilities = self.order_classifier(combined_features)
            
            return code_probabilities, order_probabilities
            
        except Exception as e:
            logger.error(f"Error en el forward pass: {str(e)}")
            raise 