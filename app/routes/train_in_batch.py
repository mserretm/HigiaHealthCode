# app/routes/train_in_batch.py

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List
from fastapi import APIRouter, HTTPException
from app.schemas.requests import BatchTrainRequest
from app.ml.engine import (
    model, 
    tokenizer, 
    DEVICE, 
    mlb, 
    NUM_LABELS, 
    save_model, 
    optimizer, 
    scheduler, 
    MODEL_PATH, 
    MODEL_DIR
)
from app.ml.utils import truncate_field, freeze_bert_layers
from app.ml.training_lock import training_lock

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
async def train_in_batch(request: BatchTrainRequest):
    """
    Endpoint per realitzar entrenament en batch amb múltiples casos.
    """
    try:
        with training_lock:  # Adquirir el lock
            logger.info("Iniciant entrenament en batch...")
            
            # 1) Congelar capes i descongelar les últimes dues
            freeze_bert_layers(model.longformer, num_unfrozen_layers=2)
            
            # 2) Preparar tots els casos
            all_inputs = []
            all_labels = []
            
            for case in request.cases:
                # Preparar text
                text_fields = [
                    truncate_field(str(case.get(field, ""))) 
                    for field in ["motiu_ingres", "malaltia_actual", "exploracio", "evolucio"]
                ]
                full_text = " [SEP] ".join(text_fields)
                
                # Tokenitzar
                encoded = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=4096
                ).to(DEVICE)
                
                all_inputs.append(encoded)
                
                # Preparar etiquetes
                label_vector = mlb.transform([case.get("codes", [])])
                labels = torch.from_numpy(np.array(label_vector)).float().to(DEVICE)
                all_labels.append(labels)
            
            # 3) Configurar entrenament
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(NUM_LABELS).to(DEVICE) * 2.0)
            batch_size = 4
            epochs = request.epochs if hasattr(request, 'epochs') and request.epochs else 3
            
            # 4) Entrenar
            model.train()
            total_batches = len(all_inputs) // batch_size + (1 if len(all_inputs) % batch_size != 0 else 0)
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for batch_idx in range(0, len(all_inputs), batch_size):
                    batch_inputs = all_inputs[batch_idx:batch_idx + batch_size]
                    batch_labels = all_labels[batch_idx:batch_idx + batch_size]
                    
                    # Concatenar batch
                    input_ids = torch.cat([x["input_ids"] for x in batch_inputs], dim=0)
                    attention_mask = torch.cat([x["attention_mask"] for x in batch_inputs], dim=0)
                    labels = torch.cat(batch_labels, dim=0)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model({"input_ids": input_ids, "attention_mask": attention_mask})
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    
                    current_batch = batch_idx // batch_size + 1
                    logger.info(f"Època {epoch + 1}/{epochs}, Batch {current_batch}/{total_batches}, Pèrdua: {loss.item():.4f}")
                
                avg_epoch_loss = epoch_loss / total_batches
                logger.info(f"Època {epoch + 1} completada. Pèrdua mitjana: {avg_epoch_loss:.4f}")
            
            # 5) Guardar model
            save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                mlb=mlb,
                tokenizer=tokenizer,
                model_path=MODEL_PATH,
                model_dir=MODEL_DIR
            )
            
            logger.info("Entrenament en batch completat.")
            return {"message": "Model entrenat correctament en batch."}
            
    except Exception as e:
        logger.error(f"Error en entrenament batch: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
