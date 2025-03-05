# app/schemas/requests.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union

class ClinicalCase(BaseModel):
    """
    Model per representar un cas clínic complet.
    """
    cas: Optional[str] = Field(default="", description="Identificador del cas")
    edat: Optional[str] = Field(default="", description="Edat del pacient")
    genere: Optional[str] = Field(default="", description="Gènere del pacient")
    c_alta: Optional[str] = Field(default="", description="Circumstància d'alta")
    periode: Optional[str] = Field(default="", description="Període d'atenció")
    servei: Optional[str] = Field(default="", description="Servei mèdic")
    motiuingres: Optional[str] = Field(default="", description="Motiu d'ingrés")
    malaltiaactual: Optional[str] = Field(default="", description="Malaltia actual")
    exploracio: Optional[str] = Field(default="", description="Exploració física")
    provescomplementariesing: Optional[str] = Field(default="", description="Proves complementàries a l'ingrés")
    provescomplementaries: Optional[str] = Field(default="", description="Proves complementàries durant l'estada")
    evolucio: Optional[str] = Field(default="", description="Evolució clínica")
    antecedents: Optional[str] = Field(default="", description="Antecedents")
    cursclinic: Optional[str] = Field(default="", description="Curs clínic")

class PredictRequest(BaseModel):
    """
    Petició per realitzar prediccions.
    No inclou dx_revisat ja que aquests són els codis que el model ha de predir.
    """
    case: ClinicalCase

class TrainingCase(ClinicalCase):
    """
    Extensió de ClinicalCase per entrenament, que inclou dx_revisat.
    """
    dx_revisat: Union[List[str], str] = Field(
        ...,  # Camp obligatori per entrenament
        description="Codis CIE-10 revisats per entrenament, poden ser una llista o string separats per '|'"
    )

class TrainRequest(BaseModel):
    """
    Petició per entrenar el model amb un nou cas.
    """
    case: TrainingCase

class BatchTrainCase(BaseModel):
    """
    Cas individual per entrenament en batch.
    """
    case: TrainingCase

class BatchTrainRequest(BaseModel):
    """
    Petició per entrenament en batch.
    """
    cases: List[BatchTrainCase]
    epochs: Optional[int] = Field(default=3, description="Nombre d'èpoques d'entrenament")
