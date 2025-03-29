# app/schemas/requests.py
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, List, Optional, Union
import re

class ClinicalCase(BaseModel):
    """
    Model per representar un cas clínic complet.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    cas: str = Field(..., min_length=1, max_length=50)
    edat: Optional[str] = Field(default="")
    genere: Optional[str] = Field(default="")
    c_alta: Optional[str] = Field(default="")
    periode: Optional[str] = Field(default="")
    servei: Optional[str] = Field(default="")
    motiuingres: Optional[str] = Field(default="", min_length=10)
    malaltiaactual: Optional[str] = Field(default="", min_length=10)
    exploracio: Optional[str] = Field(default="")
    provescomplementariesing: Optional[str] = Field(default="")
    provescomplementaries: Optional[str] = Field(default="")
    evolucio: Optional[str] = Field(default="")
    antecedents: Optional[str] = Field(default="")
    cursclinic: Optional[str] = Field(default="")

    @field_validator('edat')
    @classmethod
    def validate_edat(cls, v: str) -> str:
        if not v:
            return v
        if not v.isdigit() or not (0 <= int(v) <= 120):
            raise ValueError("L'edat ha de ser un número enter entre 0 i 120 anys")
        return v

    @field_validator('genere')
    @classmethod
    def validate_genere(cls, v: str) -> str:
        if not v:
            return v
        genere_map = {
            'H': 'H', 'D': 'D', 'HOME': 'H', 'DONA': 'D',
            'M': 'H', 'F': 'D', 'MASCULÍ': 'H', 'FEMENÍ': 'D',
            'MASCULINO': 'H', 'FEMENINO': 'D'
        }
        v_upper = v.strip().upper()
        if v_upper not in genere_map:
            raise ValueError("Gènere no vàlid")
        return genere_map[v_upper]

    @field_validator('*')
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        v = re.sub(r'<[^>]+>', '', v)
        v = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', v)
        return ' '.join(v.split())

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
    Esquema per validar les peticions d'entrenament.
    """
    cas: str = Field(..., min_length=1, max_length=50)
    dx_revisat: str = Field(..., min_length=1, max_length=10)
    edat: Optional[str] = Field(default="")
    genere: Optional[str] = Field(default="")
    c_alta: Optional[str] = Field(default="")
    periode: Optional[str] = Field(default="")
    servei: Optional[str] = Field(default="")
    motiuingres: Optional[str] = Field(default="")
    malaltiaactual: Optional[str] = Field(default="")
    exploracio: Optional[str] = Field(default="")
    provescomplementariesing: Optional[str] = Field(default="")
    provescomplementaries: Optional[str] = Field(default="")
    evolucio: Optional[str] = Field(default="")
    antecedents: Optional[str] = Field(default="")
    cursclinic: Optional[str] = Field(default="")

class ValidateRequest(BaseModel):
    """
    Esquema per validar les peticions de validació.
    """
    cas: str = Field(..., min_length=1, max_length=50)
    dx_revisat: str = Field(..., min_length=1, max_length=10)
    edat: Optional[str] = Field(default="")
    genere: Optional[str] = Field(default="")
    c_alta: Optional[str] = Field(default="")
    periode: Optional[str] = Field(default="")
    servei: Optional[str] = Field(default="")
    motiuingres: Optional[str] = Field(default="")
    malaltiaactual: Optional[str] = Field(default="")
    exploracio: Optional[str] = Field(default="")
    provescomplementariesing: Optional[str] = Field(default="")
    provescomplementaries: Optional[str] = Field(default="")
    evolucio: Optional[str] = Field(default="")
    antecedents: Optional[str] = Field(default="")
    cursclinic: Optional[str] = Field(default="")

class EvaluateRequest(BaseModel):
    """
    Esquema per validar les peticions d'avaluació.
    """
    cas: str = Field(..., min_length=1, max_length=50)
    dx_revisat: Optional[Union[List[str], str]] = Field(
        default=None,
        description="Codis CIE-10 revisats per avaluació, poden ser una llista o string separats per '|'"
    )
    edat: Optional[str] = Field(default="")
    genere: Optional[str] = Field(default="")
    c_alta: Optional[str] = Field(default="")
    periode: Optional[str] = Field(default="")
    servei: Optional[str] = Field(default="")
    motiuingres: Optional[str] = Field(default="")
    malaltiaactual: Optional[str] = Field(default="")
    exploracio: Optional[str] = Field(default="")
    provescomplementariesing: Optional[str] = Field(default="")
    provescomplementaries: Optional[str] = Field(default="")
    evolucio: Optional[str] = Field(default="")
    antecedents: Optional[str] = Field(default="")
    cursclinic: Optional[str] = Field(default="")
