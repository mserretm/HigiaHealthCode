from pydantic import BaseModel
from typing import List, Optional, Dict

class Case(BaseModel):
    cas: str
    motiuingres: Optional[str] = None
    malaltiaactual: Optional[str] = None
    exploracio: Optional[str] = None
    provescomplementariesing: Optional[str] = None
    provescomplementaries: Optional[str] = None
    evolucio: Optional[str] = None
    antecedents: Optional[str] = None
    cursclinic: Optional[str] = None
    edat: Optional[int] = None
    genere: Optional[int] = None
    c_alta: Optional[int] = None
    periode: Optional[int] = None
    servei: Optional[int] = None
    dx_revisat: Optional[str] = None

class Prediction(BaseModel):
    code: str
    probability: float

class CaseResponse(BaseModel):
    cas: str
    prediccions: List[Prediction]
    status: str
    message: str 