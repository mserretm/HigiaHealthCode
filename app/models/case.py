from sqlalchemy import Column, Integer, String, DateTime, Text
from app.db.database import Base

class Case(Base):
    __tablename__ = "f_cmbd_ha_deeplearning"

    cas = Column(String(25), primary_key=True)
    dx_revisat = Column(Text)
    edat = Column(Integer)
    genere = Column(String(1))
    c_alta = Column(String(1))
    periode = Column(String(4))
    servei = Column(String(10))
    motiuingres = Column(Text)
    malaltiaactual = Column(Text)
    exploracio = Column(Text)
    provescomplementariesing = Column(Text)
    provescomplementaries = Column(Text)
    evolucio = Column(Text)
    antecedents = Column(Text)
    cursclinic = Column(Text)
    us_estatentrenament = Column(Integer, default=0)
    us_registre = Column(String(1))
    dx_prediccio = Column(Text)
    us_dataentrenament = Column(DateTime) 