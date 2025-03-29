# Sistema de Classificació Automàtica de Codis CIE-10

Sistema de classificació automàtica de codis CIE-10 utilitzant models de deep learning.

## Estructura del Projecte

```
app/
├── core/               # Configuracions i funcions principals
│   ├── config.py      # Configuracions de l'aplicació
│   └── logging.py     # Configuració de logging
├── data/              # Dades i recursos
│   └── CIM10MC_2024-2025_20231221.txt  # Fitxer de codis CIE-10
├── db/                # Base de dades i models
│   ├── database.py    # Configuració de la base de dades
│   └── models.py      # Models de la base de dades
├── ml/                # Mòduls de machine learning
│   ├── engine.py      # Motor principal del model
│   ├── model.py       # Definició del model neural
│   └── utils.py       # Utilitats per al processament
├── models/            # Models entrenats
│   └── model.pt       # Model entrenat
├── routes/            # Endpoints de l'API
│   ├── json/         # Endpoints per a peticions JSON
│   │   ├── predict.py
│   │   ├── train.py
│   │   └── validate.py
│   └── __init__.py
├── schemas/           # Esquemes de validació
│   └── request.py     # Esquemes de peticions
├── services/          # Serveis de l'aplicació
│   ├── case_processor.py  # Processament de casos
│   └── requests.py    # Gestió de peticions
├── main.py           # Punt d'entrada de l'aplicació
└── __init__.py
```

## Requisits

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- SQLAlchemy
- PostgreSQL

## Instal·lació

1. Clonar el repositori:
```bash
git clone [URL_DEL_REPOSITORI]
cd [NOM_DEL_DIRECTORI]
```

2. Crear i activar l'entorn virtual:
```bash
python -m venv venv-tfm
source venv-tfm/bin/activate  # Linux/Mac
venv-tfm\Scripts\activate     # Windows
```

3. Instal·lar dependències:
```bash
pip install -r requirements.txt
```

4. Configurar la base de dades:
```bash
# Crear la base de dades PostgreSQL
createdb cie10_classifier

# Configurar les variables d'entorn
cp .env.example .env
# Editar .env amb les seves credencials
```

## Execució

1. Iniciar el servidor:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. Accedir a la documentació de l'API:
```
http://localhost:8000/docs
```

## Funcionalitats

- Classificació automàtica de codis CIE-10
- Entrenament del model amb dades personalitzades
- Validació de prediccions
- API REST per a integració amb altres sistemes

### Col·laboracion
- UOC - Univeristat obeta de catalunya
- XST - Xarxa Sanitaria, Social i Docent de Santa Tecla.

- [Documentació de FastAPI](https://fastapi.tiangolo.com/)
- [Documentació de PyTorch](https://pytorch.org/docs/stable/index.html)
- [Documentació de Transformers](https://huggingface.co/docs/transformers/index)


### Llicència
Aquest projecte està sota la llicència MIT. Pots modificar i redistribuir el codi, però sempre mencionant l'autor original.