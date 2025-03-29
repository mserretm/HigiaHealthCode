# HigiaHealthCode API

## Descripció
API per a la classificació automàtica de codis CIE-10 utilitzant el model Clinical-Longformer. Aquest projecte forma part d'un treball de final de master en l'àmbit de la intel·ligència artificial aplicada a la salut.

## Característiques Principals
- Predicció automàtica de codis CIE-10
- Entrenament incremental amb nous casos
- Entrenament en batch
- Gestió eficient de la memòria
- Processament intel·ligent de text clínic

## Requisits Previs
- Python 3.8 o superior
- CUDA (opcional, per acceleració GPU)
- 8GB RAM mínim (recomanat 16GB)

## Instal·lació
1. Clonar el repositori:
```bash
git clone https://github.com/usuari/higiahealthcode.git
cd higiahealthcode
```

2. Crear i activar un entorn virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instal·lar dependències:
```bash
pip install -r requirements.txt
```

## Estructura del Projecte
```
app/
├── data/               # Dades CIE-10 i altres recursos
├── ml/                # Components del model
│   ├── engine.py      # Motor principal del model
│   ├── model.py       # Definició del model
│   └── utils.py       # Utilitats i funcions auxiliars
├── routes/            # Endpoints de l'API
│   ├── predict.py     # Predicció de codis
│   ├── train.py       # Entrenament incremental
│   └── train_in_batch.py  # Entrenament en batch
└── schemas/           # Esquemes de validació
    └── requests.py    # Definició d'esquemes
```

## Ús
1. Iniciar el servidor:
```bash
python -m app.main
```

2. Accedir a la documentació:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints Principals
- `/predict`: Predicció de codis CIE-10
- `/train`: Entrenament incremental
- `/train-pending`: Entrenament automàtic dels registres pendents
- `/reset`: Reinicialització del model

## Exemples d'Ús
### Predicció
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "case": {
        "cas": "CAS001",
        "malaltiaactual": "Pacient amb dolor toràcic...",
        # ... altres camps
    }
}
response = requests.post(url, json=data)
print(response.json())
```

### Entrenament Automàtic
```python
import requests

url = "http://localhost:8000/train-pending/start"
response = requests.post(url)
print(response.json())
```

## Consideracions
- El model utilitza Longformer amb una finestra d'atenció de 4096 tokens
- Els textos llargs es processen de manera intel·ligent
- S'eliminen automàticament les etiquetes HTML del text
- Es normalitzen els textos a minúscules
- Els models entrenats no es pujaran al repository per temes de protecció de dades.


### Col·laboracion
- UOC - Univeristat obeta de catalunya
- XST - Xarxa Sanitaria, Social i Docent de Santa Tecla.

### Llicència
Aquest projecte està sota la llicència MIT. Pots modificar i redistribuir el codi, però sempre mencionant l'autor original.
