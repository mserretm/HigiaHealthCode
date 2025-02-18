## HigiaHealthCode

HigiaHealthCode és una API escalable per a la codificació mèdica automàtica amb CIE-10.
   
   ### Funcionalitats
   - Gestió de diagnòstics i procediments mèdics.
   - Connexió amb bases de dades PostgreSQL per a l'emmagatzematge de codis mèdics. (Primer sera per pantalla despres ja mirare com guardar-ho)
   - API REST basada en FastAPI.
   - Validació de dades amb Pydantic.
   - Registre i monitoratge de l'activitat amb un sistema de logging.
   - Probes automatitzades per garantir la qualitat del codi.
   - Implementació de models de Deep Learning amb PyTorch per a la codificació automàtica.
   
   ### Instal·lació
   ```bash
   git clone git@github.com:mserretm/HigiaHealthCode.git
   cd HigiaHealthCode
   pip install -r requirements.txt

   .\venv_tfm\Scripts\activate
   ```
   
   ### Execució
   ```bash
   uvicorn app.main:app --reload
   ```

    ### Limitacions
    Els models entrenats no es pujaran al repository per temes de protecció de dades.

   ### Col·laboracions

    UOC - Univeristat obeta de catalunya
    XST - Xarxa Sanitaria, Social i Docent de Santa Tecla.

   ### Llicència
   Aquest projecte està sota la llicència MIT. Pots modificar i redistribuir el codi, però sempre mencionant l'autor original.
