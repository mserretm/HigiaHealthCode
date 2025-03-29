"""
Mòdul per al processament i neteja de textos clínics.
Implementa un pipeline de transformació per normalitzar i preparar els textos
per al model de classificació.
"""

import re
import unicodedata
import nltk
import os
import pickle
from typing import List, Optional, Set
from bs4 import BeautifulSoup
from loguru import logger

class ClinicalTextProcessor:
    def __init__(self):
        """
        Inicialitza el processador de text clínic.
        """
        try:
            # Definir la ruta del fitxer de stopwords
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.stopwords_path = os.path.join(BASE_DIR, "models", "stopwords.pkl")
            
            # Intentar carregar stopwords del fitxer
            if os.path.exists(self.stopwords_path):
                logger.info("Carregant stopwords des del fitxer...")
                with open(self.stopwords_path, 'rb') as f:
                    self.stopwords = pickle.load(f)
                logger.info(f"Stopwords carregades correctament ({len(self.stopwords)} paraules)")
            else:
                logger.info("Generant i guardant stopwords...")
                self._generate_stopwords()
                
        except Exception as e:
            logger.error(f"Error inicialitzant el processador de text: {str(e)}")
            # Fallback a stopwords bàsiques si hi ha error
            self.stopwords = {'el', 'la', 'els', 'les', 'un', 'una', 'uns', 'unes', 'i', 'o', 'per', 'amb'}

    def _generate_stopwords(self) -> None:
        """
        Genera i guarda el conjunt de stopwords.
        """
        try:
            # Descarregar recursos necessaris de NLTK
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            
            # Obtenir stopwords de NLTK
            spanish_stopwords = set(nltk.corpus.stopwords.words('spanish'))
            
            # Stopwords en català (combinació de NLTK i personalitzades)
            catalan_stopwords = {
                'el', 'la', 'els', 'les', 'un', 'una', 'uns', 'unes',
                'i', 'o', 'per', 'amb', 'sense', 'sobre', 'sota', 'dins',
                'fora', 'entre', 'després', 'abans', 'durant', 'mentre',
                'que', 'quan', 'on', 'com', 'per què', 'quin', 'quina',
                'quins', 'quines', 'qui', 'què', 'a', 'al', 'als', 'de',
                'del', 'dels', 'en', 'es', 'et', 'ha', 'han', 'has',
                'hem', 'heu', 'hi', 'ho', 'l', 'la', 'les', 'li', 'ls',
                'm', 'me', 'n', 'ne', 'no', 'ns', 's', 'sa', 'se', 'ses',
                'si', 'so', 'sos', 't', 'te', 'us', 'va', 'van', 'vas',
                'veu', 'vos', 'vostè', 'vostès'
            }
            
            # Combinar stopwords de tots els idiomes
            self.stopwords = spanish_stopwords.union(catalan_stopwords)
            
            # Afegir variacions comunes
            additional_stopwords = {
                'd', 'l', 'm', 'n', 's', 't',  # Abreviatures comunes
                'dr', 'dra', 'sr', 'sra',  # Títols
                'etc', 'etcetera',  # Altres
                'mes', 'mesos', 'any', 'anys',  # Unitats de temps
                'dia', 'dies', 'setmana', 'setmanes',
                'hora', 'hores', 'minut', 'minuts',
                'segon', 'segons'
            }
            self.stopwords.update(additional_stopwords)
            
            # Crear directori models si no existeix
            os.makedirs(os.path.dirname(self.stopwords_path), exist_ok=True)
            
            # Guardar stopwords al fitxer
            with open(self.stopwords_path, 'wb') as f:
                pickle.dump(self.stopwords, f)
            
            logger.info(f"Stopwords generades i guardades correctament ({len(self.stopwords)} paraules)")
            
        except Exception as e:
            logger.error(f"Error generant stopwords: {str(e)}")
            raise

    def normalize_text(self, text: str) -> str:
        """
        Normalitza el text aplicant diverses transformacions.
        
        Args:
            text: Text a normalitzar
            
        Returns:
            Text normalitzat
        """
        if not text:
            return ""
            
        # Convertir a UTF-8 i normalitzar caràcters especials
        text = unicodedata.normalize('NFKC', text)
        
        # Convertir a minúscules
        text = text.lower()
        
        # Eliminar tags HTML si n'hi ha
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Eliminar puntuació i caràcters especials no informatius
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Eliminar espais múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar espais al principi i final
        text = text.strip()
        
        return text

    def remove_stopwords(self, text: str) -> str:
        """
        Elimina les paraules sense càrrega semàntica.
        
        Args:
            text: Text del qual eliminar les stopwords
            
        Returns:
            Text sense stopwords
        """
        words = text.split()
        return ' '.join([word for word in words if word not in self.stopwords])

    def process_text(self, text: str, remove_stops: bool = True) -> str:
        """
        Aplica el pipeline complet de processament de text.
        
        Args:
            text: Text a processar
            remove_stops: Si s'han d'eliminar les stopwords
            
        Returns:
            Text processat
        """
        try:
            # Normalitzar el text
            text = self.normalize_text(text)
            
            # Eliminar stopwords si s'especifica
            if remove_stops:
                text = self.remove_stopwords(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error processant el text: {str(e)}")
            return text

    def process_clinical_case(self, case: dict) -> dict:
        """
        Processa tots els camps de text d'un cas clínic.
        
        Args:
            case: Diccionari amb els camps del cas clínic
            
        Returns:
            Diccionari amb els camps processats
        """
        processed_case = case.copy()
        
        # Camps a processar (noms de la base de dades)
        text_fields = [
            'motiuingres',
            'malaltiaactual',
            'exploracio',
            'provescomplementariesing',
            'provescomplementaries',
            'evolucio',
            'antecedents',
            'cursclinic'
        ]
        
        for field in text_fields:
            if field in processed_case and processed_case[field]:
                processed_case[field] = self.process_text(processed_case[field])
        
        return processed_case 