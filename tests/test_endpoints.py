import unittest
import json
import os
import requests

class TestEndpoints(unittest.TestCase):
    """
    Tests per verificar el funcionament dels endpoints de l'API.
    """
    
    def setUp(self):
        """Configuració inicial per cada test."""
        self.base_url = "http://localhost:8000"  # URL base de l'API
        self.test_data_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Carregar les dades d'exemple
        with open(os.path.join(self.test_data_dir, 'predict_request_example.json'), 'r', encoding='utf-8') as f:
            self.predict_data = json.load(f)
            
        with open(os.path.join(self.test_data_dir, 'retrain_request_example.json'), 'r', encoding='utf-8') as f:
            self.retrain_data = json.load(f)
            
        with open(os.path.join(self.test_data_dir, 'train_in_batch_request_example.json'), 'r', encoding='utf-8') as f:
            self.batch_train_data = json.load(f)

    def test_predict_endpoint(self):
        """Test de l'endpoint de predicció."""
        print("\nTestejant endpoint de predicció...")
        
        try:
            response = requests.post(f"{self.base_url}/predict/", json=self.predict_data)
            
            # Verificar la resposta
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn("predictions", data)
            self.assertIn("case_id", data)
            self.assertIn("text_length", data)
            
            # Verificar l'estructura de les prediccions
            if data["predictions"]:
                prediction = data["predictions"][0]
                self.assertIn("code", prediction)
                self.assertIn("descriptive", prediction)
                self.assertIn("probability", prediction)
                
            print("✓ Test de predicció completat amb èxit")
            print(f"  Nombre de prediccions: {len(data['predictions'])}")
            
        except Exception as e:
            print(f"✗ Error en el test de predicció: {str(e)}")
            raise

    def test_retrain_endpoint(self):
        """Test de l'endpoint de reentrenament."""
        print("\nTestejant endpoint de reentrenament...")
        
        try:
            response = requests.post(f"{self.base_url}/retrain/", json=self.retrain_data)
            
            # Verificar la resposta
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn("message", data)
            
            print("✓ Test de reentrenament completat amb èxit")
            print(f"  Missatge: {data['message']}")
            
        except Exception as e:
            print(f"✗ Error en el test de reentrenament: {str(e)}")
            raise

    def test_train_in_batch_endpoint(self):
        """Test de l'endpoint d'entrenament en batch."""
        print("\nTestejant endpoint d'entrenament en batch...")
        
        try:
            response = requests.post(f"{self.base_url}/train-batch/", json=self.batch_train_data)
            
            # Verificar la resposta
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn("message", data)
            
            print("✓ Test d'entrenament en batch completat amb èxit")
            print(f"  Missatge: {data['message']}")
            print(f"  Casos processats: {len(self.batch_train_data['cases'])}")
            
        except Exception as e:
            print(f"✗ Error en el test d'entrenament en batch: {str(e)}")
            raise

    def test_reset_endpoint(self):
        """Test de l'endpoint de reinicialització."""
        print("\nTestejant endpoint de reinicialització...")
        
        try:
            # Primer fem una petició GET per veure la informació
            response_get = requests.get(f"{self.base_url}/reset/")
            self.assertEqual(response_get.status_code, 200)
            print("  Info de reinicialització obtinguda correctament")
            
            # Després fem la reinicialització amb POST
            response_post = requests.post(f"{self.base_url}/reset/")
            self.assertEqual(response_post.status_code, 200)
            
            data = response_post.json()
            self.assertIn("message", data)
            
            print("✓ Test de reinicialització completat amb èxit")
            print(f"  Missatge: {data['message']}")
            
        except Exception as e:
            print(f"✗ Error en el test de reinicialització: {str(e)}")
            raise

if __name__ == '__main__':
    # Configurar el format de sortida dels tests
    unittest.main(verbosity=2) 