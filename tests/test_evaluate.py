import pytest
from fastapi import status

def test_evaluate_endpoint(client, test_case_data):
    """Test del endpoint d'avaluació"""
    # Afegir dx_revisat al cas de test
    test_case_data['dx_revisat'] = ['A419']
    
    response = client.post("/json/evaluate/", json=test_case_data)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    # Verificar estructura de la resposta
    assert "status" in data
    assert "message" in data
    assert "cas" in data
    assert "prediccions" in data
    assert "metrics" in data
    
    # Verificar mètriques
    metrics = data['metrics']
    assert "accuracy" in metrics
    assert "predicted_diagnostic" in metrics
    assert "actual_diagnostic" in metrics
    
    # Verificar que s'ha guardat a la base de dades
    db_response = client.get("/bd/evaluate/")
    assert db_response.status_code == status.HTTP_200_OK
    db_data = db_response.json()
    assert db_data["processed_cases"] > 0

def test_evaluate_invalid_data(client):
    """Test del endpoint d'avaluació amb dades invàlides"""
    invalid_data = {
        "cas": "",  # Cas buit
        "edat": "45",
        "genere": "H"
    }
    
    response = client.post("/json/evaluate/", json=invalid_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_evaluate_duplicate_case(client, test_case_data):
    """Test del endpoint d'avaluació amb cas duplicat"""
    # Afegir dx_revisat al cas de test
    test_case_data['dx_revisat'] = ['A419']
    
    # Primer intent
    response1 = client.post("/json/evaluate/", json=test_case_data)
    assert response1.status_code == status.HTTP_200_OK
    
    # Segon intent amb el mateix cas
    response2 = client.post("/json/evaluate/", json=test_case_data)
    assert response2.status_code == status.HTTP_200_OK  # Hauria de permetre duplicats

def test_evaluate_without_dx_revisat(client, test_case_data):
    """Test del endpoint d'avaluació sense dx_revisat"""
    response = client.post("/json/evaluate/", json=test_case_data)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    # Verificar que les mètriques són 0 sense dx_revisat
    metrics = data['metrics']
    assert metrics['accuracy'] == 0.0
    assert metrics['actual_diagnostic'] is None
    assert metrics['predicted_diagnostic'] is not None

def test_evaluate_accuracy_calculation(client, test_case_data):
    """Test del càlcul de precisió en l'avaluació"""
    # Cas amb predicció correcta
    test_case_data['dx_revisat'] = ['A419']
    response1 = client.post("/json/evaluate/", json=test_case_data)
    data1 = response1.json()
    assert data1['metrics']['accuracy'] == 1.0
    
    # Cas amb predicció incorrecta
    test_case_data['dx_revisat'] = ['J189']
    response2 = client.post("/json/evaluate/", json=test_case_data)
    data2 = response2.json()
    assert data2['metrics']['accuracy'] == 0.0 