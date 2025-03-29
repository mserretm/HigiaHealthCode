import pytest
from fastapi import status

def test_train_endpoint(client, test_train_data):
    """Test del endpoint d'entrenament"""
    response = client.post("/json/train/", json=test_train_data)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    # Verificar estructura de la resposta
    assert "status" in data
    assert "message" in data
    
    # Verificar que s'ha guardat a la base de dades
    db_response = client.get("/bd/train/")
    assert db_response.status_code == status.HTTP_200_OK
    db_data = db_response.json()
    assert db_data["processed_cases"] > 0

def test_train_invalid_data(client):
    """Test del endpoint d'entrenament amb dades invàlides"""
    invalid_data = {
        "cas": "",  # Cas buit
        "dx_revisat": "",  # Diagnòstic buit
        "edat": "45",
        "genere": "H"
    }
    
    response = client.post("/json/train/", json=invalid_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_train_duplicate_case(client, test_train_data):
    """Test del endpoint d'entrenament amb cas duplicat"""
    # Primer intent
    response1 = client.post("/json/train/", json=test_train_data)
    assert response1.status_code == status.HTTP_200_OK
    
    # Segon intent amb el mateix cas
    response2 = client.post("/json/train/", json=test_train_data)
    assert response2.status_code == status.HTTP_200_OK  # Hauria de permetre duplicats

def test_train_multiple_diagnostics(client, test_case_data):
    """Test del endpoint d'entrenament amb múltiples diagnòstics"""
    data = test_case_data.copy()
    data["dx_revisat"] = ["A011", "B022", "C033"]
    
    response = client.post("/json/train/", json=data)
    assert response.status_code == status.HTTP_200_OK

def test_train_dx_revisat_formats(client, test_case_data):
    """Test del endpoint d'entrenament amb diferents formats de dx_revisat"""
    # Format llista
    data_list = test_case_data.copy()
    data_list["dx_revisat"] = ["A011"]
    response1 = client.post("/json/train/", json=data_list)
    assert response1.status_code == status.HTTP_200_OK
    
    # Format string amb separador
    data_str = test_case_data.copy()
    data_str["dx_revisat"] = "A011|B022"
    response2 = client.post("/json/train/", json=data_str)
    assert response2.status_code == status.HTTP_200_OK

def test_train_invalid_dx_revisat(client, test_case_data):
    """Test del endpoint d'entrenament amb diagnòstics invàlids"""
    # Diagnòstic buit
    data_empty = test_case_data.copy()
    data_empty["dx_revisat"] = []
    response1 = client.post("/json/train/", json=data_empty)
    assert response1.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    # Diagnòstic amb format incorrecte
    data_invalid = test_case_data.copy()
    data_invalid["dx_revisat"] = "invalid_diagnostic"
    response2 = client.post("/json/train/", json=data_invalid)
    assert response2.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY 