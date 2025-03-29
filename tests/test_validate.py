import pytest
from fastapi import status

def test_validate_endpoint(client, test_validate_data):
    """Test del endpoint de validació"""
    response = client.post("/json/validate/", json=test_validate_data)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    # Verificar estructura de la resposta
    assert "status" in data
    assert "message" in data
    assert "cas" in data
    assert "prediccions" in data
    
    # Verificar que s'ha guardat a la base de dades
    db_response = client.get("/bd/validate/")
    assert db_response.status_code == status.HTTP_200_OK
    db_data = db_response.json()
    assert db_data["processed_cases"] > 0

def test_validate_invalid_data(client):
    """Test del endpoint de validació amb dades invàlides"""
    invalid_data = {
        "cas": "",  # Cas buit
        "dx_revisat": "",  # Diagnòstic buit
        "edat": "45",
        "genere": "H"
    }
    
    response = client.post("/json/validate/", json=invalid_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_validate_duplicate_case(client, test_validate_data):
    """Test del endpoint de validació amb cas duplicat"""
    # Primer intent
    response1 = client.post("/json/validate/", json=test_validate_data)
    assert response1.status_code == status.HTTP_200_OK
    
    # Segon intent amb el mateix cas
    response2 = client.post("/json/validate/", json=test_validate_data)
    assert response2.status_code == status.HTTP_200_OK  # Hauria de permetre duplicats 