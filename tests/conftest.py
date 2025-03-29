import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.db.session import get_db
from app.models.case import Base

# Crear base de dades de test
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/test_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def db_engine():
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

# Fixtures per dades de test
@pytest.fixture
def test_case_data():
    return {
        "cas": "TEST001",
        "edat": "45",
        "genere": "H",
        "c_alta": "1",
        "periode": "2024",
        "servei": "MED",
        "motiuingres": "Test motiu ingrés",
        "malaltiaactual": "Test malaltia actual",
        "exploracio": "Test exploració",
        "provescomplementariesing": "Test proves ingrés",
        "provescomplementaries": "Test proves complementàries",
        "evolucio": "Test evolució",
        "antecedents": "Test antecedents",
        "cursclinic": "Test curs clínic"
    }

@pytest.fixture
def test_train_data(test_case_data):
    data = test_case_data.copy()
    data["dx_revisat"] = ["A011", "B022"]  # Format llista
    return data

@pytest.fixture
def test_validate_data(test_train_data):
    return test_train_data.copy()

@pytest.fixture
def test_evaluate_data(test_case_data):
    data = test_case_data.copy()
    data["dx_revisat"] = ["A419"]  # Format llista
    return data 