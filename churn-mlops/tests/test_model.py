"""
=============================================================================
TESTES DE VALIDAÇÃO DO MODELO E DA API — Churn Prediction
=============================================================================

Estes testes são executados pelo GitHub Actions no Job 'test' a cada push.
Têm três responsabilidades:

1. Validar o ARTEFATO do modelo (carregamento, estrutura do pipeline,
   features, sanidade das previsões).
2. Validar a API de INFERÊNCIA (endpoints, schema, lógica de risco).
3. Prevenir REGRESSÕES ao garantir que o modelo respeite thresholds mínimos.

Por que testar modelos de ML?
   - Um modelo pode ser salvo com versão diferente de scikit-learn e gerar
     erros silenciosos na desserialização.
   - O Pipeline pode estar sem um dos steps (ex: scaler removido).
   - As features podem ter mudado de ordem ou nome.
   - A API pode retornar status 200 mas com payload incorreto.

Referências:
   - Google for Developers: Automação para testar e implantar modelos
   - AWS Automated ML Pipeline: Validação de artefatos no pipeline de CI
=============================================================================
"""

import os
import sys
import pickle
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Garante que o módulo app pode ser encontrado a partir da raiz do projeto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "models/metadata.json")

FEATURE_NAMES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_calls",
    "payment_delay_days",
    "age",
    "satisfaction_score",
]

# Payload de cliente de ALTO RISCO (muitos chamados, baixa satisfação, atraso)
HIGH_RISK_CUSTOMER = {
    "tenure_months": 3,
    "monthly_charges": 450.00,
    "total_charges": 1350.00,
    "num_products": 1,
    "support_calls": 12,
    "payment_delay_days": 25,
    "age": 22,
    "satisfaction_score": 1.5,
}

# Payload de cliente de BAIXO RISCO (longa duração, satisfação alta)
LOW_RISK_CUSTOMER = {
    "tenure_months": 60,
    "monthly_charges": 120.00,
    "total_charges": 7200.00,
    "num_products": 5,
    "support_calls": 0,
    "payment_delay_days": 0,
    "age": 55,
    "satisfaction_score": 9.5,
}


@pytest.fixture(scope="session")
def loaded_pipeline():
    """
    Carrega o pipeline do disco uma única vez para todos os testes da sessão.
    Falha imediatamente se o artefato não existir — evita falsos negativos.
    """
    assert os.path.exists(MODEL_PATH), (
        f"Artefato não encontrado: '{MODEL_PATH}'. "
        "Execute train.py antes de rodar os testes."
    )
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


@pytest.fixture(scope="session")
def api_client(loaded_pipeline):
    """
    Cria um TestClient do FastAPI com o modelo já carregado no app_state.
    Usa a fixture loaded_pipeline para garantir que o modelo existe antes.
    """
    from app.app import app, app_state
    app_state["pipeline"] = loaded_pipeline

    # Carrega metadados se disponíveis
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            app_state["metadata"] = json.load(f)

    with TestClient(app) as client:
        yield client


# =============================================================================
# BLOCO 1: Testes do Artefato do Modelo
# =============================================================================

class TestModelArtifact:
    """Valida a integridade e estrutura do artefato model.pkl."""

    def test_model_file_exists(self):
        """O artefato deve existir no caminho configurado."""
        assert os.path.exists(MODEL_PATH), f"Modelo não encontrado em: {MODEL_PATH}"

    def test_model_file_is_not_empty(self):
        """O arquivo não pode estar vazio (ex: upload truncado)."""
        size = os.path.getsize(MODEL_PATH)
        assert size > 1024, f"Arquivo suspeito — muito pequeno: {size} bytes"

    def test_model_loads_without_error(self, loaded_pipeline):
        """O pickle.load() deve funcionar sem exceções."""
        assert loaded_pipeline is not None

    def test_model_is_sklearn_pipeline(self, loaded_pipeline):
        """O artefato deve ser um objeto sklearn.pipeline.Pipeline."""
        from sklearn.pipeline import Pipeline
        assert isinstance(loaded_pipeline, Pipeline), (
            f"Esperado Pipeline, encontrado: {type(loaded_pipeline)}"
        )

    def test_pipeline_has_scaler_step(self, loaded_pipeline):
        """O pipeline deve conter o passo 'scaler' para evitar data-leakage."""
        assert "scaler" in loaded_pipeline.named_steps, (
            "Passo 'scaler' não encontrado no pipeline. "
            "Isso pode indicar data-leakage na inferência."
        )

    def test_pipeline_has_classifier_step(self, loaded_pipeline):
        """O pipeline deve conter o passo 'classifier'."""
        assert "classifier" in loaded_pipeline.named_steps, (
            "Passo 'classifier' não encontrado no pipeline."
        )

    def test_pipeline_accepts_correct_number_of_features(self, loaded_pipeline):
        """
        O pipeline deve aceitar exatamente o número de features do treino.
        Previne erros silenciosos quando novas features são adicionadas.
        """
        n_features = len(FEATURE_NAMES)
        X_dummy = np.zeros((1, n_features))
        try:
            loaded_pipeline.predict(X_dummy)
        except Exception as e:
            pytest.fail(
                f"Pipeline falhou com {n_features} features. Erro: {e}"
            )

    def test_predict_returns_binary_label(self, loaded_pipeline):
        """predict() deve retornar apenas 0 ou 1."""
        X = np.array([[12, 199.9, 2398.8, 2, 3, 5, 35, 6.5]])
        pred = loaded_pipeline.predict(X)
        assert pred[0] in (0, 1), f"Rótulo inválido: {pred[0]}"

    def test_predict_proba_returns_valid_probability(self, loaded_pipeline):
        """predict_proba() deve retornar probabilidades entre 0 e 1."""
        X = np.array([[12, 199.9, 2398.8, 2, 3, 5, 35, 6.5]])
        proba = loaded_pipeline.predict_proba(X)

        assert proba.shape == (1, 2), f"Shape inesperado: {proba.shape}"
        assert 0.0 <= proba[0, 1] <= 1.0, f"Probabilidade fora do range: {proba[0, 1]}"
        assert abs(proba[0, 0] + proba[0, 1] - 1.0) < 1e-6, (
            "As probabilidades das classes não somam 1.0"
        )

    def test_batch_inference(self, loaded_pipeline):
        """O pipeline deve suportar inferência em lote sem erros."""
        n_samples = 50
        rng = np.random.default_rng(0)
        X_batch = rng.random((n_samples, len(FEATURE_NAMES)))
        preds = loaded_pipeline.predict(X_batch)
        assert len(preds) == n_samples


# =============================================================================
# BLOCO 2: Testes de Metadados
# =============================================================================

class TestModelMetadata:
    """Valida o arquivo de metadados para rastreabilidade."""

    def test_metadata_file_exists(self):
        """O arquivo de metadados deve existir."""
        assert os.path.exists(METADATA_PATH), (
            f"Metadados não encontrados em: {METADATA_PATH}"
        )

    def test_metadata_has_required_keys(self):
        """Os metadados devem conter campos obrigatórios de rastreabilidade."""
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        required_keys = {"model_type", "features", "trained_at", "roc_auc_test"}
        missing = required_keys - set(meta.keys())
        assert not missing, f"Campos ausentes nos metadados: {missing}"

    def test_metadata_roc_auc_above_threshold(self):
        """
        O ROC-AUC deve ser superior ao threshold mínimo aceitável.

        Este teste BLOQUEIA o deploy se o modelo for pior que o esperado.
        Em produção, compare também com a versão em produção atual (champion).
        """
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        roc_auc = meta.get("roc_auc_test", 0.0)
        threshold = 0.70  # Threshold mínimo — ajuste conforme o negócio
        assert roc_auc >= threshold, (
            f"ROC-AUC abaixo do threshold mínimo: {roc_auc:.4f} < {threshold}. "
            "O modelo não passou no critério de qualidade."
        )

    def test_metadata_features_match_expected(self):
        """As features devem corresponder exatamente às esperadas pelo código."""
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta.get("features") == FEATURE_NAMES, (
            f"Features divergentes.\n"
            f"Esperado: {FEATURE_NAMES}\n"
            f"Encontrado: {meta.get('features')}"
        )


# =============================================================================
# BLOCO 3: Testes da API FastAPI
# =============================================================================

class TestAPIEndpoints:
    """Valida o comportamento dos endpoints da API."""

    def test_root_endpoint_returns_200(self, api_client):
        """GET / deve retornar 200."""
        response = api_client.get("/")
        assert response.status_code == 200

    def test_health_endpoint_returns_healthy(self, api_client):
        """GET /health deve retornar status 'healthy' com modelo carregado."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_metrics_endpoint_returns_model_info(self, api_client):
        """GET /metrics deve retornar informações do modelo."""
        response = api_client.get("/metrics")
        assert response.status_code == 200

    def test_predict_valid_payload_returns_200(self, api_client):
        """POST /predict com payload válido deve retornar 200."""
        response = api_client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_schema(self, api_client):
        """A resposta do /predict deve conter os campos corretos."""
        response = api_client.post("/predict", json=LOW_RISK_CUSTOMER)
        data = response.json()

        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert data["churn_prediction"] in (0, 1)
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_level"] in ("BAIXO", "MÉDIO", "ALTO")

    def test_predict_missing_field_returns_422(self, api_client):
        """Payload incompleto deve retornar 422 (Unprocessable Entity)."""
        incomplete_payload = {"tenure_months": 12}  # Faltam 7 campos
        response = api_client.post("/predict", json=incomplete_payload)
        assert response.status_code == 422

    def test_predict_invalid_field_type_returns_422(self, api_client):
        """Campo com tipo errado deve retornar 422."""
        invalid_payload = {**LOW_RISK_CUSTOMER, "tenure_months": "doze"}
        response = api_client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_predict_out_of_range_value_returns_422(self, api_client):
        """Valor fora do range definido deve retornar 422."""
        out_of_range = {**LOW_RISK_CUSTOMER, "tenure_months": 9999}
        response = api_client.post("/predict", json=out_of_range)
        assert response.status_code == 422

    def test_predict_deterministic_output(self, api_client):
        """
        A mesma entrada deve sempre produzir a mesma saída.
        Verifica que o modelo é determinístico (sem aleatoriedade na inferência).
        """
        r1 = api_client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
        r2 = api_client.post("/predict", json=HIGH_RISK_CUSTOMER).json()
        assert r1["churn_prediction"] == r2["churn_prediction"]
        assert r1["churn_probability"] == r2["churn_probability"]

    def test_predict_risk_level_consistency(self, api_client):
        """
        O risk_level deve ser consistente com a probabilidade retornada.
        """
        response = api_client.post("/predict", json=HIGH_RISK_CUSTOMER)
        data = response.json()
        prob = data["churn_probability"]
        risk = data["risk_level"]

        if prob < 0.30:
            assert risk == "BAIXO"
        elif prob < 0.60:
            assert risk == "MÉDIO"
        else:
            assert risk == "ALTO"

    def test_process_time_header_present(self, api_client):
        """A resposta deve incluir o header X-Process-Time-Ms (observabilidade)."""
        response = api_client.post("/predict", json=LOW_RISK_CUSTOMER)
        assert "x-process-time-ms" in response.headers
