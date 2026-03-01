"""
=============================================================================
PASSO 3: API DE INFERÊNCIA — FastAPI
=============================================================================

Expõe o modelo de Churn via HTTP para que outros sistemas possam consumir
previsões em tempo real (serving online).

PRINCÍPIOS APLICADOS:
- O pipeline (scaler + model) é carregado UMA ÚNICA VEZ no startup, evitando
  leitura de disco a cada requisição.
- O schema Pydantic valida e documenta os dados de entrada automaticamente,
  gerando a documentação OpenAPI (Swagger) sem esforço extra.
- O endpoint /health permite que orquestradores (Kubernetes, ECS) verifiquem
  a saúde do container antes de rotear tráfego.
- O endpoint /metrics expõe metadados do modelo para rastreabilidade.

Estrutura de endpoints:
  GET  /health    → Liveness/Readiness probe
  GET  /metrics   → Metadados e versão do modelo
  POST /predict   → Inferência de churn

Referências:
- Pr-Peri Blog: CI/CD for ML Models — serving com FastAPI + Docker
- Google for Developers: Monitoramento de pipelines de ML em produção
- AWS Well-Architected ML Lens: Design seguro e reprodutível para serving
=============================================================================
"""

import os
import json
import logging
import pickle
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Configuração de Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("churn_api")

# ---------------------------------------------------------------------------
# Caminhos dos artefatos (relativos ao container ou ambiente local)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "models/metadata.json")

# ---------------------------------------------------------------------------
# Estado global da aplicação — carregado no startup
# ---------------------------------------------------------------------------
app_state: dict = {
    "pipeline": None,
    "metadata": {},
    "request_count": 0,
    "startup_time": None,
}


# ---------------------------------------------------------------------------
# Lifespan — carrega artefatos uma única vez ao iniciar o servidor
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.

    Carrega o pipeline e os metadados do modelo durante o startup.
    Em um ambiente Kubernetes, isso ocorre ANTES do readiness probe passar,
    garantindo que o container só receba tráfego quando estiver pronto.
    """
    # ── STARTUP ──────────────────────────────────────────────────────────
    logger.info("Iniciando carregamento dos artefatos do modelo...")
    startup_start = time.time()

    # Carrega o Pipeline treinado (scaler + classifier)
    if not os.path.exists(MODEL_PATH):
        logger.error("Artefato não encontrado: %s", MODEL_PATH)
        raise FileNotFoundError(
            f"Modelo não encontrado em '{MODEL_PATH}'. "
            "Execute train.py antes de iniciar a API."
        )

    with open(MODEL_PATH, "rb") as f:
        app_state["pipeline"] = pickle.load(f)

    # Carrega metadados (opcional — não impede o startup se ausente)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            app_state["metadata"] = json.load(f)
        logger.info(
            "Modelo carregado: %s | ROC-AUC: %s",
            app_state["metadata"].get("model_type", "?"),
            app_state["metadata"].get("roc_auc_test", "?"),
        )
    else:
        logger.warning("Arquivo de metadados não encontrado: %s", METADATA_PATH)

    app_state["startup_time"] = time.time() - startup_start
    app_state["request_count"] = 0
    logger.info("API pronta! Tempo de startup: %.2fs", app_state["startup_time"])

    yield  # ← A aplicação fica ativa aqui

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    logger.info(
        "Encerrando API. Total de requisições processadas: %d",
        app_state["request_count"],
    )
    app_state["pipeline"] = None


# ---------------------------------------------------------------------------
# Instância principal do FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Churn Prediction API",
    description=(
        "API de inferência para o modelo de Previsão de Churn de Clientes. "
        "Retorna a probabilidade e o rótulo binário de churn para um cliente."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware CORS — configure os origins permitidos em produção
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Em produção: substitua por domínios específicos
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware de métricas simples — rastreia latência por requisição
# ---------------------------------------------------------------------------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000

    if request.url.path == "/predict":
        app_state["request_count"] += 1
        logger.info(
            "POST /predict — status=%d | latência=%.1fms | req_total=%d",
            response.status_code,
            latency_ms,
            app_state["request_count"],
        )

    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.1f}"
    return response


# ---------------------------------------------------------------------------
# Schemas Pydantic — validação e documentação automática
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """
    Dados de entrada para previsão de churn de um cliente.

    Todos os campos correspondem exatamente às features usadas no treino
    (train.py → FEATURE_NAMES). A ordem importa para a inferência.
    """

    tenure_months: int = Field(
        ..., ge=1, le=72,
        description="Tempo de contrato em meses (1–72).",
        example=12,
    )
    monthly_charges: float = Field(
        ..., ge=50.0, le=500.0,
        description="Valor da mensalidade em R$ (50–500).",
        example=199.90,
    )
    total_charges: float = Field(
        ..., ge=100.0, le=20000.0,
        description="Total pago pelo cliente em R$ (100–20000).",
        example=2398.80,
    )
    num_products: int = Field(
        ..., ge=1, le=6,
        description="Número de produtos contratados (1–6).",
        example=2,
    )
    support_calls: int = Field(
        ..., ge=0, le=15,
        description="Chamados ao suporte nos últimos 6 meses (0–15).",
        example=3,
    )
    payment_delay_days: int = Field(
        ..., ge=0, le=30,
        description="Atraso médio no pagamento em dias (0–30).",
        example=5,
    )
    age: int = Field(
        ..., ge=18, le=80,
        description="Idade do cliente em anos (18–80).",
        example=35,
    )
    satisfaction_score: float = Field(
        ..., ge=0.0, le=10.0,
        description="Pontuação de satisfação do cliente (0–10).",
        example=6.5,
    )

    @field_validator("satisfaction_score")
    @classmethod
    def round_satisfaction(cls, v: float) -> float:
        """Arredonda para 1 casa decimal para consistência."""
        return round(v, 1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 12,
                "monthly_charges": 199.90,
                "total_charges": 2398.80,
                "num_products": 2,
                "support_calls": 3,
                "payment_delay_days": 5,
                "age": 35,
                "satisfaction_score": 6.5,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Resposta do endpoint /predict."""

    churn_prediction: int = Field(
        ..., description="Rótulo binário: 1 = Churn, 0 = Não Churn."
    )
    churn_probability: float = Field(
        ..., description="Probabilidade de churn (0.0–1.0)."
    )
    risk_level: str = Field(
        ..., description="Nível de risco: BAIXO | MÉDIO | ALTO."
    )
    model_version: Optional[str] = Field(
        None, description="Identificador de versão do modelo."
    )


class HealthResponse(BaseModel):
    """Resposta do endpoint /health."""

    status: str
    model_loaded: bool
    request_count: int
    uptime_seconds: Optional[float]


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Observabilidade"],
    summary="Health check — liveness/readiness probe",
)
async def health_check():
    """
    Verifica se a API está saudável e o modelo carregado.

    Utilizado por Kubernetes, ECS e load balancers para determinar se
    o container está pronto para receber tráfego.
    """
    model_loaded = app_state["pipeline"] is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        request_count=app_state["request_count"],
        uptime_seconds=round(app_state["startup_time"] or 0.0, 2),
    )


@app.get(
    "/metrics",
    tags=["Observabilidade"],
    summary="Metadados e informações do modelo",
)
async def model_metrics():
    """
    Retorna metadados do modelo carregado.

    Em produção, essas informações alimentariam um painel de monitoramento
    (ex: Grafana, CloudWatch) para rastrear versão, ROC-AUC de treino
    e características usadas.
    """
    if not app_state["metadata"]:
        return {"message": "Metadados não disponíveis."}
    return {
        **app_state["metadata"],
        "total_requests": app_state["request_count"],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inferência"],
    summary="Prevê o risco de churn de um cliente",
    status_code=200,
)
async def predict_churn(customer: CustomerFeatures):
    """
    Recebe os dados de um cliente e retorna a previsão de churn.

    **Fluxo interno:**
    1. Converte o payload JSON em array numpy na ordem correta das features.
    2. Aplica `pipeline.predict_proba()` — o scaler interno transforma
       automaticamente com os parâmetros aprendidos no treino.
    3. Classifica o nível de risco: BAIXO (<30%), MÉDIO (30–60%), ALTO (>60%).

    **Nota sobre data-leakage em produção:**
    Como o scaler está encapsulado no Pipeline, não há risco de usar
    estatísticas erradas na inferência — o artefato `.pkl` já contém
    toda a informação necessária.
    """
    pipeline = app_state["pipeline"]

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não disponível. Tente novamente em instantes.",
        )

    # Monta o array de features na mesma ordem do treinamento (CRÍTICO)
    feature_order = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_products",
        "support_calls",
        "payment_delay_days",
        "age",
        "satisfaction_score",
    ]

    try:
        X = np.array(
            [[getattr(customer, feat) for feat in feature_order]],
            dtype=np.float64,
        )

        # O Pipeline aplica scaler → classifier internamente
        churn_proba = float(pipeline.predict_proba(X)[0, 1])
        churn_label = int(pipeline.predict(X)[0])

        # Classificação de risco baseada na probabilidade
        if churn_proba < 0.30:
            risk_level = "BAIXO"
        elif churn_proba < 0.60:
            risk_level = "MÉDIO"
        else:
            risk_level = "ALTO"

        model_version = app_state["metadata"].get("trained_at", "unknown")

        return PredictionResponse(
            churn_prediction=churn_label,
            churn_probability=round(churn_proba, 4),
            risk_level=risk_level,
            model_version=model_version,
        )

    except Exception as exc:
        logger.exception("Erro durante a inferência: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno durante a inferência: {str(exc)}",
        )


@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """Redireciona para a documentação Swagger."""
    return JSONResponse(
        content={
            "message": "Churn Prediction API v1.0.0",
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict",
        }
    )
