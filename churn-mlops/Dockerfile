# =============================================================================
# PASSO 4: DOCKERFILE OTIMIZADO — Churn Prediction API
# =============================================================================
#
# PROBLEMA RESOLVIDO PELO DOCKER — Reprodutibilidade:
#   "Funciona na minha máquina" é o pesadelo do MLOps. Um modelo treinado
#   em Python 3.11 com scikit-learn 1.5.1 pode se comportar diferente em
#   outro ambiente com versões distintas. O Docker encapsula:
#     - Versão exata do Python
#     - Versão exata de cada biblioteca (requirements.txt)
#     - O artefato do modelo (model.pkl)
#     - As variáveis de ambiente e configurações do servidor
#   Qualquer pessoa com Docker pode rodar `docker run` e obter
#   o mesmo comportamento, em qualquer máquina ou nuvem.
#
# OTIMIZAÇÕES APLICADAS:
#   1. Multi-stage NÃO utilizado aqui por simplicidade, mas recomendado
#      em produção para separar build e runtime.
#   2. Imagem base slim → menor superfície de ataque e pull mais rápido.
#   3. Layer caching: requirements.txt copiado ANTES do código-fonte.
#      Se o código mudar mas não as deps, o Docker reutiliza o cache
#      da camada `pip install`, economizando minutos no CI.
#   4. Usuário não-root (appuser) → boa prática de segurança.
#   5. PYTHONDONTWRITEBYTECODE e PYTHONUNBUFFERED → logs em tempo real
#      e imagem menor (sem arquivos .pyc).
#
# Referências:
#   - AWS Well-Architected ML Lens: segurança em containers para ML
#   - Pr-Peri Blog: Docker na camada de serving do pipeline de ML
# =============================================================================

# ── Estágio único (simplificado para fins didáticos) ──────────────────────
FROM python:3.11-slim

# --- Metadados -----------------------------------------------------------
LABEL maintainer="mlops-team@empresa.com"
LABEL description="Churn Prediction API — FastAPI + scikit-learn"
LABEL version="1.0.0"

# --- Variáveis de ambiente -----------------------------------------------
# Evita arquivos .pyc (reduz tamanho da imagem)
ENV PYTHONDONTWRITEBYTECODE=1
# Força stdout/stderr sem buffer (logs em tempo real no container)
ENV PYTHONUNBUFFERED=1
# Caminho do artefato — sobrescrito via docker run -e MODEL_PATH=...
ENV MODEL_PATH=models/model.pkl
ENV METADATA_PATH=models/metadata.json
# Porta da aplicação
ENV PORT=8000

# --- Diretório de trabalho -----------------------------------------------
WORKDIR /app

# --- Dependências do sistema (mínimas) -----------------------------------
# curl: usado pelo health check do Docker
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- [CAMADA CACHEÁVEL] Instala dependências Python ----------------------
# ESTRATÉGIA DE CACHE: copiamos SÓ o requirements.txt antes do código.
# Se apenas o código mudar, o Docker reutiliza esta camada do cache.
# Se as dependências mudarem, o cache é invalidado a partir daqui.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Copia o código-fonte e os artefatos ---------------------------------
# Copiado APÓS a instalação das deps para aproveitar o cache do Docker
COPY app/app.py app/app.py
COPY models/      models/

# --- Usuário não-root (segurança) ----------------------------------------
# Cria um usuário sem privilégios para executar a aplicação
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup /app

USER appuser

# --- Expose da porta ------------------------------------------------------
EXPOSE ${PORT}

# --- Health check nativo do Docker ----------------------------------------
# O Docker (e o Kubernetes) verificam este endpoint para saber se o
# container está saudável antes de rotear tráfego.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# --- Comando de inicialização ---------------------------------------------
# --host 0.0.0.0   → aceita conexões externas ao container
# --workers 1      → 1 worker para ML (modelo em memória compartilhada)
# --log-level info → logs estruturados para observabilidade
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
