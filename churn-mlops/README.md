# Pipeline MLOps Completo — Previsão de Churn de Clientes

> **Stack:** Python · Scikit-Learn · FastAPI · Docker · GitHub Actions  
> **Caso de uso:** Classificação binária — o cliente vai cancelar o contrato?

---

## Índice

1. [Estrutura do Projeto](#1-estrutura-do-projeto)
2. [Treinamento e Prevenção de Data-Leakage](#2-treinamento-e-prevenção-de-data-leakage)
3. [API de Inferência com FastAPI](#3-api-de-inferência-com-fastapi)
4. [Containerização com Docker](#4-containerização-com-docker)
5. [CI/CD com GitHub Actions](#5-cicd-com-github-actions)
6. [Como Executar Localmente](#6-como-executar-localmente)
7. [Referências e Conceitos-chave](#7-referências-e-conceitos-chave)

---

## 1. Estrutura do Projeto

Uma boa estrutura de pastas separa **responsabilidades** e facilita a automação. Cada diretório tem um papel claro no pipeline de MLOps:

```
churn-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml              ← Pipeline de CI/CD (GitHub Actions)
│
├── app/
│   └── app.py                  ← API de inferência (FastAPI)
│
├── models/                     ← Repositório de artefatos (gerados pelo treino)
│   ├── model.pkl               ← Pipeline scikit-learn serializado
│   └── metadata.json           ← Rastreabilidade: versão, métricas, features
│
├── tests/
│   └── test_model.py           ← Testes de validação: modelo + API
│
├── train.py                    ← Script de treinamento
├── Dockerfile                  ← Containerização da API + modelo
├── requirements.txt            ← Dependências com versões fixadas
├── .gitignore
└── README.md
```

### Por que essa separação é importante?

| Diretório | Responsabilidade | Quem consome |
|---|---|---|
| `app/` | Código de serving (produção) | Docker, Kubernetes |
| `models/` | Artefatos versionados | API, CI/CD, Model Registry |
| `tests/` | Validação automática | GitHub Actions |
| `.github/workflows/` | Automação de pipeline | GitHub Actions Runner |

---

## 2. Treinamento e Prevenção de Data-Leakage

### O que é Data-Leakage?

**Data-leakage** ocorre quando informações que só estariam disponíveis no futuro (ou no conjunto de teste) "vazam" para o treinamento, produzindo métricas artificialmente otimistas. O modelo parece excelente no experimento, mas falha em produção.

**Causa mais comum:** Aplicar `StandardScaler.fit()` em todos os dados *antes* de dividir em treino/teste:

```python
# ❌ ERRADO — Data-Leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # Aprende média/std de TODOS os dados
X_train, X_test = train_test_split(X_scaled)  # Teste "contaminou" o treino
```

```python
# ✅ CORRETO — Com sklearn Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),           # fit() só nos dados de TREINO
    ("classifier", GradientBoostingClassifier()),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
pipeline.fit(X_train, y_train)              # Scaler aprende SÓ com X_train
pipeline.predict(X_test)                    # Aplica transformação aprendida
```

### Como o `Pipeline` resolve o problema

```
pipeline.fit(X_train, y_train)
   │
   ├─ Step 1: scaler.fit_transform(X_train)   ← aprende média/std de X_train APENAS
   └─ Step 2: classifier.fit(X_train_scaled)  ← treina no dado já escalado

pipeline.predict(X_test)
   │
   ├─ Step 1: scaler.transform(X_test)        ← usa média/std aprendida no treino
   └─ Step 2: classifier.predict(X_test_scaled)
```

O artefato `model.pkl` contém **ambos os passos encapsulados**. A API de produção aplica automaticamente a mesma transformação aprendida no treino, sem risco de drift.

### Executar o treinamento

```bash
python train.py
```

**Saída esperada:**
```
2026-02-28 [INFO] Gerando 5000 amostras sintéticas de clientes...
2026-02-28 [INFO] Dataset gerado. Taxa de churn: 25.3%
2026-02-28 [INFO] Divisão concluída — Treino: 4000 amostras | Teste: 1000 amostras
2026-02-28 [INFO] Executando validação cruzada (5-fold estratificado)...
2026-02-28 [INFO] ROC-AUC (CV) — Média: 0.8821 ± 0.0123
2026-02-28 [INFO] Treinando pipeline no conjunto completo de treino...
2026-02-28 [INFO] ROC-AUC (Teste): 0.8934

RELATÓRIO DE CLASSIFICAÇÃO — CONJUNTO DE TESTE
============================================================
              precision    recall  f1-score   support
Não Churnou       0.91      0.95      0.93       753
    Churnou       0.82      0.71      0.76       247
============================================================

2026-02-28 [INFO] Artefato salvo em 'models/model.pkl' (285.3 KB)
2026-02-28 [INFO] Metadados salvos em 'models/metadata.json'
```

---

## 3. API de Inferência com FastAPI

### Arquitetura do Serving

```
Cliente HTTP
    │  POST /predict  { "tenure_months": 6, ... }
    ▼
FastAPI (app/app.py)
    │
    ├─ Validação Pydantic (schema automático)
    │
    ├─ pipeline.predict_proba(X)
    │     ├─ scaler.transform(X)    ← mesmos parâmetros do treino
    │     └─ classifier.predict_proba(X_scaled)
    │
    └─ Resposta JSON:
       {
         "churn_prediction": 1,
         "churn_probability": 0.7823,
         "risk_level": "ALTO",
         "model_version": "2026-02-28T..."
       }
```

### Executar a API localmente

```bash
uvicorn app.app:app --reload --port 8000
```

Acesse a documentação interativa em: http://localhost:8000/docs

### Exemplo de requisição com `curl`

```bash
# Cliente de alto risco
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 3,
    "monthly_charges": 450.00,
    "total_charges": 1350.00,
    "num_products": 1,
    "support_calls": 12,
    "payment_delay_days": 25,
    "age": 22,
    "satisfaction_score": 1.5
  }' | python -m json.tool
```

**Resposta:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8234,
  "risk_level": "ALTO",
  "model_version": "2026-02-28T10:30:00Z"
}
```

### Endpoints disponíveis

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET`  | `/health` | Liveness/Readiness probe (Kubernetes) |
| `GET`  | `/metrics` | Metadados e versão do modelo |
| `POST` | `/predict` | Inferência de churn |
| `GET`  | `/docs` | Documentação Swagger automática |

---

## 4. Containerização com Docker

### Por que o Docker resolve o problema de reprodutibilidade?

```
Sem Docker:
  Dev (Python 3.11, sklearn 1.5.1) ──✅──▶ Funciona
  Prod (Python 3.9,  sklearn 1.3.0) ──❌──▶ Erro de deserialização do .pkl

Com Docker:
  Dev  ──▶ docker build ──▶ [Python 3.11 + sklearn 1.5.1 + model.pkl]
  Prod ──▶ docker run   ──▶ [Python 3.11 + sklearn 1.5.1 + model.pkl] ✅
```

O container empacota **tudo**: runtime, dependências, código e artefato. Qualquer máquina com Docker executa o mesmo ambiente.

### Build e execução local

```bash
# 1. Treinar o modelo (gera models/model.pkl)
python train.py

# 2. Build da imagem
docker build -t churn-api:latest .

# 3. Executar o container
docker run -p 8000:8000 churn-api:latest

# 4. Testar
curl http://localhost:8000/health
```

### Estratégia de cache de layers do Dockerfile

```dockerfile
# requirements.txt copiado ANTES do código-fonte
COPY requirements.txt .
RUN pip install -r requirements.txt  # ← Layer cacheada se deps não mudaram

COPY app/ app/                        # ← Invalida cache só quando o código mudar
COPY models/ models/
```

Se apenas o código mudar, o Docker reutiliza a layer `pip install`, economizando 2-3 minutos no CI.

---

## 5. CI/CD com GitHub Actions

### Fluxo do pipeline

```
git push
    │
    ▼
┌──────────────────────────────────────────────┐
│  Job: test                                   │
│                                              │
│  1. Checkout do código                       │
│  2. Setup Python 3.11 + cache do pip        │
│  3. pip install -r requirements.txt          │
│  4. python train.py    ← retreina toda vez  │
│  5. Valida model.pkl   ← deve existir       │
│  6. pytest tests/      ← testes automáticos │
│  7. Upload model.pkl   ← artefato GitHub    │
└──────────────┬───────────────────────────────┘
               │ (somente se tests passarem)
               ▼
┌──────────────────────────────────────────────┐
│  Job: docker                                 │
│                                              │
│  1. Download model.pkl do Job anterior       │
│  2. docker build                             │
│  3. Smoke test: container UP + /predict ✓   │
│  4. Relatório de tamanho da imagem           │
└──────────────────────────────────────────────┘
```

### O que cada teste do CI valida?

| Teste | O que previne |
|-------|--------------|
| Existência do `model.pkl` | Deploy sem modelo treinado |
| `isinstance(pipeline, Pipeline)` | Artefato corrompido ou versão errada |
| Presença do step `scaler` | Remoção acidental do pré-processamento |
| ROC-AUC ≥ 0.70 | Deploy de modelo abaixo da qualidade mínima |
| Features correspondem às esperadas | Mudança silenciosa na ordem/nome das features |
| `POST /predict` retorna 200 | Regressão na API |
| Saída determinística | Inferência com componente aleatório |

### Ativando o Push para Docker Hub

No arquivo `.github/workflows/ci.yml`, descomente a seção de push e configure os secrets no repositório:

```
GitHub Repository → Settings → Secrets and Variables → Actions
  DOCKERHUB_USERNAME: seu_usuario
  DOCKERHUB_TOKEN:    seu_token_de_acesso
```

---

## 6. Como Executar Localmente

### Pré-requisitos

- Python 3.11+
- Docker Desktop (para testes com container)

### Passo a passo

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/churn-mlops.git
cd churn-mlops

# 2. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Treinar o modelo
python train.py

# 5. Executar os testes
pytest tests/ -v

# 6. Iniciar a API
uvicorn app.app:app --reload --port 8000

# 7. (Opcional) Rodar via Docker
docker build -t churn-api:latest .
docker run -p 8000:8000 churn-api:latest
```

---

## 7. Referências e Conceitos-chave

### Data-Leakage
A causa mais comum de modelos que performam bem no experimento mas falham em produção. Ocorre quando informações do futuro ou do conjunto de teste contaminam o treinamento. A `Pipeline` do scikit-learn é a principal ferramenta de prevenção para pipelines tabulares.

### CI/CD para ML
Diferente do CI/CD tradicional, um pipeline de ML precisa validar não só o código, mas também o **artefato** (modelo) — suas métricas, estrutura e comportamento na inferência. O GitHub Actions automatiza o retreinamento e a validação a cada push.

### Reprodutibilidade com Docker
O container empacota o ambiente completo: runtime Python, bibliotecas com versões fixadas e o artefato do modelo. Elimina o problema "funciona na minha máquina" e garante que o mesmo código rode de forma idêntica em desenvolvimento, CI e produção.

### Observabilidade
A API expõe `/health` (probe de saúde para orquestradores), `/metrics` (metadados do modelo) e headers de latência (`X-Process-Time-Ms`). Em produção, esses dados alimentariam um painel de monitoramento para detectar degradação de performance (concept drift).

### Fontes consultadas
- [A Beginner's Guide to CI/CD for ML Models](https://pr-peri.github.io/blogpost/2026/01/10/ci-cd-ml-models.html)
- Google for Developers — Pipelines e Sistemas de ML de Produção
- AWS Well-Architected Framework — Machine Learning Lens
- Wikipedia — MLOps: Automação CI/CD para Machine Learning
- Databricks — Pipelines de ML com scikit-learn
- Datarisk — Como evitar data-leakage em ML
