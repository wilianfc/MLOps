# Pipeline MLOps Completo — Previsão de Churn de Clientes

> **Stack:** Python · Scikit-Learn · FastAPI · Docker · GitHub Actions · AWS Glue · AWS SageMaker  
> **Caso de uso:** Classificação binária — o cliente vai cancelar o contrato?

---

## Índice

1. [Estrutura do Projeto](#1-estrutura-do-projeto)
2. [Treinamento e Prevenção de Data-Leakage](#2-treinamento-e-prevenção-de-data-leakage)
3. [API de Inferência com FastAPI](#3-api-de-inferência-com-fastapi)
4. [Containerização com Docker](#4-containerização-com-docker)
5. [CI/CD com GitHub Actions](#5-cicd-com-github-actions)
6. [Como Executar Localmente](#6-como-executar-localmente)
7. [Clusterização — Segmentação de Clientes](#7-clusterização--segmentação-de-clientes)
8. [Variante AWS — Glue + SageMaker](#8-variante-aws--glue--sagemaker)
9. [Diagrama CRISP-DM — BPMN](#9-diagrama-crisp-dm--bpmn)
10. [Referências e Conceitos-chave](#10-referências-e-conceitos-chave)

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
│   ├── app.py                  ← API de inferência (FastAPI)
│   └── rag.py                  ← Motor de busca semântica (RAG — TF-IDF / FAISS)
│
├── models/                     ← Repositório de artefatos (gerados pelo treino)
│   ├── model.pkl               ← Pipeline scikit-learn serializado
│   ├── metadata.json           ← Rastreabilidade: versão, métricas, features
│   ├── cluster_report.json     ← Métricas e perfis dos clusters
│   ├── elbow_curve.png         ← Gráfico do Elbow Method
│   ├── silhouette_plot.png     ← Gráfico de Silhouette por cluster
│   └── pca_clusters.png        ← Projeção 2D dos clusters via PCA
│
├── tests/
│   ├── test_model.py           ← Testes de validação: modelo + API
│   └── test_clustering.py      ← Testes de validação: clusterização
│
├── aws/
│   ├── glue_etl_job.py             ← ETL Spark serverless (AWS Glue)
│   ├── train_sagemaker.py          ← Training Job (SageMaker)
│   ├── evaluate.py                 ← Avaliação formal p/ Model Registry
│   └── sagemaker_pipeline.py       ← Orquestrador do pipeline AWS
│
├── clustering_analysis.py     ← Segmentação não supervisionada (K-Means)
├── train.py                    ← Script de treinamento (local)
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
| `aws/` | Pipeline gerenciado na nuvem | SageMaker, Glue |
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
| `POST` | `/predict` | Inferência de churn (+ cluster + perfil RAG) |
| `POST` | `/rag/query` | Busca semântica sobre perfis de cluster |
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

# 4b. (Opcional) Gerar segmentação de clientes e perfis RAG
python clustering_analysis.py
# Gera: models/cluster_report.json, cluster_artifacts.pkl, profile_cards.json,
#        elbow_curve.png, silhouette_plot.png, pca_clusters.png

# 5. Executar os testes
pytest tests/ -v

# 6. Iniciar a API
uvicorn app.app:app --reload --port 8000

# 7. (Opcional) Rodar via Docker
docker build -t churn-api:latest .
docker run -p 8000:8000 churn-api:latest
```

---

## 7. Clusterização — Segmentação de Clientes

O script `clustering_analysis.py` complementa o pipeline supervisionado com **aprendizado não supervisionado**: em vez de prever churn, identifica *perfis naturais de comportamento* sem usar os rótulos.

### Usos no pipeline MLOps

| Etapa | Como a clusterização ajuda |
|---|---|
| **Exploração** | Entender grupos antes de treinar o classificador |
| **Feature Engineering** | Usar o `cluster_id` como feature adicional no `train.py` |
| **Monitoramento** | Detectar clientes que não se encaixam em nenhum perfil histórico (concept drift) |

### Métricas geradas

```
┌──────────────────────────────────────────────────────────┐
│  MÉTRICAS DE CLUSTERIZAÇÃO                               │
├──────────────────┬──────────────────┬───────────────────┤
│  INTERNAS         │  EXTERNAS         │  ESCOLHA DO K       │
│  (sem gabarito)   │  (com rótulo)    │                     │
│                  │                  │                     │
│  Silhouette       │  ARI             │  Elbow Method       │
│  Davies-Bouldin   │  Homogeneidade   │                     │
│  Calinski-        │  (≈ Precision)   │                     │
│  Harabasz         │  Completude      │                     │
│                  │  (≈ Recall)      │                     │
│                  │  V-Measure       │                     │
│                  │  (≈ F1-Score)    │                     │
└──────────────────┴──────────────────┴───────────────────┘
```

### Analogia com classificação

| Classificação | Clusterização | Interpretação |
|---|---|---|
| Precision | Homogeneidade | Clusters puros? Cada um contém só uma classe? |
| Recall | Completude | Classes concentradas? Todos os churners juntos? |
| F1-Score | V-Measure | Equilíbrio entre homogeneidade e completude |
| ROC-AUC | Silhouette | Qualidade geral de separação |
| Threshold | Elbow Method | Escolha do número ideal de clusters (K) |

### Arquivos gerados

```bash
python clustering_analysis.py
```

```
models/
├── cluster_report.json    ← Métricas + perfis de cada cluster
├── elbow_curve.png        ← Inércia por K (encontra o cotovelo)
├── silhouette_plot.png    ← Qualidade individual de cada ponto
└── pca_clusters.png       ← Projeção 2D para inspecção visual
```

### Exemplo de saída do `cluster_report.json`

```json
{
  "chosen_k": 2,
  "metrics": {
    "internal": {
      "silhouette_score": 0.6821,
      "davies_bouldin_index": 0.4903,
      "calinski_harabasz_score": 1847.3
    },
    "external": {
      "adjusted_rand_index": 0.9134,
      "homogeneity": 0.8976,
      "completeness": 0.8891,
      "v_measure": 0.8933
    }
  },
  "cluster_profiles": {
    "cluster_0": {
      "n_samples": 1023,
      "churn_rate": 0.04,
      "risk_label": "BAIXO RISCO"
    },
    "cluster_1": {
      "n_samples": 977,
      "churn_rate": 0.96,
      "risk_label": "ALTO RISCO"
    }
  }
}
```

### Validação automática no CI

O arquivo `tests/test_clustering.py` valida o `cluster_report.json` a cada push:

| Teste | O que previne |
|---|---|
| `silhouette >= 0.3` | Clusters sem separação útil |
| `davies_bouldin < 1.5` | Clusters sobrepostos |
| `v_measure >= 0.5` | Clusterização descorrelacionada do churn |
| Perfil ALTO RISCO existe | Modelo não discrimina churners |
| Inércia monomótona | Erro no cálculo do Elbow |
| Gráficos PNG gerados | Falha silenciosa na visualização |

---

## 8. Variante AWS — Glue + SageMaker

Os arquivos em `aws/` transpõem o mesmo pipeline para serviços gerenciados da AWS, sem alterar a lógica de ML.

### Arquitetura

```
Dado Bruto (S3)
    │
    ▼
┌──────────────────────────────────────┐
│  AWS GLUE (glue_etl_job.py)          │
│  Spark serverless — escala para TB   │
│  · Lê CSV particionado do S3         │
│  · Trata nulos e duplicatas          │
│  · Feature engineering               │
│  · Split treino/teste estratificado  │
│  · Salva Parquet comprimido no S3    │
└────────────────┬─────────────────────┘
                 │ Parquet limpo
                 ▼
┌──────────────────────────────────────┐
│  SAGEMAKER PIPELINES                 │
│  (sagemaker_pipeline.py)             │
│                                      │
│  Step 1 ─ ProcessingStep (ETL)       │
│  Step 2 ─ TrainingStep               │ ← ml.m5.xlarge automático
│             train_sagemaker.py       │
│  Step 3 ─ EvaluationStep             │
│             evaluate.py              │
│  Step 4 ─ ConditionStep              │ ← Bloqueia se ROC-AUC < 0.75
│             RegisterModel            │ → SageMaker Model Registry
└────────────────┬─────────────────────┘
                 │ modelo aprovado
                 ▼
┌──────────────────────────────────────┐
│  SAGEMAKER ENDPOINT                  │
│  · Real-time, AutoScaling gerenciado │
│  · DataCapture para drift monitoring │
│  · Substitui o FastAPI+Docker local  │
└──────────────────────────────────────┘
```

### Comparativo: local vs AWS

| Componente | Local | AWS |
|---|---|---|
| **ETL** | `pandas` em memória | **AWS Glue** (Spark distribuído) |
| **Treinamento** | `python train.py` | **SageMaker Training Job** |
| **Artefato** | `models/model.pkl` | **S3** + **Model Registry** versionado |
| **CI/CD** | GitHub Actions `ci.yml` | **SageMaker Pipelines** (DAG nativo) |
| **Serving** | FastAPI + Docker | **SageMaker Endpoint** + AutoScaling |
| **Monitoramento** | Logs do container | **CloudWatch** + **Model Monitor** |

### Descrição dos arquivos

#### `aws/glue_etl_job.py` — ETL com AWS Glue
Substitui a geração sintética de dados do `train.py`. Conecta-se a fontes reais no S3, executa transformações Spark serverless (sem servidor para gerenciar) e entrega dados limpos em Parquet para o SageMaker.

Pontos principais:
- **DynamicFrame** do Glue lida com esquemas inconsistentes entre registros
- Deduplicação por `customer_id` com `Window` function
- **Data Quality Gate** sinaliza registros fora de range sem removê-los (auditoria)
- Feature engineering derivada (taxa de chamados, receita por produto)
- Split treino/teste estratificado diretamente no Spark

#### `aws/train_sagemaker.py` — Training Job
Equivalente ao `train.py` local, adaptado para o ambiente do SageMaker. **A lógica do Pipeline scikit-learn é idêntica** — a diferença está nas convenções de I/O:

```
/opt/ml/
├── input/data/
│   ├── train/    ← Dados injetados do S3 pelo SageMaker
│   └── test/
├── model/        ← Artefato salvo aqui é enviado para S3 automaticamente
└── output/failure
```

Hiperparâmetros chegam como argumentos de linha de comando (`--n-estimators 200`), permitindo que o **SageMaker Hyperparameter Tuning (HPO)** execute múltiplas variações automaticamente buscando o melhor ROC-AUC.

#### `aws/evaluate.py` — Avaliação formal
Executado como `ProcessingStep` após o treinamento. Gera o `evaluation.json` no formato exigido pelo **SageMaker Model Registry**, que vincula as métricas (ROC-AUC, F1, Avg-Precision) ao Model Package para auditoria e comparação de versões.

#### `aws/sagemaker_pipeline.py` — Orquestrador
Equivalente ao `ci.yml` do GitHub Actions, porém como código Python. Define o DAG completo com parâmetros configuráveis:

```python
# Parâmetros do pipeline (ajustáveis sem alterar código)
param_roc_auc_threshold = ParameterFloat(name="RocAucThreshold", default_value=0.75)
param_n_estimators      = ParameterInteger(name="NEstimators",    default_value=200)

# Executa com parâmetros diferentes sem redeployar o pipeline
execution = pipeline.start(parameters={"RocAucThreshold": 0.80})
```

O `ConditionStep` bloqueia o registro no Model Registry se o ROC-AUC estiver abaixo do threshold — equivalente ao teste `test_metadata_roc_auc_above_threshold` do `tests/test_model.py`, mas nativo na nuvem.

### Como executar

```bash
# Instala dependências AWS
pip install sagemaker boto3

# Configura credenciais
aws configure

# Edite as variáveis no topo do arquivo:
#   BUCKET_NAME, ROLE_ARN, AWS_REGION
vim aws/sagemaker_pipeline.py

# Registra e executa o pipeline
python aws/sagemaker_pipeline.py
```

> **Pré-requisitos:** conta AWS com SageMaker e Glue habilitados, IAM Role com `AmazonSageMakerFullAccess` e bucket S3 criado.

---

## 9. Diagrama CRISP-DM — BPMN

O arquivo [`docs/crisp_dm_bpmn_pipeline.drawio`](docs/crisp_dm_bpmn_pipeline.drawio) contém o diagrama completo do pipeline com **notação BPMN** seguindo o modelo **CRISP-DM**.

### Como abrir
1. Acesse [draw.io](https://app.diagrams.net/) → **File → Open from → Device** → selecione o arquivo; ou
2. Com a extensão **hediet.vscode-drawio** no VS Code, abra o `.drawio` diretamente.

### Estrutura do diagrama (raias / pools BPMN)

| Raia | Fase CRISP-DM | Componentes principais |
|------|--------------|------------------------|
| Negócio | Business Understanding | KPIs, threshold ROC-AUC, aprovação |
| Entendimento de Dados | Data Understanding | EDA, K-Means exploratório, Elbow, Silhouette |
| Preparação de Dados | Data Preparation | AWS Glue ETL, Feature Engineering |
| **Feature Store** ★ | Data Preparation | Offline Store (S3), Online Store (Redis), Feature Registry |
| Modelagem | Modeling | `train.py`, sklearn Pipeline (Scaler+GBM), HPO, CV 5-fold |
| CI/CD | Evaluation | GitHub Actions, ruff, pytest, docker build, smoke test |
| Deploy | Deployment | FastAPI + Docker, SageMaker Endpoint, Blue/Green Rollout |
| **Drift Detection** ★ | Monitoring | Data Drift (KS/PSI), Concept Drift, Prediction Drift, Feature Drift |
| Loop de Retreino | Deployment→Understanding | Trigger automático, atualiza Feature Store, promove Challenger |

> ★ = componentes **não presentes nos arquivos locais**, contemplados no diagrama como extensões MLOps ao CRISP-DM original (Feature Store via AWS SageMaker Feature Store / Redis; Drift Detection via SageMaker Model Monitor / Evidently AI).

### Verificação: Feature Store e Drift Detection

**Feature Store** — representado como subprocesso dedicado entre Data Preparation e Modeling:
- **Offline Store**: features históricas em S3 Parquet para treinamento reprodutível
- **Online Store**: lookup em tempo real (`< 20ms`) durante inferência no `/predict`
- **Feature Registry**: versão, schema, lineage e estatísticas de referência — base para detecção de derivação

**Drift Detection** — representado como raia autônoma pós-Deployment com gateway de decisão:
- **Data Drift**: distribuição das features de entrada diverge do treinamento (KS Test, PSI, χ²)
- **Concept Drift**: relação `X → y` mudou no mundo real (monitor de ROC-AUC em janelas 7d / 30d)
- **Prediction Drift**: taxa de churn prevista diverge do baseline pós-deploy
- **Feature Store Drift** ★: estatísticas atuais comparadas ao snapshot do Feature Registry — detecta problemas na origem dos dados
- Gateway de decisão: Drift detectado → Alerta (CloudWatch / Slack) + Trigger automático de retreino → Loop CRISP-DM

---

## 10. Referências e Conceitos-chave

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
