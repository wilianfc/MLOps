"""
=============================================================================
AWS SAGEMAKER — Script de Treinamento (Entry Point)
=============================================================================

Este script é executado DENTRO do container do SageMaker Training Job.
É o equivalente ao train.py local, mas adaptado para o ambiente gerenciado
da AWS, respeitando as convenções de diretórios do SageMaker.

COMO O SAGEMAKER TRAINING JOB FUNCIONA:
  1. A AWS provisiona uma instância (ex: ml.m5.xlarge) automaticamente.
  2. Injeta os dados do S3 nos caminhos /opt/ml/input/data/{channel}/.
  3. Executa este script.
  4. Comprime tudo que estiver em /opt/ml/model/ e envia de volta ao S3.
  5. Encerra e desaloca a instância — você paga só pelo tempo de treino.

CONVENÇÕES DE DIRETÓRIOS DO SAGEMAKER:
  /opt/ml/
  ├── input/
  │   ├── data/
  │   │   ├── train/    ← Dados de treino (injetados do S3)
  │   │   └── test/     ← Dados de teste  (injetados do S3)
  │   └── config/
  │       └── hyperparameters.json  ← Hiperparâmetros do job
  ├── model/            ← DESTINO DO ARTEFATO (sincronizado com S3)
  └── output/
      └── failure       ← Mensagem de erro (se o job falhar)

COMO SUBIR ESTE SCRIPT NO SAGEMAKER (via Python SDK):
  Ver aws/sagemaker_pipeline.py para o código completo de orquestração.

  Resumo rápido:
    from sagemaker.sklearn import SKLearn
    estimator = SKLearn(
        entry_point="aws/train_sagemaker.py",
        framework_version="1.2-1",
        instance_type="ml.m5.xlarge",
        hyperparameters={"n_estimators": 200, "learning_rate": 0.05},
    )
    estimator.fit({"train": train_s3_uri, "test": test_s3_uri})

REFERÊNCIAS:
  - AWS MLOps: Continuous Delivery for Machine Learning on AWS
  - AWS Samples: automated-machine-learning-pipeline-options-on-aws
  - Machine Learning Lens — AWS Well-Architected Framework
=============================================================================
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    average_precision_score,
)

# ---------------------------------------------------------------------------
# Configuração de Logging
# Logs enviados para stdout são capturados pelo CloudWatch Logs automaticamente
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sagemaker_train")

# ---------------------------------------------------------------------------
# Caminhos injetados pelo SageMaker — NÃO altere estes caminhos
# ---------------------------------------------------------------------------
SM_INPUT_DIR   = "/opt/ml/input/data"
SM_MODEL_DIR   = "/opt/ml/model"
SM_OUTPUT_DIR  = "/opt/ml/output"

# Para desenvolvimento local, usa caminhos relativos como fallback
LOCAL_TRAIN_DIR = os.path.join("models", "_sm_local", "train")
LOCAL_TEST_DIR  = os.path.join("models", "_sm_local", "test")
LOCAL_MODEL_DIR = os.path.join("models")

# Detecta se está rodando dentro do SageMaker ou localmente
IS_SAGEMAKER = os.path.exists("/opt/ml")


def get_path(sm_path: str, local_path: str) -> str:
    """Retorna o caminho correto dependendo do ambiente."""
    return sm_path if IS_SAGEMAKER else local_path


# ---------------------------------------------------------------------------
# 1. Parsing de Hiperparâmetros
#    O SageMaker injeta os hiperparâmetros como argumentos de linha de comando
# ---------------------------------------------------------------------------
def parse_args():
    """
    Parseia os hiperparâmetros passados pelo SageMaker ou pela linha de comando.

    No SageMaker, ao criar o Training Job você define:
      hyperparameters={"n_estimators": 200, "learning_rate": "0.05"}

    O SageMaker converterá isso em:
      python train_sagemaker.py --n-estimators 200 --learning-rate 0.05
    """
    parser = argparse.ArgumentParser(description="Churn Prediction — SageMaker Training")

    # Hiperparâmetros do modelo
    parser.add_argument("--n-estimators",    type=int,   default=200,  help="Número de árvores no GBM")
    parser.add_argument("--learning-rate",   type=float, default=0.05, help="Taxa de aprendizado")
    parser.add_argument("--max-depth",       type=int,   default=4,    help="Profundidade máxima das árvores")
    parser.add_argument("--subsample",       type=float, default=0.8,  help="Fração de amostras por árvore")
    parser.add_argument("--min-samples-split", type=int, default=10,   help="Mínimo de amostras para dividir nó")
    parser.add_argument("--random-state",    type=int,   default=42,   help="Semente aleatória")

    # Caminhos — sobrescrevem os padrões do SageMaker se necessário
    parser.add_argument(
        "--model-dir",
        type=str,
        default=get_path(SM_MODEL_DIR, LOCAL_MODEL_DIR),
        help="Diretório de saída do artefato do modelo",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=get_path(f"{SM_INPUT_DIR}/train", LOCAL_TRAIN_DIR),
        help="Diretório com dados de treino",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=get_path(f"{SM_INPUT_DIR}/test", LOCAL_TEST_DIR),
        help="Diretório com dados de teste",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Carregamento de Dados
#    O Glue ETL (glue_etl_job.py) salvou os dados como Parquet no S3.
#    O SageMaker os copia para /opt/ml/input/data/{channel}/ antes do treino.
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_calls",
    "payment_delay_days",
    "age",
    "satisfaction_score",
    # Features derivadas criadas pelo Glue ETL
    "avg_revenue_per_product",
    "support_call_rate",
    "is_new_customer",
]

TARGET = "churn"


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Carrega todos os arquivos Parquet de um diretório.

    O SageMaker pode dividir os dados em múltiplos shards entre instâncias
    (treinamento distribuído). Esta função lida com múltiplos arquivos.

    Args:
        data_dir: Caminho para o diretório com arquivos .parquet

    Returns:
        DataFrame consolidado.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Diretório de dados não encontrado: '{data_dir}'. "
            f"Certifique-se de que o canal S3 está configurado no Training Job."
        )

    # Suporta .parquet (Glue) e .csv (fallback para testes locais)
    parquet_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet")
    ]
    csv_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv")
    ]

    if parquet_files:
        dfs = [pd.read_parquet(f) for f in parquet_files]
        logger.info("Carregados %d arquivo(s) Parquet de '%s'", len(parquet_files), data_dir)
    elif csv_files:
        dfs = [pd.read_csv(f) for f in csv_files]
        logger.info("Carregados %d arquivo(s) CSV de '%s'", len(csv_files), data_dir)
    else:
        raise FileNotFoundError(
            f"Nenhum arquivo .parquet ou .csv encontrado em '{data_dir}'"
        )

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Dataset carregado: %d linhas × %d colunas", *df.shape)
    return df


def prepare_features(df: pd.DataFrame):
    """
    Seleciona e valida as features do DataFrame.

    Garante que todas as features esperadas existam — se o Glue ETL adicionou
    ou removeu features, este check detecta o problema antes de treinar.

    Args:
        df: DataFrame carregado do S3.

    Returns:
        (X, y) como DataFrame e Series.
    """
    # Usa apenas features disponíveis (Glue pode não ter gerado as derivadas)
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]

    if missing_features:
        logger.warning(
            "Features ausentes (não geradas pelo ETL): %s. "
            "Treinando com %d features disponíveis.",
            missing_features, len(available_features)
        )

    X = df[available_features]
    y = df[TARGET]

    logger.info(
        "Features selecionadas: %d | Taxa de churn: %.1f%%",
        len(available_features),
        y.mean() * 100,
    )
    return X, y, available_features


# ---------------------------------------------------------------------------
# 3. Pipeline de Treinamento (mesmo conceito do train.py local)
# ---------------------------------------------------------------------------
def build_and_train_pipeline(X_train, y_train, args) -> Pipeline:
    """
    Constrói e treina o Pipeline scikit-learn com os hiperparâmetros
    recebidos do SageMaker (via argparse).

    O SageMaker Hyperparameter Tuning (HPO) irá chamar este script
    múltiplas vezes com diferentes combinações de hiperparâmetros,
    buscando o melhor ROC-AUC. O objetivo principal (n_estimators,
    learning_rate) é configurado no aws/sagemaker_pipeline.py.
    """
    logger.info(
        "Construindo pipeline — n_estimators=%d, learning_rate=%.3f, max_depth=%d",
        args.n_estimators, args.learning_rate, args.max_depth,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
        )),
    ])

    # Cross-validation para estimar generalização
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info("ROC-AUC (CV 5-fold) — Média: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # Treino final
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# 4. Avaliação e Métricas para o SageMaker
#    O SageMaker captura métricas do stdout via Regex configurado no job.
#    Formato: "metricName: valor" é capturado pelo MetricDefinition.
# ---------------------------------------------------------------------------
def evaluate_and_log_metrics(pipeline, X_test, y_test):
    """
    Avalia o modelo e imprime métricas em formato capturável pelo SageMaker.

    O SageMaker CloudWatch Metrics captura padrões de texto do stdout.
    Ao criar o Training Job, você define:
      metric_definitions=[
        {"Name": "validation:roc_auc", "Regex": "ROC-AUC.*Teste.*: ([0-9\\.]+)"},
        {"Name": "validation:avg_precision", "Regex": "Avg-Precision.*: ([0-9\\.]+)"},
      ]

    Returns:
        dict com as métricas calculadas.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc    = roc_auc_score(y_test, y_proba)
    avg_prec   = average_precision_score(y_test, y_proba)

    # Formato capturável pelo SageMaker (via MetricDefinition Regex)
    print(f"ROC-AUC (Teste): {roc_auc:.4f}")
    print(f"Avg-Precision: {avg_prec:.4f}")

    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred, target_names=["Não Churnou", "Churnou"]))

    logger.info("Avaliação concluída — ROC-AUC: %.4f | Avg-Precision: %.4f", roc_auc, avg_prec)
    return {"roc_auc": roc_auc, "avg_precision": avg_prec}


# ---------------------------------------------------------------------------
# 5. Salvamento do Artefato
#    O SageMaker comprime e envia /opt/ml/model/ para o S3 automaticamente
# ---------------------------------------------------------------------------
def save_artifacts(pipeline, features, metrics, model_dir, args):
    """
    Salva o artefato do modelo e os metadados no diretório do SageMaker.

    Após o treino, o SageMaker comprime /opt/ml/model/ em model.tar.gz
    e envia para o S3, tornando o artefato disponível para:
      - SageMaker Model Registry (versionamento)
      - SageMaker Endpoints (serving em tempo real)
      - SageMaker Batch Transform (inferência batch)
    """
    os.makedirs(model_dir, exist_ok=True)

    # Artefato principal
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Artefato salvo: %s (%.1f KB)", model_path, os.path.getsize(model_path) / 1024)

    # Metadados de rastreabilidade
    from datetime import datetime
    metadata = {
        "model_type": type(pipeline.named_steps["classifier"]).__name__,
        "scaler":     type(pipeline.named_steps["scaler"]).__name__,
        "features":   features,
        "n_features": len(features),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "environment": "sagemaker" if IS_SAGEMAKER else "local",
        "hyperparameters": {
            "n_estimators":      args.n_estimators,
            "learning_rate":     args.learning_rate,
            "max_depth":         args.max_depth,
            "subsample":         args.subsample,
            "min_samples_split": args.min_samples_split,
            "random_state":      args.random_state,
        },
        "metrics": metrics,
    }

    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadados salvos: %s", meta_path)

    # inference.py — necessário para SageMaker Endpoints com SKLearn container
    # O SDK do SageMaker chama model_fn() para carregar o modelo no endpoint
    inference_code = '''"""
Handler de inferência para SageMaker Endpoints com container SKLearn.
O SageMaker chama model_fn() no startup e predict_fn() a cada request.
"""
import pickle
import os
import numpy as np

def model_fn(model_dir):
    """Carrega o pipeline do artefato e retorna para o SageMaker."""
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_fn(input_data, model):
    """Recebe array numpy e retorna probabilidades de churn."""
    probabilities = model.predict_proba(input_data)[:, 1]
    labels = (probabilities >= 0.5).astype(int)
    return np.column_stack([labels, probabilities])

def input_fn(request_body, content_type="application/json"):
    """Deserializa o payload de entrada."""
    import json
    if content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["instances"], dtype=np.float64)
    raise ValueError(f"Content-type não suportado: {content_type}")

def output_fn(prediction, accept="application/json"):
    """Serializa a resposta."""
    import json
    results = [
        {"churn_prediction": int(pred[0]), "churn_probability": round(float(pred[1]), 4)}
        for pred in prediction
    ]
    return json.dumps(results), accept
'''
    inference_path = os.path.join(model_dir, "inference.py")
    with open(inference_path, "w") as f:
        f.write(inference_code)
    logger.info("inference.py salvo para SageMaker Endpoint: %s", inference_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("SAGEMAKER TRAINING JOB — CHURN PREDICTION")
    logger.info("Ambiente: %s", "SageMaker" if IS_SAGEMAKER else "Local")
    logger.info("=" * 60)

    # 1. Carrega dados dos canais S3 (injetados pelo SageMaker)
    df_train = load_data(args.train)
    df_test  = load_data(args.test)

    # 2. Prepara features
    X_train, y_train, feature_names = prepare_features(df_train)
    X_test,  y_test,  _             = prepare_features(df_test)
    X_test = X_test[feature_names]   # Garante mesma ordem de colunas

    # 3. Treina pipeline
    pipeline = build_and_train_pipeline(X_train, y_train, args)

    # 4. Avalia e publica métricas no CloudWatch
    metrics = evaluate_and_log_metrics(pipeline, X_test, y_test)

    # 5. Salva artefato em /opt/ml/model/ (SageMaker envia para S3)
    save_artifacts(pipeline, feature_names, metrics, args.model_dir, args)

    logger.info("Training Job finalizado com sucesso!")


if __name__ == "__main__":
    main()
