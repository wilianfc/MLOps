"""
=============================================================================
PASSO 2: TREINAMENTO DO MODELO E PREVENÇÃO DE DATA-LEAKAGE
=============================================================================

Script de treinamento para o modelo de Previsão de Churn de Clientes.

CONCEITO CENTRAL — Data-Leakage (Vazamento de Dados):
    Ocorre quando informações do conjunto de teste "vazam" para o treinamento,
    produzindo métricas artificialmente infladas e um modelo que falha em
    produção. A causa mais comum é fazer o `fit` do StandardScaler em TODOS
    os dados antes da divisão treino/teste.

SOLUÇÃO — sklearn.pipeline.Pipeline:
    Encadeia pré-processamento + estimador em um único objeto. O `.fit()`
    é chamado APENAS nos dados de treino. Na inferência, a mesma
    transformação aprendida é aplicada automaticamente, sem exposição
    aos dados de teste durante o treinamento.

    Referências:
    - Google for Developers: Prevenção de label leakage em pipelines de ML
    - Databricks: Sequência de transformadores e estimadores com scikit-learn
    - Datarisk: Como evitar data-leakage no gerenciamento de dados para ML
=============================================================================
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Configuração de Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_PATH = os.path.join("models", "model.pkl")

# Nomes das features — documentar explicitamente evita erros silenciosos
# em produção quando a ordem das colunas muda
FEATURE_NAMES = [
    "tenure_months",       # Tempo de contrato (meses)
    "monthly_charges",     # Valor da mensalidade (R$)
    "total_charges",       # Valor total pago (R$)
    "num_products",        # Qtd. de produtos contratados
    "support_calls",       # Chamados ao suporte (últimos 6 meses)
    "payment_delay_days",  # Atraso médio no pagamento (dias)
    "age",                 # Idade do cliente
    "satisfaction_score",  # Pontuação de satisfação (0–10)
]


# ---------------------------------------------------------------------------
# 1. Geração de Dados Sintéticos
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Gera um dataset sintético de clientes com características realistas.

    Utiliza `make_classification` para criar base correlacionada e enriquece
    com features de negócio para tornar o cenário mais verossímil.

    Args:
        n_samples: Número de clientes a gerar.

    Returns:
        DataFrame com FEATURE_NAMES + coluna 'churn' (0 = não churnou,
        1 = churnou).
    """
    logger.info("Gerando %d amostras sintéticas de clientes...", n_samples)

    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.75, 0.25],   # Dataset desbalanceado (~25% churn) — realista
        flip_y=0.03,            # Ruído de rótulo — simula erros de registro
        random_state=RANDOM_STATE,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    # Mapeia as features brutas para valores de negócio interpretáveis
    df = pd.DataFrame(
        {
            "tenure_months":       np.clip(X_raw[:, 0] * 12 + 24, 1, 72).astype(int),
            "monthly_charges":     np.clip(X_raw[:, 1] * 50 + 150, 50, 500).round(2),
            "total_charges":       np.clip(X_raw[:, 2] * 2000 + 5000, 100, 20000).round(2),
            "num_products":        np.clip((X_raw[:, 3] * 2 + 3).astype(int), 1, 6),
            "support_calls":       np.clip((X_raw[:, 4] * 3 + 2).astype(int), 0, 15),
            "payment_delay_days":  np.clip((X_raw[:, 5] * 5 + 5).astype(int), 0, 30),
            "age":                 np.clip((X_raw[:, 6] * 10 + 40).astype(int), 18, 80),
            "satisfaction_score":  np.clip(X_raw[:, 7] * 2 + 5, 0, 10).round(1),
            "churn":               y,
        }
    )

    churn_rate = df["churn"].mean() * 100
    logger.info("Dataset gerado. Taxa de churn: %.1f%%", churn_rate)
    return df


# ---------------------------------------------------------------------------
# 2. Divisão Treino/Teste — estratificada para preservar proporção de classes
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    """
    Divide o DataFrame em conjuntos de treino e teste de forma estratificada.

    IMPORTANTE — Por que estratificar?
        Com datasets desbalanceados (ex: 25% churn), uma divisão aleatória
        simples pode gerar um conjunto de teste sem exemplos suficientes da
        classe minoritária, tornando as métricas não confiáveis.

    Args:
        df: DataFrame completo com features e rótulo.

    Returns:
        (X_train, X_test, y_train, y_test) como DataFrames/Series.
    """
    X = df[FEATURE_NAMES]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,             # Garante mesma proporção de classes em cada split
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Divisão concluída — Treino: %d amostras | Teste: %d amostras",
        len(X_train), len(X_test),
    )
    logger.info(
        "Proporção de churn — Treino: %.1f%% | Teste: %.1f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 3. Construção do Pipeline — NÚCLEO DA PREVENÇÃO DE DATA-LEAKAGE
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """
    Constrói o Pipeline scikit-learn com pré-processamento + classificador.

    ------------------------------------------------------------------
    | Step 1: StandardScaler                                         |
    |   - Remove a média e escalona para variância unitária           |
    |   - .fit() aprende média/std SOMENTE nos dados de treino       |
    |   - .transform() aplica a mesma escala em qualquer novo dado   |
    |                                                                |
    | Step 2: GradientBoostingClassifier                             |
    |   - Algoritmo robusto para dados tabulares desbalanceados      |
    |   - subsample < 1 adiciona regularização estocástica           |
    ------------------------------------------------------------------

    Ao empacotar os dois passos no mesmo objeto Pipeline, garantimos que:
    1. Durante o treino: scaler aprende SÓ com X_train.
    2. Durante a inferência: scaler transforma com os parâmetros aprendidos.
    3. O artefato salvo (model.pkl) contém scaler + modelo — zero drift.

    Returns:
        Pipeline scikit-learn não treinado.
    """
    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    min_samples_split=10,
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        verbose=False,
    )
    return pipeline


# ---------------------------------------------------------------------------
# 4. Treinamento e Avaliação
# ---------------------------------------------------------------------------
def train_and_evaluate(pipeline: Pipeline, X_train, X_test, y_train, y_test):
    """
    Treina o pipeline e avalia no conjunto de teste.

    Utiliza Cross-Validation no conjunto de treino para estimar a
    generalização ANTES de avaliar no teste — boa prática para detectar
    overfitting cedo.

    Args:
        pipeline: Pipeline scikit-learn construído por build_pipeline().
        X_train, y_train: Dados de treinamento.
        X_test, y_test:   Dados de teste (usados SOMENTE para avaliação final).

    Returns:
        Pipeline treinado.
    """
    # --- Validação Cruzada no treino (estimativa de generalização) ----------
    logger.info("Executando validação cruzada (5-fold estratificado)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    logger.info(
        "ROC-AUC (CV) — Média: %.4f ± %.4f",
        cv_scores.mean(),
        cv_scores.std(),
    )

    # --- Treinamento final no conjunto completo de treino -------------------
    logger.info("Treinando pipeline no conjunto completo de treino...")
    pipeline.fit(X_train, y_train)

    # --- Avaliação no conjunto de TESTE (dados nunca vistos) ----------------
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    logger.info("ROC-AUC (Teste): %.4f", roc_auc)

    print("\n" + "=" * 60)
    print("RELATÓRIO DE CLASSIFICAÇÃO — CONJUNTO DE TESTE")
    print("=" * 60)
    print(
        classification_report(
            y_test, y_pred, target_names=["Não Churnou (0)", "Churnou (1)"]
        )
    )
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print("=" * 60 + "\n")

    return pipeline


# ---------------------------------------------------------------------------
# 5. Persistência do Artefato
# ---------------------------------------------------------------------------
def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> None:
    """
    Serializa o Pipeline treinado em disco usando pickle.

    O artefato contém TODOS os passos do pipeline (scaler + classifier),
    garantindo que a inferência em produção aplique exatamente as mesmas
    transformações aprendidas no treino.

    Boas práticas aplicadas:
    - Caminho configurável via constante (facilita CI/CD e Docker)
    - Diretório criado automaticamente se não existir
    - Log com confirmação do caminho salvo

    Args:
        pipeline: Pipeline scikit-learn treinado.
        path:     Caminho do arquivo de saída.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(pipeline, f)

    model_size_kb = os.path.getsize(path) / 1024
    logger.info("Artefato salvo em '%s' (%.1f KB)", path, model_size_kb)


# ---------------------------------------------------------------------------
# 6. Metadados do Modelo — rastreabilidade básica para MLOps
# ---------------------------------------------------------------------------
def save_model_metadata(pipeline: Pipeline, roc_auc: float) -> None:
    """
    Salva metadados básicos do modelo para rastreabilidade.

    Em um pipeline MLOps maduro, esses dados seriam registrados em um
    Model Registry (ex: MLflow, SageMaker Model Registry). Aqui usamos
    um JSON simples como ponto de partida.
    """
    import json
    from datetime import datetime

    metadata = {
        "model_type": type(pipeline.named_steps["classifier"]).__name__,
        "scaler": type(pipeline.named_steps["scaler"]).__name__,
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "roc_auc_test": round(roc_auc, 4),
    }

    meta_path = os.path.join("models", "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Metadados salvos em '%s'", meta_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE TREINAMENTO — CHURN PREDICTION")
    logger.info("=" * 60)

    # 1. Dados
    df = generate_synthetic_data(n_samples=5000)

    # 2. Divisão treino/teste (ANTES de qualquer pré-processamento)
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Pipeline (scaler + classificador)
    pipeline = build_pipeline()

    # 4. Treino + avaliação
    pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

    # 5. Persiste artefato
    save_model(pipeline)

    # 6. Metadados
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    roc_auc_final = roc_auc_score(y_test, y_proba_test)
    save_model_metadata(pipeline, roc_auc_final)

    logger.info("Pipeline de treinamento concluído com sucesso!")


if __name__ == "__main__":
    main()
