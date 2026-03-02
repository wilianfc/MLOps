"""
=============================================================================
ANÁLISE DE CLUSTERIZAÇÃO — Segmentação de Clientes por Risco de Churn
=============================================================================

Este script complementa o pipeline supervisionado (train.py) com uma análise
não supervisionada. Em vez de prever "vai ou não churnar", identificamos
PERFIS NATURAIS DE COMPORTAMENTO na base de clientes sem usar os rótulos.

USO NO PIPELINE MLOps:
  1. EXPLORAÇÃO: entender grupos antes de treinar o classificador
  2. FEATURE ENGINEERING: usar o cluster como feature adicional no train.py
  3. MONITORAMENTO: detectar quando novos clientes não se encaixam em nenhum
     perfil histórico (out-of-distribution) — sinal de concept drift

COMO RODAR:
    python clustering_analysis.py
    # Gera: models/cluster_report.json
    #       models/elbow_curve.png
    #       models/silhouette_plot.png

=============================================================================
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from sklearn.decomposition import PCA

# Modo sem interface gráfica (compatível com CI/CD e containers Docker)
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("clustering")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
K_RANGE      = range(2, 9)
CHOSEN_K     = 2        # Alterado após inspeção do Elbow + Silhouette
OUTPUT_DIR   = "models"

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


# ---------------------------------------------------------------------------
# 1. Geração de dados sintéticos (reutiliza a lógica do train.py)
# ---------------------------------------------------------------------------
def generate_data(n_samples: int = 2000) -> pd.DataFrame:
    """Gera dados sintéticos com dois perfis bem definidos para visualização."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_each = n_samples // 2

    # Perfil A — Baixo Risco: longa duração, satisfação alta, poucas chamadas
    perfil_baixo = pd.DataFrame({
        "tenure_months":       rng.integers(24, 72, n_each),
        "monthly_charges":     rng.normal(130, 25, n_each).clip(50, 300),
        "total_charges":       rng.normal(6000, 1500, n_each).clip(1000, 15000),
        "num_products":        rng.integers(3, 7, n_each),
        "support_calls":       rng.integers(0, 4, n_each),
        "payment_delay_days":  rng.integers(0, 5, n_each),
        "age":                 rng.integers(35, 70, n_each),
        "satisfaction_score":  rng.normal(7.5, 1.0, n_each).clip(5, 10).round(1),
        "churn":               np.zeros(n_each, dtype=int),
    })

    # Perfil B — Alto Risco: recente, caro, muitas chamadas, insatisfeito
    perfil_alto = pd.DataFrame({
        "tenure_months":       rng.integers(1, 12, n_each),
        "monthly_charges":     rng.normal(380, 60, n_each).clip(150, 500),
        "total_charges":       rng.normal(2000, 800, n_each).clip(100, 6000),
        "num_products":        rng.integers(1, 3, n_each),
        "support_calls":       rng.integers(7, 15, n_each),
        "payment_delay_days":  rng.integers(10, 30, n_each),
        "age":                 rng.integers(18, 40, n_each),
        "satisfaction_score":  rng.normal(2.5, 0.8, n_each).clip(0, 5).round(1),
        "churn":               np.ones(n_each, dtype=int),
    })

    df = pd.concat([perfil_baixo, perfil_alto]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    logger.info("Dataset gerado: %d amostras | Churn: %.1f%%", len(df), df["churn"].mean() * 100)
    return df


# ---------------------------------------------------------------------------
# 2. Elbow Method — encontra o K ideal pela inércia (WCSS)
# ---------------------------------------------------------------------------
def elbow_method(X: np.ndarray, k_range=K_RANGE) -> dict:
    """
    Testa vários valores de K e calcula a inércia (WCSS — Within-Cluster Sum of Squares).

    O "cotovelo" da curva indica o K onde adicionar mais clusters
    gera retornos decrescentes de compactação.

    Args:
        X:       Array de features já escalado.
        k_range: Intervalo de valores de K a testar.

    Returns:
        Dicionário {k: inertia}.
    """
    results = {}
    logger.info("Executando Elbow Method para K=%s...", list(k_range))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        results[k] = round(km.inertia_, 2)
        logger.info("  K=%d → Inércia: %.2f", k, km.inertia_)

    return results


def plot_elbow(inertias: dict, chosen_k: int) -> str:
    """Plota a curva de inércia com marcação do K escolhido."""
    ks = list(inertias.keys())
    vals = list(inertias.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, vals, "bo-", linewidth=2, markersize=8)
    ax.axvline(x=chosen_k, color="red", linestyle="--", linewidth=1.5,
               label=f"K escolhido = {chosen_k}")
    ax.set_xlabel("Número de Clusters (K)", fontsize=12)
    ax.set_ylabel("Inércia (WCSS)", fontsize=12)
    ax.set_title("Elbow Method — Escolha do K Ideal", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "elbow_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Gráfico salvo: %s", path)
    return path


# ---------------------------------------------------------------------------
# 3. Silhouette Plot — qualidade individual de cada ponto
# ---------------------------------------------------------------------------
def plot_silhouette(X: np.ndarray, labels: np.ndarray, k: int) -> str:
    """
    Plota o Silhouette de cada amostra agrupada por cluster.

    A largura de cada barra representa o s(i) do ponto — quanto mais à
    direita, melhor alocado. Cortes abruptos indicam pontos mal alocados.

    Args:
        X:      Array de features escalado.
        labels: Rótulos dos clusters (saída do KMeans).
        k:      Número de clusters.

    Returns:
        Caminho do arquivo PNG salvo.
    """
    silhouette_vals = silhouette_samples(X, labels)
    avg_score = silhouette_vals.mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    y_lower = 10

    for cluster_id in range(k):
        cluster_silhouette = np.sort(silhouette_vals[labels == cluster_id])
        cluster_size = cluster_silhouette.shape[0]
        y_upper = y_lower + cluster_size

        color = cm.nipy_spectral(float(cluster_id) / k)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_silhouette,
            alpha=0.7, color=color, label=f"Cluster {cluster_id}",
        )
        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(cluster_id), fontsize=11)
        y_lower = y_upper + 10

    ax.axvline(x=avg_score, color="red", linestyle="--",
               label=f"Média = {avg_score:.3f}")
    ax.set_xlabel("Coeficiente de Silhouette  [ s(i) ]", fontsize=12)
    ax.set_ylabel("Amostras por Cluster", fontsize=12)
    ax.set_title(f"Silhouette Plot — K={k}", fontsize=14, fontweight="bold")
    ax.set_xlim([-0.2, 1.0])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "silhouette_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Gráfico salvo: %s", path)
    return path


# ---------------------------------------------------------------------------
# 4. Visualização PCA — projeção 2D dos clusters
# ---------------------------------------------------------------------------
def plot_pca(X: np.ndarray, labels: np.ndarray, k: int) -> str:
    """
    Reduz para 2 dimensões com PCA e plota os clusters coloridos.

    Permite inspecionar visualmente a separação dos grupos em 2D,
    mesmo que o espaço original seja de alta dimensão.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cm.nipy_spectral(np.linspace(0, 1, k))

    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors[cluster_id]], label=f"Cluster {cluster_id}",
            alpha=0.5, s=20, edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title(
        f"Clusters em 2D via PCA  (variância explicada: {var_explained:.1f}%)",
        fontsize=13, fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "pca_clusters.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Gráfico salvo: %s", path)
    return path


# ---------------------------------------------------------------------------
# 5. Métricas consolidadas
# ---------------------------------------------------------------------------
def compute_metrics(X: np.ndarray, labels: np.ndarray, y_real: np.ndarray) -> dict:
    """
    Calcula métricas internas e externas do agrupamento.

    INTERNAS (sem gabarito)           EXTERNAS (com rótulo real de churn)
    ────────────────────────          ────────────────────────────────────
    Silhouette   → separação          ARI          → concordância global
    Davies-Bouldin → compactação      Homogeneidade → clusters puros?
    Calinski-H   → razão variâncias   Completude   → classes concentradas?
                                      V-Measure    → equilíbrio H + C
    """
    silhouette  = silhouette_score(X, labels)
    db          = davies_bouldin_score(X, labels)
    ch          = calinski_harabasz_score(X, labels)

    ari         = adjusted_rand_score(y_real, labels)
    h, c, v     = homogeneity_completeness_v_measure(y_real, labels)

    metrics = {
        "internal": {
            "silhouette_score":          round(silhouette, 4),
            "davies_bouldin_index":      round(db, 4),
            "calinski_harabasz_score":   round(ch, 2),
        },
        "external": {
            "adjusted_rand_index":       round(ari, 4),
            "homogeneity":               round(h, 4),
            "completeness":              round(c, 4),
            "v_measure":                 round(v, 4),
        },
    }

    print("\n" + "=" * 58)
    print("MÉTRICAS INTERNAS  (sem gabarito de rótulo)")
    print("=" * 58)
    print(f"  Silhouette Score    : {silhouette:.4f}  (melhor → 1.0)")
    print(f"  Davies-Bouldin      : {db:.4f}  (melhor → 0.0)")
    print(f"  Calinski-Harabasz   : {ch:.2f}  (melhor → máximo)")
    print()
    print("MÉTRICAS EXTERNAS  (análogas a Precision / Recall / F1)")
    print("=" * 58)
    print(f"  Adjusted Rand Index : {ari:.4f}  (melhor → 1.0)")
    print(f"  Homogeneidade       : {h:.4f}  (≈ Precision — clusters puros?)")
    print(f"  Completude          : {c:.4f}  (≈ Recall   — classes juntas?)")
    print(f"  V-Measure           : {v:.4f}  (≈ F1-Score — equilíbrio H+C)")
    print("=" * 58 + "\n")

    return metrics


# ---------------------------------------------------------------------------
# 6. Perfil de cada cluster — interpretação de negócio
# ---------------------------------------------------------------------------
def profile_clusters(df_original: pd.DataFrame, labels: np.ndarray, features: list) -> dict:
    """
    Calcula estatísticas descritivas por cluster e infere o perfil de risco.

    Esta etapa transforma o resultado estatístico em linguagem de negócio:
    "Cluster 0 = clientes estabelecidos de baixo risco"
    "Cluster 1 = clientes novos de alto risco"

    Args:
        df_original: DataFrame com features originais (não escalado).
        labels:      Rótulos dos clusters.
        features:    Lista de nomes de features.

    Returns:
        Dicionário com estatísticas e rótulo de risco por cluster.
    """
    df = df_original[features].copy()
    df["cluster"] = labels
    df["churn"]   = df_original["churn"].values

    profiles = {}
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        churn_rate = subset["churn"].mean()
        risk_label = "ALTO RISCO" if churn_rate >= 0.5 else "BAIXO RISCO"

        profiles[f"cluster_{cluster_id}"] = {
            "n_samples":    int(len(subset)),
            "churn_rate":   round(float(churn_rate), 4),
            "risk_label":   risk_label,
            "feature_means": {
                feat: round(float(subset[feat].mean()), 3)
                for feat in features
            },
        }

        print(f"Cluster {cluster_id} — {risk_label}  ({len(subset)} clientes, {churn_rate*100:.1f}% churn)")
        print(f"  satisfaction_score : {subset['satisfaction_score'].mean():.2f}")
        print(f"  tenure_months      : {subset['tenure_months'].mean():.1f}")
        print(f"  support_calls      : {subset['support_calls'].mean():.1f}")
        print(f"  monthly_charges    : R$ {subset['monthly_charges'].mean():.2f}")
        print()

    return profiles


# ---------------------------------------------------------------------------
# 7. Salva relatório JSON consolidado
# ---------------------------------------------------------------------------
def save_report(metrics: dict, profiles: dict, inertias: dict, chosen_k: int) -> str:
    """
    Persiste todos os resultados em cluster_report.json para rastreabilidade.

    Este JSON pode ser consumido por:
    - Testes automáticos (test_clustering.py)
    - Dashboard de monitoramento (Grafana, CloudWatch)
    - SageMaker Experiments para comparação de experimentos
    """
    from datetime import datetime
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "chosen_k":     chosen_k,
        "features":     FEATURE_NAMES,
        "metrics":      metrics,
        "cluster_profiles": profiles,
        "elbow_inertias": {str(k): v for k, v in inertias.items()},
    }

    path = os.path.join(OUTPUT_DIR, "cluster_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Relatório salvo: %s", path)
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 58)
    logger.info("ANÁLISE DE CLUSTERIZAÇÃO — SEGMENTAÇÃO DE CLIENTES")
    logger.info("=" * 58)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dados
    df = generate_data(n_samples=2000)
    X_raw = df[FEATURE_NAMES].values
    y_real = df["churn"].values

    # 2. Pipeline de escalonamento (mesma abordagem do train.py)
    #    O scaler é ajustado SOMENTE nos dados de análise, não no teste
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. Elbow Method — inspeciona K de 2 a 8
    inertias = elbow_method(X_scaled, K_RANGE)
    plot_elbow(inertias, CHOSEN_K)

    # 4. Treina KMeans com o K escolhido
    logger.info("Treinando KMeans com K=%d...", CHOSEN_K)
    kmeans = KMeans(n_clusters=CHOSEN_K, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 5. Silhouette plot
    plot_silhouette(X_scaled, labels, CHOSEN_K)

    # 6. PCA 2D
    plot_pca(X_scaled, labels, CHOSEN_K)

    # 7. Métricas internas + externas
    metrics = compute_metrics(X_scaled, labels, y_real)

    # 8. Perfil de negócio
    print("PERFIL DOS CLUSTERS")
    print("-" * 58)
    profiles = profile_clusters(df, labels, FEATURE_NAMES)

    # 9. Relatório JSON
    report_path = save_report(metrics, profiles, inertias, CHOSEN_K)

    logger.info("Clusterização concluída! Arquivos gerados:")
    logger.info("  - %s", os.path.join(OUTPUT_DIR, "elbow_curve.png"))
    logger.info("  - %s", os.path.join(OUTPUT_DIR, "silhouette_plot.png"))
    logger.info("  - %s", os.path.join(OUTPUT_DIR, "pca_clusters.png"))
    logger.info("  - %s", report_path)


if __name__ == "__main__":
    main()
