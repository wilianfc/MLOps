"""
=============================================================================
TESTES DE CLUSTERIZAÇÃO — Validação do clustering_analysis.py
=============================================================================

Verifica que:
  1. O relatório cluster_report.json existe e tem estrutura correta
  2. As métricas de qualidade estão acima de thresholds mínimos
  3. Os perfis de cluster fazem sentido de negócio
  4. Os gráficos foram gerados

Execute com:
    pytest tests/test_clustering.py -v
=============================================================================
"""

import os
import json
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
REPORT_PATH    = os.environ.get("CLUSTER_REPORT_PATH", "models/cluster_report.json")
ELBOW_PATH     = "models/elbow_curve.png"
SILHOUETTE_PATH = "models/silhouette_plot.png"
PCA_PATH        = "models/pca_clusters.png"


@pytest.fixture(scope="session")
def cluster_report():
    """Carrega o relatório gerado pelo clustering_analysis.py."""
    if not os.path.exists(REPORT_PATH):
        pytest.skip(
            f"Relatório não encontrado em '{REPORT_PATH}'. "
            "Execute clustering_analysis.py antes dos testes."
        )
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Testes do Relatório JSON
# =============================================================================

class TestClusterReport:

    def test_report_file_exists(self):
        """O arquivo cluster_report.json deve existir após a análise."""
        assert os.path.exists(REPORT_PATH), (
            f"Relatório não encontrado: '{REPORT_PATH}'"
        )

    def test_report_has_required_keys(self, cluster_report):
        """O relatório deve conter todos os campos obrigatórios."""
        required = {"generated_at", "chosen_k", "features", "metrics",
                    "cluster_profiles", "elbow_inertias"}
        missing = required - set(cluster_report.keys())
        assert not missing, f"Campos ausentes no relatório: {missing}"

    def test_chosen_k_is_valid(self, cluster_report):
        """K deve ser >= 2."""
        assert cluster_report["chosen_k"] >= 2, (
            f"K inválido: {cluster_report['chosen_k']}"
        )

    def test_elbow_inertias_decrease_monotonically(self, cluster_report):
        """
        A inércia deve decrescer à medida que K aumenta.
        Uma curva crescente indica erro no cálculo ou dado inconsistente.
        """
        inertias = [v for v in cluster_report["elbow_inertias"].values()]
        for i in range(1, len(inertias)):
            assert inertias[i] < inertias[i - 1], (
                f"Inércia não decresce em K={i+2}: {inertias[i-1]} → {inertias[i]}"
            )

    def test_number_of_cluster_profiles_matches_k(self, cluster_report):
        """O número de perfis deve ser igual ao K escolhido."""
        k = cluster_report["chosen_k"]
        n_profiles = len(cluster_report["cluster_profiles"])
        assert n_profiles == k, (
            f"Esperado {k} perfis, encontrado {n_profiles}"
        )


# =============================================================================
# Testes de Métricas Internas
# =============================================================================

class TestInternalMetrics:

    def test_silhouette_above_threshold(self, cluster_report):
        """
        Silhouette Score deve ser >= 0.3.
        Abaixo disso, os clusters não têm separação útil.
        """
        score = cluster_report["metrics"]["internal"]["silhouette_score"]
        threshold = 0.3
        assert score >= threshold, (
            f"Silhouette abaixo do limiar: {score:.4f} < {threshold}. "
            "Os clusters podem estar sobrepostos."
        )

    def test_davies_bouldin_below_threshold(self, cluster_report):
        """Davies-Bouldin deve ser < 1.5 (quanto menor, melhor)."""
        db = cluster_report["metrics"]["internal"]["davies_bouldin_index"]
        threshold = 1.5
        assert db < threshold, (
            f"Davies-Bouldin acima do limiar: {db:.4f} >= {threshold}"
        )

    def test_calinski_harabasz_is_positive(self, cluster_report):
        """Calinski-Harabasz deve ser positivo."""
        ch = cluster_report["metrics"]["internal"]["calinski_harabasz_score"]
        assert ch > 0, f"Calinski-Harabasz negativo: {ch}"


# =============================================================================
# Testes de Métricas Externas
# =============================================================================

class TestExternalMetrics:

    def test_v_measure_above_threshold(self, cluster_report):
        """
        V-Measure (≈ F1-Score) deve ser >= 0.5.
        Indica que os clusters correlacionam razoavelmente com o label de churn.
        """
        v = cluster_report["metrics"]["external"]["v_measure"]
        threshold = 0.5
        assert v >= threshold, (
            f"V-Measure abaixo do threshold: {v:.4f} < {threshold}"
        )

    def test_homogeneity_and_completeness_present(self, cluster_report):
        """Homogeneidade e Completude devem existir e estar entre 0 e 1."""
        ext = cluster_report["metrics"]["external"]
        for key in ("homogeneity", "completeness"):
            val = ext[key]
            assert 0.0 <= val <= 1.0, f"{key} fora do intervalo [0, 1]: {val}"

    def test_adjusted_rand_index_present(self, cluster_report):
        """ARI deve existir e estar no intervalo válido [-1, 1]."""
        ari = cluster_report["metrics"]["external"]["adjusted_rand_index"]
        assert -1.0 <= ari <= 1.0, f"ARI fora do intervalo [-1, 1]: {ari}"


# =============================================================================
# Testes de Perfil de Negócio
# =============================================================================

class TestClusterProfiles:

    def test_profiles_have_required_fields(self, cluster_report):
        """Cada perfil deve conter campos de negócio obrigatórios."""
        required = {"n_samples", "churn_rate", "risk_label", "feature_means"}
        for name, profile in cluster_report["cluster_profiles"].items():
            missing = required - set(profile.keys())
            assert not missing, f"Perfil '{name}' sem campos: {missing}"

    def test_churn_rates_are_valid_probabilities(self, cluster_report):
        """Taxa de churn de cada cluster deve estar entre 0 e 1."""
        for name, profile in cluster_report["cluster_profiles"].items():
            rate = profile["churn_rate"]
            assert 0.0 <= rate <= 1.0, (
                f"Cluster '{name}' com churn_rate inválido: {rate}"
            )

    def test_risk_labels_are_valid(self, cluster_report):
        """risk_label deve ser 'ALTO RISCO' ou 'BAIXO RISCO'."""
        valid = {"ALTO RISCO", "BAIXO RISCO"}
        for name, profile in cluster_report["cluster_profiles"].items():
            assert profile["risk_label"] in valid, (
                f"Cluster '{name}' com rótulo inválido: '{profile['risk_label']}'"
            )

    def test_total_samples_consistent(self, cluster_report):
        """
        Soma das amostras em todos os clusters deve ser positiva.
        (Não temos o total original aqui, mas garante que não está zerado.)
        """
        total = sum(p["n_samples"] for p in cluster_report["cluster_profiles"].values())
        assert total > 0, "Total de amostras nos clusters é zero."

    def test_at_least_one_high_risk_cluster(self, cluster_report):
        """
        Deve existir pelo menos um cluster de ALTO RISCO.
        Se todos forem baixo risco, o modelo não está discriminando churners.
        """
        labels = [p["risk_label"] for p in cluster_report["cluster_profiles"].values()]
        assert "ALTO RISCO" in labels, (
            "Nenhum cluster classificado como ALTO RISCO. "
            "Verifique os dados e o K escolhido."
        )

    def test_at_least_one_low_risk_cluster(self, cluster_report):
        """Deve existir pelo menos um cluster de BAIXO RISCO."""
        labels = [p["risk_label"] for p in cluster_report["cluster_profiles"].values()]
        assert "BAIXO RISCO" in labels, (
            "Nenhum cluster classificado como BAIXO RISCO."
        )


# =============================================================================
# Testes dos Gráficos Gerados
# =============================================================================

class TestOutputFiles:

    def test_elbow_plot_exists(self):
        """O gráfico do Elbow Method deve ser gerado."""
        assert os.path.exists(ELBOW_PATH), f"Gráfico não encontrado: {ELBOW_PATH}"

    def test_silhouette_plot_exists(self):
        """O Silhouette Plot deve ser gerado."""
        assert os.path.exists(SILHOUETTE_PATH), f"Gráfico não encontrado: {SILHOUETTE_PATH}"

    def test_pca_plot_exists(self):
        """O gráfico PCA 2D deve ser gerado."""
        assert os.path.exists(PCA_PATH), f"Gráfico não encontrado: {PCA_PATH}"

    def test_plots_are_not_empty(self):
        """Os arquivos PNG não podem estar vazios (upload incompleto)."""
        for path in (ELBOW_PATH, SILHOUETTE_PATH, PCA_PATH):
            if os.path.exists(path):
                size = os.path.getsize(path)
                assert size > 1024, f"Arquivo suspeito — muito pequeno: {path} ({size} bytes)"
