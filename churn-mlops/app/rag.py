"""
=============================================================================
MÓDULO RAG — Retrieval-Augmented Generation para Perfis de Cluster
=============================================================================

Implementa o passo de RAG descrito na Fase 9 do CRISP-DM e no diagrama
clustering_rag_guide: dado um perfil de cliente (segmento + cluster_id),
recupera os documentos mais relevantes do Vector Store (profile_cards)
para compor o contexto de uma resposta contextualizada.

ARQUITETURA:
  1. INDEXAÇÃO  : profile_cards.json → documentos de texto → TF-IDF matrix
  2. RETRIEVAL  : cosine similarity entre query e documentos indexados
  3. RESPOSTA   : top-k perfis retornados como contexto estruturado

NOTA SOBRE FAISS:
  A versão de produção deve substituir TF-IDF por embeddings densos +
  faiss-cpu para escala. O módulo tenta importar faiss e usa o índice
  IVF se disponível; caso contrário, cai no backend TF-IDF (sklearn).

Artefatos esperados:
  models/profile_cards.json  — gerado por clustering_analysis.py

Uso:
  from app.rag import get_engine
  engine = get_engine()
  results = engine.query("cliente insatisfeito com suporte PJ", segmento="PJ")
=============================================================================
"""

import json
import logging
import os
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("churn_api.rag")

# ---------------------------------------------------------------------------
# Caminhos padrão (sobrepõem por variável de ambiente)
# ---------------------------------------------------------------------------
PROFILE_CARDS_PATH_DEFAULT = os.environ.get(
    "PROFILE_CARDS_PATH", "models/profile_cards.json"
)

# ---------------------------------------------------------------------------
# Tentativa de importar FAISS (opcional — fallback para TF-IDF se ausente)
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
    logger.info("FAISS disponível — backend vetorial: faiss-cpu")
except ImportError:
    _FAISS_AVAILABLE = False
    logger.info("FAISS não disponível — usando backend TF-IDF (sklearn)")


# =============================================================================
# RAGEngine
# =============================================================================
class RAGEngine:
    """
    Motor de busca semântica sobre os perfis de cluster (profile_cards.json).

    Suporta dois backends:
      - TF-IDF + cosine similarity (padrão, zero deps extras)
      - FAISS (se faiss-cpu instalado)

    Attributes:
        profile_cards : Dicionário completo carregado do JSON.
        _loaded       : True após load() ser chamado com sucesso.
    """

    def __init__(self, profile_cards_path: str = PROFILE_CARDS_PATH_DEFAULT):
        self.profile_cards: dict = {}
        self._documents:    list[str] = []   # texto de cada perfil
        self._doc_meta:     list[dict] = []  # metadados por documento
        self._vectorizer:   Optional[TfidfVectorizer] = None
        self._tfidf_matrix  = None

        # FAISS internals (apenas se disponível)
        self._faiss_index   = None
        self._faiss_vecs:   Optional[np.ndarray] = None

        self._loaded = False
        self._profile_cards_path = profile_cards_path

        if os.path.exists(profile_cards_path):
            self.load(profile_cards_path)
        else:
            logger.warning(
                "profile_cards.json não encontrado em '%s'. "
                "Execute clustering_analysis.py para gerá-lo.",
                profile_cards_path,
            )

    # ── Carregamento ─────────────────────────────────────────────────────────

    def load(self, path: str) -> None:
        """Carrega profile_cards.json e constrói o índice vetorial."""
        with open(path, encoding="utf-8") as f:
            self.profile_cards = json.load(f)
        logger.info("profile_cards.json carregado: %d segmentos", len(self.profile_cards))
        self._build_index()
        self._loaded = True

    def reload(self) -> None:
        """Recarrega o índice — útil após atualização do profile_cards.json."""
        if os.path.exists(self._profile_cards_path):
            self.load(self._profile_cards_path)

    # ── Construção do índice ─────────────────────────────────────────────────

    def _profile_to_text(self, profile: dict) -> str:
        """Serializa um perfil de cluster em texto pesquisável."""
        means = profile.get("feature_means", {})
        parts = [
            profile.get("perfil_label", ""),
            f"segmento {profile.get('segmento', '')}",
            f"risco {profile.get('risk_level', '')}",
            profile.get("description", ""),
            profile.get("action_recommendation", ""),
            " ".join(profile.get("rag_tags", [])),
            f"tenure {means.get('tenure_months', 0):.0f} meses",
            f"cobranca mensal {means.get('monthly_charges', 0):.0f}",
            f"chamados suporte {means.get('support_calls', 0):.0f}",
            f"atraso pagamento {means.get('payment_delay_days', 0):.0f} dias",
            f"satisfacao {means.get('satisfaction_score', 0):.1f}",
            f"churn rate {profile.get('churn_rate', 0):.2f}",
        ]
        return " ".join(str(p) for p in parts if p)

    def _build_index(self) -> None:
        """Constrói o índice TF-IDF (e opcionalmente FAISS) sobre todos os perfis."""
        self._documents = []
        self._doc_meta  = []

        for seg, clusters in self.profile_cards.items():
            for ckey, profile in clusters.items():
                self._documents.append(self._profile_to_text(profile))
                self._doc_meta.append(
                    {
                        "segmento":     seg,
                        "cluster_key":  ckey,
                        "cluster_id":   profile.get("cluster_id"),
                        "perfil_label": profile.get("perfil_label", ""),
                        "risk_level":   profile.get("risk_level", ""),
                    }
                )

        if not self._documents:
            logger.warning("Nenhum documento para indexar.")
            return

        # ── TF-IDF (sempre construído como fallback / único) ─────────────
        self._vectorizer = TfidfVectorizer(
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._documents)
        logger.info(
            "Índice TF-IDF construído: %d documentos × %d termos",
            self._tfidf_matrix.shape[0],
            self._tfidf_matrix.shape[1],
        )

        # ── FAISS (se disponível, complementa o TF-IDF com L2) ──────────
        if _FAISS_AVAILABLE:
            self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        """Constrói índice FAISS sobre os vetores TF-IDF densos."""
        dense = self._tfidf_matrix.toarray().astype("float32")  # type: ignore[union-attr]
        # Normaliza para L2 (equivale a cosine similarity com IndexFlatIP)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense /= norms
        self._faiss_vecs = dense

        dim = dense.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)  # type: ignore[name-defined]
        self._faiss_index.add(dense)
        logger.info("Índice FAISS construído: dim=%d, n=%d", dim, len(dense))

    # ── Consulta ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        segmento: Optional[str] = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Busca por similaridade semântica nos perfis indexados.

        Args:
            question : Texto da consulta (pergunta livre ou descrição do cliente).
            segmento : Filtro opcional por segmento — 'PF' ou 'PJ'.
            top_k    : Número máximo de resultados a retornar.

        Returns:
            Lista de dicts com perfil + relevance_score, ordenados por relevância.
        """
        if not self._loaded or not self._documents:
            logger.warning("RAGEngine não carregado. Retornando lista vazia.")
            return []

        if _FAISS_AVAILABLE and self._faiss_index is not None:
            return self._query_faiss(question, segmento, top_k)
        return self._query_tfidf(question, segmento, top_k)

    def _query_tfidf(
        self,
        question: str,
        segmento: Optional[str],
        top_k: int,
    ) -> list[dict]:
        """Backend TF-IDF: cosine similarity entre query e documentos."""
        q_vec   = self._vectorizer.transform([question])  # type: ignore[union-attr]
        scores  = cosine_similarity(q_vec, self._tfidf_matrix).flatten()  # type: ignore[arg-type]
        ranked  = np.argsort(scores)[::-1]

        results = []
        for idx in ranked:
            meta = self._doc_meta[idx]
            if segmento and meta["segmento"] != segmento:
                continue
            results.append(self._build_result(idx, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def _query_faiss(
        self,
        question: str,
        segmento: Optional[str],
        top_k: int,
    ) -> list[dict]:
        """Backend FAISS: inner-product (equivalente a cosine com norma L2)."""
        q_vec = self._vectorizer.transform([question]).toarray().astype("float32")  # type: ignore
        norm  = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        # Busca top_k * 4 para permitir filtrar por segmento depois
        k     = min(top_k * 4, len(self._documents))
        scores, indices = self._faiss_index.search(q_vec, k)  # type: ignore

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._doc_meta[idx]
            if segmento and meta["segmento"] != segmento:
                continue
            results.append(self._build_result(int(idx), float(score)))
            if len(results) >= top_k:
                break
        return results

    def _build_result(self, idx: int, score: float) -> dict:
        """Monta o dict de resultado para um documento indexado."""
        meta    = self._doc_meta[idx]
        profile = (
            self.profile_cards
            .get(meta["segmento"], {})
            .get(meta["cluster_key"], {})
        )
        return {
            "segmento":        meta["segmento"],
            "cluster_id":      meta["cluster_id"],
            "cluster_key":     meta["cluster_key"],
            "perfil_label":    meta["perfil_label"],
            "risk_level":      meta["risk_level"],
            "relevance_score": round(score, 4),
            "profile":         profile,
        }

    # ── Lookup direto ─────────────────────────────────────────────────────────

    def get_profile(self, segmento: str, cluster_id: int) -> Optional[dict]:
        """
        Retorna o perfil de um cluster específico sem fazer busca vetorial.

        Usado pelo endpoint /predict para enriquecer a resposta com o
        perfil correspondente ao cluster atribuído.

        Args:
            segmento   : 'PF' ou 'PJ'
            cluster_id : ID numérico do cluster (0-based)

        Returns:
            Dict com os dados do perfil ou None se não encontrado.
        """
        return (
            self.profile_cards
            .get(segmento, {})
            .get(f"cluster_{cluster_id}")
        )

    def list_profiles(self, segmento: Optional[str] = None) -> list[dict]:
        """Retorna todos os perfis disponíveis, opcionalmente filtrados por segmento."""
        result = []
        for seg, clusters in self.profile_cards.items():
            if segmento and seg != segmento:
                continue
            for profile in clusters.values():
                result.append(profile)
        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return "faiss" if (_FAISS_AVAILABLE and self._faiss_index is not None) else "tfidf"


# =============================================================================
# Singleton thread-safe para uso no FastAPI
# =============================================================================
_engine: Optional[RAGEngine] = None


def get_engine(
    profile_cards_path: str = PROFILE_CARDS_PATH_DEFAULT,
    force_reload: bool = False,
) -> RAGEngine:
    """
    Retorna (ou cria) a instância singleton do RAGEngine.

    Chamado durante o lifespan do FastAPI para garantir que o índice
    seja construído uma única vez no startup e reutilizado por todas
    as requisições.

    Args:
        profile_cards_path : Caminho para o profile_cards.json.
        force_reload       : Se True, reconstrói o índice (ex: após re-treino).

    Returns:
        RAGEngine com índice carregado.
    """
    global _engine
    if _engine is None or force_reload:
        _engine = RAGEngine(profile_cards_path)
    return _engine
