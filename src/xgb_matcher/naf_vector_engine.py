"""
naf_vector_engine.py — Continuous Semantic Vector Space for INSEE NAF Projection.

Maps unstructured commercial, artisanal and liberal trade names into official
INSEE NAF/APE activity codes using sublinear n-gram vector projection and
cosine similarity, without requiring static manual keyword dictionaries.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .naf_taxonomy import INSEE_NAF_DESCRIPTIONS


def _normalize_text(text: str | None) -> str:
    """Strip accents, lowercase, de-concatenate alphanumeric, clean punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d+)([a-zA-Z]+)", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


class NafVectorEngine:
    """
    Continuous semantic vector engine for official French NAF nomenclature.
    Precomputes an authoritative vector space from INSEE activity descriptions.
    """

    _instance: Optional["NafVectorEngine"] = None

    def __init__(self) -> None:
        self.naf_codes: List[str] = []
        self.descriptions: List[str] = []
        for code, desc in INSEE_NAF_DESCRIPTIONS.items():
            self.naf_codes.append(code)
            self.descriptions.append(_normalize_text(f"{desc} {code}"))

        # Multigram sublinear TF-IDF vector space with L2 unit normalization
        stopwords = [
            "de", "du", "des", "la", "le", "les", "et", "en", "un", "une",
            "d", "l", "au", "aux", "pour", "sur", "sous", "par", "dans", "autres",
            "autre", "tous", "tout", "toutes", "locaux"
        ]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            stop_words=stopwords,
        )
        self.matrix = self.vectorizer.fit_transform(self.descriptions)
        self._clean_to_idx = {code.replace(".", "").strip().upper(): idx for idx, code in enumerate(self.naf_codes)}
        self._code_to_idx = {code.strip().upper(): idx for idx, code in enumerate(self.naf_codes)}

    @classmethod
    def get_instance(cls) -> "NafVectorEngine":
        """Singleton accessor to ensure the vector space is loaded only once."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_naf_similarity(self, query_text: str, naf_code: str) -> float:
        """
        Compute continuous cosine similarity between query text and a specific NAF code.
        Returns 0.0 if naf_code is unknown or query is empty.
        """
        if not naf_code or not query_text:
            return 0.0

        clean_code = str(naf_code).replace(".", "").strip().upper()
        idx = self._clean_to_idx.get(clean_code)
        if idx is None:
            idx = self._code_to_idx.get(str(naf_code).strip().upper())
            if idx is None:
                return 0.0

        clean_query = _normalize_text(query_text)
        if not clean_query:
            return 0.0

        try:
            q_vec = self.vectorizer.transform([clean_query])
            sim = float((q_vec @ self.matrix[idx].T)[0, 0])
            return max(0.0, sim)
        except Exception:
            return 0.0

    def get_batch_naf_similarities(self, query_text: str, naf_codes: List[str]) -> List[float]:
        """
        Compute continuous cosine similarity for a batch of candidate NAF codes against query text.
        Projects query text once, then performs fast array lookups.
        """
        if not query_text or not naf_codes:
            return [0.0] * len(naf_codes)

        clean_query = _normalize_text(query_text)
        if not clean_query:
            return [0.0] * len(naf_codes)

        try:
            q_vec = self.vectorizer.transform([clean_query])
            sims = (q_vec @ self.matrix.T).toarray()[0]
        except Exception:
            return [0.0] * len(naf_codes)

        results: List[float] = []
        for code in naf_codes:
            if not code:
                results.append(0.0)
                continue
            clean_code = str(code).replace(".", "").strip().upper()
            idx = self._clean_to_idx.get(clean_code)
            if idx is None:
                idx = self._code_to_idx.get(str(code).strip().upper())
            if idx is not None and 0 <= idx < len(sims):
                results.append(max(0.0, float(sims[idx])))
            else:
                results.append(0.0)
        return results

    def predict_top_naf(
        self,
        query_text: str,
        top_k: int = 3,
        min_similarity: float = 0.12,
    ) -> List[Tuple[str, float]]:
        """
        Project query text into NAF vector space and return closest NAF codes
        with cosine similarity scores.
        
        Args:
            query_text: Raw CRM business/trade name.
            top_k: Maximum number of closest NAF codes to return.
            min_similarity: Minimum cosine similarity threshold.
            
        Returns:
            List of (naf_code, similarity_score) sorted descending by similarity.
        """
        clean_query = _normalize_text(query_text)
        if not clean_query:
            return []

        try:
            q_vec = self.vectorizer.transform([clean_query])
            sims = (q_vec @ self.matrix.T).toarray()[0]
        except Exception:
            return []

        if not np.any(sims > 0):
            return []

        # Find top_k indices exceeding min_similarity
        best_indices = np.argsort(sims)[::-1][:top_k]
        results: List[Tuple[str, float]] = []
        for idx in best_indices:
            score = float(sims[idx])
            if score >= min_similarity:
                results.append((self.naf_codes[idx], round(score, 4)))

        return results


def predict_naf(
    query_text: str,
    top_k: int = 3,
    min_similarity: float = 0.12,
) -> List[Tuple[str, float]]:
    """Functional convenience wrapper for NafVectorEngine."""
    return NafVectorEngine.get_instance().predict_top_naf(
        query_text=query_text,
        top_k=top_k,
        min_similarity=min_similarity,
    )
