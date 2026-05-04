"""
Module 2: Hybrid Search — BM25 (Vietnamese) + Dense (bge-m3) + RRF.

Owner: Thành viên 2
Test: pytest tests/test_m2.py
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


# ─── Vietnamese segmentation ───────────────────────────────


def segment_vietnamese(text: str) -> str:
    """
    Word-tokenize tiếng Việt bằng underthesea.
    "nghỉ phép" -> "nghỉ_phép" (token đơn) → cải thiện BM25.
    """
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        # Fallback: trả nguyên text (BM25 sẽ tách theo whitespace)
        return text


def _bm25_tokenize(text: str) -> list[str]:
    """
    Tokenize for BM25: segment → expand compound tokens.
    "nghỉ_phép" → ["nghỉ_phép", "nghỉ", "phép"]
    Đảm bảo query ngắn (chưa có đủ context cho underthesea) vẫn khớp với
    compound tokens đã index trong corpus.
    """
    segmented = segment_vietnamese(text).lower()
    tokens: list[str] = []
    for tok in segmented.split():
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(tok.split("_"))
    return tokens


# ─── BM25 ─────────────────────────────────────────────────


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Index chunks with Vietnamese-aware BM25Okapi."""
        from rank_bm25 import BM25Okapi
        self.documents = chunks
        self.corpus_tokens = [_bm25_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if self.bm25 is None or not self.documents:
            return []
        tokenized_query = _bm25_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


# ─── Dense (bge-m3 + Qdrant) ──────────────────────────────


class DenseSearch:
    def __init__(self):
        self._client = None
        self._encoder = None
        self._inmem_vectors: list = []
        self._inmem_docs: list = []

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
                # Probe to fail fast if Qdrant unavailable
                self._client.get_collections()
            except Exception:
                self._client = False
        return self._client or None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                self._encoder = False
        return self._encoder or None

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        encoder = self._get_encoder()
        if encoder is None:
            return  # Cannot index without encoder

        texts = [c["text"] for c in chunks]
        vectors = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        client = self._get_client()
        if client is None:
            # Fallback: in-memory cosine search
            self._inmem_vectors = list(vectors)
            self._inmem_docs = chunks
            return

        from qdrant_client.models import Distance, VectorParams, PointStruct
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=i,
                vector=v.tolist() if hasattr(v, "tolist") else list(v),
                payload={**c.get("metadata", {}), "text": c["text"]},
            )
            for i, (v, c) in enumerate(zip(vectors, chunks))
        ]
        client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        encoder = self._get_encoder()
        if encoder is None:
            return []
        qv = encoder.encode(query, normalize_embeddings=True)

        client = self._get_client()
        if client is None:
            # In-memory fallback
            if not self._inmem_vectors:
                return []
            import numpy as np
            mat = np.array(self._inmem_vectors)
            scores = mat @ np.array(qv)
            top = scores.argsort()[::-1][:top_k]
            return [
                SearchResult(
                    text=self._inmem_docs[i]["text"],
                    score=float(scores[i]),
                    metadata=self._inmem_docs[i].get("metadata", {}),
                    method="dense",
                )
                for i in top
            ]

        hits = client.search(
            collection_name=collection,
            query_vector=qv.tolist() if hasattr(qv, "tolist") else list(qv),
            limit=top_k,
        )
        return [
            SearchResult(
                text=h.payload.get("text", ""),
                score=float(h.score),
                metadata={k: v for k, v in h.payload.items() if k != "text"},
                method="dense",
            )
            for h in hits
        ]


# ─── Reciprocal Rank Fusion ───────────────────────────────


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """RRF: score(d) = Σ 1/(k + rank_i(d))."""
    bucket: dict[str, dict] = {}
    for results in results_list:
        for rank, r in enumerate(results):
            entry = bucket.setdefault(r.text, {"score": 0.0, "result": r, "metadata": r.metadata})
            entry["score"] += 1.0 / (k + rank + 1)
    merged = sorted(bucket.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=item["score"],
            metadata=item["metadata"],
            method="hybrid",
        )
        for item in merged
    ]


# ─── HybridSearch facade ──────────────────────────────────


class HybridSearch:
    """Combines BM25 + Dense + RRF."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        if not bm25_results and not dense_results:
            return []
        if not bm25_results:
            return [SearchResult(r.text, r.score, r.metadata, "hybrid") for r in dense_results[:top_k]]
        if not dense_results:
            return [SearchResult(r.text, r.score, r.metadata, "hybrid") for r in bm25_results[:top_k]]
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    sample = "Nhân viên được nghỉ phép năm"
    print(f"Original:  {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")
