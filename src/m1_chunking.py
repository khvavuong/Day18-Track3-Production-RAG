"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
import warnings
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)

# Suppress known upstream FutureWarning from transformers tokenizer defaults.
warnings.filterwarnings(
    "ignore",
    message=r".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


_SEMANTIC_MODEL = None


def _semantic_model():
    """Lazily load and cache sentence transformer model."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units for semantic chunking."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]


def _cosine_sim(vec_a, vec_b) -> float:
    """Compute cosine similarity for two vectors."""
    from numpy import dot
    from numpy.linalg import norm

    denom = (norm(vec_a) * norm(vec_b))
    if denom == 0:
        return 0.0
    return float(dot(vec_a, vec_b) / denom)


def _stats(lengths: list[int]) -> dict:
    """Return safe stats for chunk lengths."""
    if not lengths:
        return {"num_chunks": 0, "avg_length": 0.0, "min_length": 0, "max_length": 0}
    return {
        "num_chunks": len(lengths),
        "avg_length": round(sum(lengths) / len(lengths), 2),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def _chunk_meta(metadata: dict, index: int, strategy: str) -> dict:
    """Build common metadata fields for non-hierarchical chunks."""
    return {**metadata, "chunk_index": index, "strategy": strategy}


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata=_chunk_meta(metadata, 0, "semantic"))]

    embeddings = _semantic_model().encode(sentences, show_progress_bar=False)

    chunks: list[Chunk] = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(
                Chunk(
                    text=" ".join(current_group).strip(),
                    metadata=_chunk_meta(metadata, len(chunks), "semantic"),
                )
            )
            current_group = []
        current_group.append(sentences[i])

    if current_group:
        chunks.append(
            Chunk(
                text=" ".join(current_group).strip(),
                metadata=_chunk_meta(metadata, len(chunks), "semantic"),
            )
        )

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [], []

    parents: list[Chunk] = []
    children: list[Chunk] = []

    current_parent_parts: list[str] = []
    current_len = 0

    def flush_parent() -> None:
        nonlocal current_parent_parts, current_len
        if not current_parent_parts:
            return
        parent_text = "\n\n".join(current_parent_parts).strip()
        pid = f"parent_{len(parents)}"
        parent_chunk = Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
        )
        parents.append(parent_chunk)

        # Split parent into fixed-size children.
        for start in range(0, len(parent_text), child_size):
            child_text = parent_text[start:start + child_size].strip()
            if not child_text:
                continue
            children.append(
                Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                    parent_id=pid,
                )
            )

        current_parent_parts = []
        current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current_parent_parts and current_len + para_len > parent_size:
            flush_parent()
        current_parent_parts.append(para)
        current_len += para_len

    flush_parent()
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    for part in sections:
        if not part:
            continue
        if re.match(r'^#{1,3}\s+', part.strip()):
            if current_content.strip():
                header_label = current_header if current_header else "Preamble"
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **metadata,
                            "section": header_label,
                            "strategy": "structure",
                            "chunk_index": len(chunks),
                        },
                    )
                )
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    if current_content.strip() or current_header:
        header_label = current_header if current_header else "Preamble"
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "section": header_label,
                        "strategy": "structure",
                        "chunk_index": len(chunks),
                    },
                )
            )

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    basic_lengths: list[int] = []
    semantic_lengths: list[int] = []
    structure_lengths: list[int] = []
    parent_lengths: list[int] = []
    child_lengths: list[int] = []

    for doc in documents:
        text = doc.get("text", "")
        meta = doc.get("metadata", {})

        basic_chunks = chunk_basic(text, metadata=meta)
        semantic_chunks = chunk_semantic(text, metadata=meta)
        parents, children = chunk_hierarchical(text, metadata=meta)
        structure_chunks = chunk_structure_aware(text, metadata=meta)

        basic_lengths.extend(len(c.text) for c in basic_chunks)
        semantic_lengths.extend(len(c.text) for c in semantic_chunks)
        structure_lengths.extend(len(c.text) for c in structure_chunks)
        parent_lengths.extend(len(c.text) for c in parents)
        child_lengths.extend(len(c.text) for c in children)

    results = {
        "basic": _stats(basic_lengths),
        "semantic": _stats(semantic_lengths),
        "hierarchical": {
            "num_parents": len(parent_lengths),
            "num_children": len(child_lengths),
            "avg_parent_length": round(sum(parent_lengths) / len(parent_lengths), 2) if parent_lengths else 0.0,
            "avg_child_length": round(sum(child_lengths) / len(child_lengths), 2) if child_lengths else 0.0,
            "min_parent_length": min(parent_lengths) if parent_lengths else 0,
            "max_parent_length": max(parent_lengths) if parent_lengths else 0,
            "min_child_length": min(child_lengths) if child_lengths else 0,
            "max_child_length": max(child_lengths) if child_lengths else 0,
        },
        "structure": _stats(structure_lengths),
    }

    print("\nStrategy      | Chunks        | Avg Len")
    print("-" * 44)
    print(f"basic         | {results['basic']['num_chunks']:<13} | {results['basic']['avg_length']}")
    print(f"semantic      | {results['semantic']['num_chunks']:<13} | {results['semantic']['avg_length']}")
    print(
        f"hierarchical  | {results['hierarchical']['num_parents']}p/{results['hierarchical']['num_children']}c"
        f"{'':<6} | {results['hierarchical']['avg_child_length']} (child)"
    )
    print(f"structure     | {results['structure']['num_chunks']:<13} | {results['structure']['avg_length']}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
