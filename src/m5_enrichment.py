"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
import json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_client():
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


def _safe_json_loads(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()
    return json.loads(raw)


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    if not text.strip():
        return ""
    client = _get_client()
    if client is None:
        return ""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=120,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tóm tắt đoạn văn sau trong 1-2 câu ngắn gọn bằng tiếng Việt, "
                        "giữ nguyên số liệu/chính sách quan trọng."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    if not text.strip():
        return []
    client = _get_client()
    if client is None:
        return []

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=220,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. "
                        "Mỗi câu hỏi trên một dòng, tự nhiên, hữu ích cho truy hồi tài liệu."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
    except Exception:
        return []

    content = (resp.choices[0].message.content or "").strip()
    questions = []
    for line in content.splitlines():
        q = line.strip().lstrip("0123456789.-) ").strip()
        if not q:
            continue
        if not q.endswith("?"):
            q = f"{q}?"
        questions.append(q)

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for q in questions:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    if not text.strip():
        return text
    client = _get_client()
    if client is None:
        return text

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=80,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Viết đúng 1 câu tiếng Việt mô tả đoạn văn này thuộc phần nào và chủ đề gì trong tài liệu. "
                        "Ngắn gọn, giàu thông tin truy hồi."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Tài liệu: {document_title or 'Không rõ'}\n\nĐoạn văn:\n{text}",
                },
            ],
        )
        context = (resp.choices[0].message.content or "").strip()
        return f"{context}\n\n{text}" if context else text
    except Exception:
        return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    if not text.strip():
        return {}
    client = _get_client()
    if client is None:
        return {}

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=180,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Trích xuất metadata từ đoạn văn và trả về JSON hợp lệ với đúng keys: "
                        "topic (string), entities (array of strings), category (policy|hr|it|finance|legal|other), "
                        "language (vi|en)."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        data = _safe_json_loads(content)
    except Exception:
        return {}
    return {
        "topic": data.get("topic", ""),
        "entities": data.get("entities", []) if isinstance(data.get("entities", []), list) else [],
        "category": data.get("category", "other"),
        "language": data.get("language", "vi"),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    active = set(methods)
    if "full" in active:
        active.update({"summary", "hyqa", "contextual", "metadata"})

    for chunk in chunks:
        text = chunk.get("text", "")
        base_meta = chunk.get("metadata", {})

        summary = summarize_chunk(text) if "summary" in active else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in active else []
        contextual_text = (
            contextual_prepend(text, base_meta.get("source", ""))
            if "contextual" in active else text
        )
        auto_meta = extract_metadata(text) if "metadata" in active else {}

        # Build embedding-optimized enriched text.
        segments = []
        if summary:
            segments.append(f"[Tóm tắt]\n{summary}")
        if questions:
            q_block = "\n".join(f"- {q}" for q in questions)
            segments.append(f"[Câu hỏi liên quan]\n{q_block}")
        segments.append(f"[Nội dung]\n{contextual_text}")
        enriched_text = "\n\n".join(segments).strip()

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text or text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**base_meta, **auto_meta},
                method="+".join(sorted(active)),
            )
        )

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
