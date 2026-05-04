"""
Production RAG Pipeline — Bài tập NHÓM.
Ghép M1 (chunking) → M5 (enrichment) → M2 (hybrid search) → M3 (rerank) → LLM → M4 (eval).

Owner: Lưu Lương Vi Nhân — 2A202600120 (group lead, pipeline integration)
Run:   python src/pipeline.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY


# ─── LLM generation ───────────────────────────────────────

_openai_client = None


def _get_llm():
    global _openai_client
    if _openai_client is not None:
        return _openai_client or None
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        _openai_client = False
        return None
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _openai_client = False
    return _openai_client or None


def generate_answer(query: str, contexts: list[str]) -> str:
    """
    Sinh câu trả lời từ top-k contexts. Fallback về top-1 context nếu không có LLM.
    Prompt groundé: trả lời CHỈ từ context, copy nguyên văn số liệu.
    """
    if not contexts:
        return "Không tìm thấy thông tin trong tài liệu."

    client = _get_llm()
    if client is None:
        return contexts[0]

    context_str = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý chính sách nội bộ. Trả lời câu hỏi CHỈ dựa trên CONTEXT được cung cấp. "
                        "Quy tắc bắt buộc:\n"
                        "- Nếu câu hỏi hỏi 'bao nhiêu', 'ngày nào', 'tháng nào' → PHẢI trích dẫn con số chính xác từ context.\n"
                        "- Nếu không tìm thấy thông tin trong context → trả lời 'Không tìm thấy thông tin trong tài liệu.'\n"
                        "- Không suy luận, không thêm thông tin ngoài context.\n"
                        "- Trả lời ngắn gọn, đúng trọng tâm (1-3 câu)."
                    ),
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context_str}\n\nCÂU HỎI: {query}",
                },
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  [pipeline] LLM error: {e} — fallback context.")
        return contexts[0]


# ─── Build pipeline ──────────────────────────────────────


def build_pipeline(use_enrichment: bool = True):
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("M1 → M5 → M2 (BM25+Dense+RRF) → M3 (CrossEncoder) → LLM → M4")
    print("=" * 60)
    timings: dict[str, float] = {}

    # ── Step 1: Chunk (M1 — Khuất Văn Vương) ──
    t0 = time.perf_counter()
    print("\n[1/4] Chunking (M1 — hierarchical)...")
    docs = load_documents()
    all_chunks: list[dict] = []
    for doc in docs:
        _, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })
    timings["chunk_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  {len(all_chunks)} chunks from {len(docs)} docs  ({timings['chunk_ms']:.0f} ms)")

    # ── Step 2: Enrichment (M5 — Khuất Văn Vương, bonus) ──
    t0 = time.perf_counter()
    if use_enrichment:
        print("\n[2/4] Enrichment (M5 — contextual + metadata)...")
        try:
            enriched = enrich_chunks(all_chunks, methods=["contextual", "metadata"])
            if enriched:
                all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata}
                              for e in enriched]
                print(f"  Enriched {len(enriched)} chunks")
            else:
                print("  ⚠  No OPENAI_API_KEY — skipping enrichment, using raw chunks.")
        except Exception as e:
            print(f"  ⚠  Enrichment error: {e} — using raw chunks.")
    else:
        print("\n[2/4] Enrichment skipped.")
    timings["enrich_ms"] = (time.perf_counter() - t0) * 1000

    # ── Step 3: Index + build search (M2 — Lê Hà An) ──
    t0 = time.perf_counter()
    print("\n[3/4] Indexing (M2 — BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks)
    timings["index_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  Indexed {len(all_chunks)} chunks  ({timings['index_ms']:.0f} ms)")

    # ── Step 4: Load reranker (M3 — Huỳnh Văn Nghĩa) ──
    print("\n[4/4] Loading reranker (M3 — CrossEncoder bge-reranker-v2-m3)...")
    reranker = CrossEncoderReranker()

    print("\n  Pipeline ready.\n")
    return search, reranker, timings


# ─── Single query ────────────────────────────────────────


def run_query(
    query: str,
    search: HybridSearch,
    reranker: CrossEncoderReranker,
    latency: dict | None = None,
) -> tuple[str, list[str]]:
    t0 = time.perf_counter()
    results = search.search(query)
    t1 = time.perf_counter()

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    t2 = time.perf_counter()

    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:RERANK_TOP_K]]
    answer = generate_answer(query, contexts)
    t3 = time.perf_counter()

    if latency is not None:
        latency.setdefault("search_ms", []).append((t1 - t0) * 1000)
        latency.setdefault("rerank_ms", []).append((t2 - t1) * 1000)
        latency.setdefault("generate_ms", []).append((t3 - t2) * 1000)

    return answer, contexts


# ─── Evaluation ─────────────────────────────────────────


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    print("[Eval] Running queries on test set...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []
    latency: dict[str, list] = {}

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, latency=latency)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        if (i + 1) % 5 == 0 or (i + 1) == len(test_set):
            print(f"  [{i+1}/{len(test_set)}] done")

    print("\n[Eval] Computing RAGAS metrics (M4 — Lưu Lương Vi Nhân)...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0.0)
        mark = "✓" if s >= 0.75 else "✗"
        print(f"  {mark} {m:<25}: {s:.4f}")

    # Latency breakdown (bonus)
    if latency:
        print("\nLATENCY BREAKDOWN (avg ms / query)")
        for stage, vals in latency.items():
            if vals:
                print(f"  {stage:<15}: {sum(vals)/len(vals):.1f} ms")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, build_timings = build_pipeline(use_enrichment=True)
    evaluate_pipeline(search, reranker)
    print(f"\nTotal elapsed: {time.time() - start:.1f}s")
