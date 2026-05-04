"""
Module 4: RAGAS Evaluation — 4 metrics + Diagnostic-Tree failure analysis.

Owner: Lưu Lương Vi Nhân
Test: pytest tests/test_m4.py
"""

import os
import sys
import json
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


# ─── Helpers ──────────────────────────────────────────────


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set từ JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _heuristic_scores(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    """
    Fallback scoring (token-overlap heuristic) khi RAGAS hoặc API key không khả dụng.
    Đảm bảo pipeline luôn chạy được; đủ để so sánh tương đối.
    """

    def tokens(s: str) -> set[str]:
        return {t.lower().strip(",.;:?!") for t in s.split() if t.strip()}

    a_tok = tokens(answer)
    gt_tok = tokens(ground_truth)
    q_tok = tokens(question)
    ctx_tok = set().union(*[tokens(c) for c in contexts]) if contexts else set()

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    return {
        "faithfulness": jaccard(a_tok, ctx_tok),
        "answer_relevancy": jaccard(a_tok, q_tok | gt_tok),
        "context_precision": jaccard(ctx_tok, gt_tok) if ctx_tok else 0.0,
        "context_recall": (len(gt_tok & ctx_tok) / len(gt_tok)) if gt_tok else 0.0,
    }


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """
    Run RAGAS evaluation. Fallback về heuristic nếu RAGAS không khả dụng
    (thiếu OPENAI_API_KEY hoặc package).

    Returns:
        dict với 4 aggregate metrics + per_question (list[EvalResult]).
    """
    n = len(questions)
    per_question: list[EvalResult] = []
    aggregate = {"faithfulness": 0.0, "answer_relevancy": 0.0,
                 "context_precision": 0.0, "context_recall": 0.0}

    use_ragas = os.getenv("OPENAI_API_KEY", "").startswith("sk-")
    ragas_ok = False
    if use_ragas:
        try:
            from ragas import evaluate
            from ragas.metrics import (faithfulness, answer_relevancy,
                                        context_precision, context_recall)
            from datasets import Dataset

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                                 context_precision, context_recall])
            df = result.to_pandas()
            for _, row in df.iterrows():
                per_question.append(EvalResult(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    contexts=list(row["contexts"]) if row["contexts"] is not None else [],
                    ground_truth=str(row.get("ground_truth", "")),
                    faithfulness=float(row.get("faithfulness", 0) or 0),
                    answer_relevancy=float(row.get("answer_relevancy", 0) or 0),
                    context_precision=float(row.get("context_precision", 0) or 0),
                    context_recall=float(row.get("context_recall", 0) or 0),
                ))
            ragas_ok = True
        except Exception as e:
            print(f"[m4_eval] RAGAS unavailable ({e}); fallback to heuristic.")

    if not ragas_ok:
        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
            s = _heuristic_scores(q, a, ctx, gt)
            per_question.append(EvalResult(
                question=q, answer=a, contexts=ctx, ground_truth=gt,
                faithfulness=s["faithfulness"],
                answer_relevancy=s["answer_relevancy"],
                context_precision=s["context_precision"],
                context_recall=s["context_recall"],
            ))

    if per_question:
        for k in aggregate:
            aggregate[k] = round(sum(getattr(r, k) for r in per_question) / len(per_question), 4)

    return {**aggregate, "per_question": per_question}


# ─── Failure Analysis ─────────────────────────────────────


_DIAGNOSTIC_RULES = [
    ("faithfulness", 0.85, "LLM hallucinating",
     "Tighten prompt, lower temperature, ground answer strictly to context"),
    ("context_recall", 0.75, "Missing relevant chunks",
     "Improve chunking strategy hoặc thêm BM25 keyword search"),
    ("context_precision", 0.75, "Too many irrelevant chunks",
     "Add cross-encoder reranking hoặc metadata filter"),
    ("answer_relevancy", 0.80, "Answer doesn't match question",
     "Improve prompt template, include question rewrite step"),
]


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Sort by avg score, take bottom_n, map worst metric → diagnosis + fix."""
    if not eval_results:
        return []

    def avg(r: EvalResult) -> float:
        return (r.faithfulness + r.answer_relevancy
                + r.context_precision + r.context_recall) / 4.0

    sorted_results = sorted(eval_results, key=avg)
    bottom = sorted_results[:bottom_n]

    failures = []
    for r in bottom:
        scores = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(scores, key=scores.get)
        worst_score = scores[worst_metric]

        diagnosis = "Unknown failure mode"
        suggested_fix = "Inspect manually"
        for metric, threshold, diag, fix in _DIAGNOSTIC_RULES:
            if scores[metric] < threshold and metric == worst_metric:
                diagnosis = diag
                suggested_fix = fix
                break
        else:
            for metric, threshold, diag, fix in _DIAGNOSTIC_RULES:
                if scores[metric] < threshold:
                    diagnosis = diag
                    suggested_fix = fix
                    worst_metric = metric
                    worst_score = scores[metric]
                    break

        failures.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "all_scores": {k: round(v, 4) for k, v in scores.items()},
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return failures


# ─── Report ───────────────────────────────────────────────


def save_report(results: dict, failures: list[dict],
                path: str = "ragas_report.json") -> None:
    """Persist JSON report. (Đã có sẵn — giữ nguyên schema check_lab kỳ vọng.)"""
    aggregate = {k: v for k, v in results.items() if k != "per_question"}
    per_question = [asdict(r) for r in results.get("per_question", [])]
    report = {
        "aggregate": aggregate,
        "num_questions": len(results.get("per_question", [])),
        "per_question": per_question,
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
