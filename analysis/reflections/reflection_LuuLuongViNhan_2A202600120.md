# Individual Reflection — Lab 18: Production RAG

**Tên:** Lưu Lương Vi Nhân
**MSSV:** 2A202600120
**Module phụ trách:** M4 (RAGAS Evaluation) + Pipeline Integration
**Ngày:** 2026-05-04

---

## 1. Đóng góp kỹ thuật

### Các hàm/class chính đã viết

**`src/m4_eval.py`**
- `load_test_set()` — đọc `test_set.json`, validate 20 Q&A pairs
- `evaluate_ragas()` — chạy RAGAS gốc khi có `OPENAI_API_KEY` (faithfulness, answer_relevancy, context_precision, context_recall), fallback về heuristic Jaccard khi không có key
- `_heuristic_scores()` — tính 4 metrics offline qua Jaccard overlap (dùng khi không có API key)
- `failure_analysis()` — sort per-question scores, xác định bottom-N, áp Diagnostic Tree (4 rule map từ worst metric → root cause)
- `save_report()` — lưu `{aggregate, num_questions, per_question, failures}` vào JSON

**`src/pipeline.py`**
- `generate_answer()` — gpt-4o-mini, temperature=0.0, grounded prompt ("PHẢI trích dẫn con số chính xác"), fallback về contexts[0] khi không có key
- `build_pipeline()` — orchestrate M1 → M5 → M2 → M3 với per-step timing dict
- `run_query()` — single query: search → rerank → generate, track latency (search_ms, rerank_ms, generate_ms)
- `evaluate_pipeline()` — chạy 20 queries, gọi M4, in bảng RAGAS + latency breakdown, gọi `save_report()`

### Số tests pass
- M4: **4/4** tests pass (`tests/test_m4.py`)
- Pipeline: chạy end-to-end thành công 20/20 queries với real LLM

---

## 2. Kết quả thực tế

### Scores thu được (run ngày 2026-05-04, gpt-4o-mini, RAGAS v0.1.21)

| Metric | Naive Baseline | Production | Δ | Pass ≥ 0.75? |
|--------|---------------|------------|---|-------------|
| Faithfulness | 0.9750 | **0.8750** | −0.1000 | ✓ |
| Answer Relevancy | 0.7922 | **0.7464** | −0.0458 | ✗ (−0.004) |
| Context Precision | 0.9833 | **0.9750** | −0.0083 | ✓ |
| Context Recall | 1.0000 | **1.0000** | 0.0000 | ✓ |

**3/4 metrics đạt ≥ 0.75.**

### Latency Breakdown

| Stage | Avg ms/query |
|-------|-------------|
| Search (M2: BM25+Dense+RRF) | 477.8 ms |
| Rerank (M3: CrossEncoder) | 4686.1 ms |
| Generate (gpt-4o-mini) | 1263.2 ms |
| **Total** | **~6.4 s/query** |

---

## 3. Kiến thức học được

### Khái niệm mới nhất: RAGAS 4 metrics và ý nghĩa thực tế

Trước bài lab, tôi nghĩ RAG evaluation chỉ là "retrieval có tìm đúng không". Sau khi implement M4, tôi hiểu từng metric đo một chiều riêng biệt:

- **Faithfulness**: LLM có bịa đặt ngoài context không? (= grounding)
- **Answer Relevancy**: Câu trả lời có đúng hướng câu hỏi không? (≠ chỉ cần đúng facts)
- **Context Precision**: Trong k chunks retrieve, bao nhiêu thực sự hữu ích?
- **Context Recall**: Pipeline có bỏ sót chunk quan trọng nào không?

Quan trọng nhất: **4 metrics độc lập**. Pipeline có thể recall=1.0 (tìm đủ) nhưng relevancy=0.7 (LLM trả lời dài dòng).

### Điều bất ngờ nhất: Baseline naive đã rất mạnh (faithfulness=0.9750)

Tôi kỳ vọng production pipeline (hierarchical chunking + hybrid search + rerank + enrichment) sẽ cải thiện đáng kể so với baseline. Thực tế: baseline đã gần như perfect trên dataset này. Lý do: tài liệu HR/IT/Finance ngắn gọn, có cấu trúc rõ ràng — paragraph chunks đơn giản đã đủ.

Bài học: complexity không tự động = quality. Advanced pipeline chỉ thực sự bứt phá khi corpus lớn, unstructured, hoặc cần cross-document reasoning.

### Kết nối với bài giảng

- Slide **RAGAS framework**: hiểu sâu hơn sau khi tự implement từng metric
- Slide **RAG failure modes** ("wrong chunk", "answer drift", "missed context"): thực tế thấy "answer drift" ở answer_relevancy thấp — LLM thêm info không được hỏi
- Slide **latency vs quality tradeoff**: reranker (4.7s/query) là bottleneck rõ ràng — cần cân nhắc kỹ trước khi deploy production

---

## 4. Khó khăn & Cách giải quyết

### Khó khăn 1: RAGAS eval khi không có OpenAI key

**Vấn đề:** Lúc phát triển M4, chưa có API key để test RAGAS gốc. `ragas==0.1.21` yêu cầu OpenAI để tính semantic faithfulness.

**Giải pháp:** Implement `_heuristic_scores()` với Jaccard fallback — có thể chạy offline hoàn toàn, interface giống hệt. Khi có key, tự động dùng real RAGAS; khi không có, fallback trong suốt.

**Kết quả:** 4/4 tests pass ở cả hai mode. Khi chạy với real key, RAGAS scores thực tế khác đáng kể so với Jaccard (baseline faithfulness: Jaccard=0.44 vs RAGAS=0.975 — minh chứng Jaccard là heuristic yếu cho generative RAG).

---

### Khó khăn 2: `main.py` unpack 2 giá trị từ `build_pipeline()` trả 3

**Vấn đề:** Tôi thêm `timings` dict vào return của `build_pipeline()` → `ValueError: too many values to unpack` khi chạy `main.py`.

**Giải pháp:** Fix `main.py`: `search, reranker, _ = build_pipeline()`.

**Bài học:** Thay đổi function signature trong shared code cần kiểm tra tất cả call sites, kể cả `main.py` nằm ngoài `src/`.

---

### Khó khăn 3: BM25 trả 0 kết quả — tokenization mismatch (cross-module)

**Vấn đề:** Khi chạy integration test, BM25 tìm được 0 kết quả cho query "nghỉ phép năm". Root cause: `underthesea` tokenize "nghỉ phép" trong câu văn dài → `"nghỉ_phép"` (1 compound token), nhưng query ngắn "nghỉ phép" → `["nghỉ", "phép"]` (2 tokens). Không overlap → BM25 score = 0.

**Giải pháp** (phối hợp với Lê Hà An — M2): token expansion trong `_bm25_tokenize()`:
```python
# "nghỉ_phép" → ["nghỉ_phép", "nghỉ", "phép"]
for tok in segmented.split():
    tokens.append(tok)
    if "_" in tok:
        tokens.extend(tok.split("_"))
```
Áp dụng ở cả index và search time → BM25 hoạt động đúng.

**Thời gian debug:** ~45 phút trace từ `evaluate_pipeline()` → `run_query()` → `HybridSearch.search()` → `BM25Search.search()` → `get_scores()` trả all-zeros.

---

## 5. Nếu làm lại

### Sẽ làm khác điều gì

**1. Tune LLM prompt ngay từ đầu để target answer_relevancy:**
Thêm constraint "Trả lời bằng đúng 1 câu ngắn nhất có thể, chỉ trả lời điều được hỏi, không thêm thông tin khác" → kỳ vọng answer_relevancy tăng từ 0.7464 lên ≥ 0.80.

**2. Transparent enrichment trong M5:**
Thay vì modify chunk text với prefix "[Nội dung]", lưu enrichment vào metadata field riêng. Pass enriched text cho LLM, nhưng dùng original text để RAGAS đánh giá faithfulness — tránh prefix noise.

**3. Integration test sớm hơn:**
Test toàn bộ pipeline từ ngày 1 thay vì test từng module riêng. Interface mismatch (`build_pipeline()` return count) chỉ phát hiện khi chạy `main.py` cuối cùng.

### Module muốn thử tiếp

- **Extractive QA layer**: với câu hỏi dạng số (bao nhiêu, ngày nào) — regex extract trực tiếp từ reranked context, faithfulness = 1.0 guaranteed
- **Streaming generation**: stream gpt-4o-mini response để UX không block 1.2s/query
- **M5 với Anthropic contextual retrieval**: thay OpenAI enrichment bằng Claude Haiku để so sánh

---

## 6. Tự đánh giá

| Tiêu chí | Tự chấm (1–5) | Ghi chú |
|----------|--------------|---------|
| Hiểu bài giảng | 5 | Hiểu RAGAS metrics, failure modes, latency tradeoff — có thể giải thích từng metric |
| Code quality | 4 | Pipeline clean, fallback đầy đủ, latency tracking — có thể add streaming |
| Teamwork | 4 | Coordinate pipeline integration với 3 modules, identify cross-module BM25 bug |
| Problem solving | 5 | Giải quyết RAGAS fallback, interface mismatch, BM25 tokenization trong 1 session |

---

## 7. Kết luận

Pipeline chạy thành công end-to-end với real LLM. **3/4 metrics đạt ≥ 0.75**: faithfulness=0.8750 ✓, context_precision=0.9750 ✓, context_recall=1.0000 ✓. Answer Relevancy = 0.7464 (thiếu 0.004) — có thể fix bằng 1 dòng prompt.

Điểm bất ngờ lớn nhất: baseline naive đã rất mạnh (0.9750 faithfulness) trên dataset có cấu trúc này. Production pipeline chứng minh giá trị qua khả năng scale (hierarchical chunks, hybrid search) và latency transparency — không phải chỉ qua score tuyệt đối.
