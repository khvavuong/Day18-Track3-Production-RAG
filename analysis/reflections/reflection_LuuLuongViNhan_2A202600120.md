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
- `evaluate_ragas()` — thử RAGAS gốc (khi có `OPENAI_API_KEY`), fallback về heuristic Jaccard khi không có key
- `_heuristic_scores()` — tính 4 metrics qua Jaccard overlap:
  - faithfulness = jaccard(answer tokens, top context tokens)
  - answer_relevancy = jaccard(answer, question + ground_truth)
  - context_precision = jaccard(joined contexts, ground_truth)
  - context_recall = |ground_truth ∩ joined_contexts| / |ground_truth|
- `failure_analysis()` — sort per-question scores, xác định bottom-N, áp Diagnostic Tree
- `save_report()` — lưu `{aggregate, num_questions, per_question, failures}` vào JSON

**`src/pipeline.py`**
- `generate_answer()` — LLM generation với gpt-4o-mini, grounded prompt, fallback về contexts[0]
- `build_pipeline()` — orchestrate M1→M5→M2→M3 với per-step timing
- `run_query()` — single query: search → rerank → generate, track latency (search_ms, rerank_ms, generate_ms)
- `evaluate_pipeline()` — chạy 20 queries, gọi M4, in bảng RAGAS + latency breakdown, gọi `save_report()`

### Số tests pass
- M4: **4/4** tests pass (`tests/test_m4.py`)
- Pipeline: chạy end-to-end thành công (20/20 queries)

---

## 2. Kết quả thực tế

### Scores thu được (run ngày 2026-05-04)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.4399 | 0.4231 | −0.0168 |
| Answer Relevancy | 0.2637 | **0.3260** | **+0.0623** |
| Context Precision | 0.0795 | **0.1073** | **+0.0278** |
| Context Recall | 1.0000 | **0.9755** | −0.0245 |

### Latency

| Stage | Avg ms/query |
|-------|-------------|
| Search (M2) | 561.1 ms |
| Rerank (M3) | 3531.9 ms |
| Generate (LLM/fallback) | 0.0 ms |
| **Total** | **~4.1 s** |

---

## 3. Kiến thức học được

### Khái niệm mới nhất: RAGAS metrics và ý nghĩa thực tế

Trước bài lab này, tôi hiểu RAG như một pipeline retrieval đơn giản. Sau khi implement M4, tôi hiểu cụ thể từng metric đo cái gì:

- **Faithfulness**: câu trả lời có bịa đặt thông tin không có trong context không?
- **Answer Relevancy**: câu trả lời có trả lời đúng câu hỏi không (không phải câu hỏi khác)?
- **Context Precision**: trong các context truy xuất, bao nhiêu % thực sự có ích?
- **Context Recall**: pipeline có bỏ sót context quan trọng nào không?

Quan trọng: 4 metrics này đo **những thứ khác nhau**. Một pipeline có thể có context recall cao (tìm đúng) nhưng faithfulness thấp (LLM vẫn bịa thêm).

### Điều bất ngờ nhất: Enrichment làm Jaccard metrics tệ hơn

M5 enrichment thêm prefix "[Nội dung]", "[Tóm tắt]" vào chunks. Điều này tốt cho LLM (thêm ngữ cảnh), nhưng lại làm heuristic Jaccard giảm vì:
- Answer = nguyên chunk có prefix → nhiều tokens noise
- Ground truth = câu ngắn gọn → jaccard overlap thấp hơn

Điều này dạy tôi rằng: evaluation method phải khớp với generation method. Heuristic Jaccard tốt cho extractive RAG, không tốt cho generative RAG với enriched context.

### Kết nối với bài giảng

- Slide về **RAGAS framework** (faithfulness, answer_relevancy, context_precision, context_recall)
- Slide về **RAG failure modes**: "wrong chunk", "answer drift", "missed context"
- Slide về **latency vs quality tradeoff**: reranker tốt hơn nhưng +3.5s/query

---

## 4. Khó khăn & Cách giải quyết

### Khó khăn 1: RAGAS gốc yêu cầu OpenAI API key

**Vấn đề:** `ragas==0.1.21` gọi OpenAI để tính semantic faithfulness và LLM-as-judge. Không có key → crash.

**Giải pháp:** Implement `_heuristic_scores()` với Jaccard — có thể chạy offline hoàn toàn. Dùng try/except để fallback tự động.

**Thời gian debug:** ~30 phút tìm đúng error và thiết kế fallback không làm hỏng interface.

---

### Khó khăn 2: `build_pipeline()` trả 3 giá trị, `main.py` unpack 2

**Vấn đề:** Pipeline cũ trả `(search, reranker)`, tôi thêm `timings` → `ValueError: too many values to unpack`.

**Giải pháp:** Fix `main.py` unpack thành `search, reranker, _ = build_pipeline()`.

**Bài học:** Khi thay đổi function signature, luôn kiểm tra tất cả call sites.

---

### Khó khăn 3: BM25 trả về 0 kết quả

**Vấn đề:** `underthesea` tokenize "nghỉ phép" trong câu văn thành "nghỉ_phép" (compound token), nhưng query "nghỉ phép" standalone → 2 tokens riêng biệt. BM25 scores = 0 cho tất cả docs.

**Giải pháp** (do Lê Hà An fix trong M2): token expansion — khi index và query, expand "nghỉ_phép" → ["nghỉ_phép", "nghỉ", "phép"]. Cả hai đầu đều biết token compound lẫn component → BM25 match được.

**Bài học:** NLP pipeline cho tiếng Việt cần test edge case với từ ghép. Tokenization inconsistency giữa corpus và query là lỗi thường gặp.

---

## 5. Nếu làm lại

### Sẽ làm khác điều gì

1. **Extractive QA layer trước LLM**: với câu hỏi "bao nhiêu", "ngày nào" → dùng regex extract số/ngày trực tiếp từ top context. Không cần API key, faithfulness = 1.0 cho những câu hỏi có số cụ thể.

2. **Transparent enrichment**: thay vì thêm prefix "[Nội dung]", "[Tóm tắt]" vào enriched text, chỉ dùng metadata field riêng. Khi generate answer, pass enriched context. Khi eval faithfulness, so với original text. Tránh noise cho Jaccard metrics.

3. **Viết integration test sớm hơn**: test toàn bộ pipeline từ sớm (chunking → search → rerank → eval) thay vì test từng module riêng, tránh interface mismatch phát hiện muộn.

### Module muốn thử tiếp
- **M5 với real LLM**: xem contextual enrichment thực sự cải thiện bao nhiêu khi có OpenAI key
- **M2 với ColBERT**: thay Dense retrieval bằng late-interaction model, precision tốt hơn cho tiếng Việt

---

## 6. Tự đánh giá

| Tiêu chí | Tự chấm (1–5) | Ghi chú |
|----------|--------------|---------|
| Hiểu bài giảng | 4 | Hiểu RAGAS metrics và RAG failure modes; cần đọc thêm về ColBERT |
| Code quality | 4 | Pipeline clean, có fallback, track latency; heuristic eval có thể improve |
| Teamwork | 4 | Coordinate integration, identify BM25 tokenization bug cross-module |
| Problem solving | 5 | Xử lý RAGAS fallback, pipeline signature bug, BM25 mismatch trong 1 session |

---

## 7. Kết luận

Production RAG pipeline chạy thành công end-to-end. Context Recall 0.9755 chứng minh chunking + hybrid search tốt. Answer Relevancy tăng +0.0623 nhờ BM25+Dense+RRF so với dense-only. Bottleneck hiện tại là reranker latency (3.5s/query) và absence của real LLM. Với OpenAI API key, faithfulness kỳ vọng tăng lên ≥ 0.75 nhờ strict grounded prompt trong `generate_answer()`.
