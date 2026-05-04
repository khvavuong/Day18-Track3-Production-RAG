# Group Report — Lab 18: Production RAG

**Nhóm:** Group (Lưu Lương Vi Nhân, Khuất Văn Vương, Huỳnh Văn Nghĩa, Lê Hà An)
**Ngày:** 2026-05-04

---

## Thành viên & Phân công

| Tên | MSSV | Module | File | Tests pass |
|-----|------|--------|------|-----------|
| Khuất Văn Vương | 2A202600087 | M1: Hierarchical Chunking | `src/m1_chunking.py` | 13/13 |
| Lê Hà An | 2A202600376 | M2: Hybrid Search (BM25+Dense+RRF) | `src/m2_search.py` | 5/5 |
| Huỳnh Văn Nghĩa | 2A202600085 | M3: Cross-Encoder Reranking | `src/m3_rerank.py` | 5/5 |
| Lưu Lương Vi Nhân | 2A202600120 | M4: RAGAS Evaluation + Pipeline | `src/m4_eval.py`, `src/pipeline.py` | 4/4 |
| Khuất Văn Vương | 2A202600087 | M5: Chunk Enrichment (bonus) | `src/m5_enrichment.py` | 9/9 |

---

## Kết quả RAGAS

Đánh giá trên 20 câu hỏi (test_set.json), chạy với **gpt-4o-mini**, RAGAS v0.1.21.

| Metric | Naive Baseline | Production Pipeline | Δ | Đạt ≥ 0.75? |
|--------|---------------|---------------------|---|-------------|
| Faithfulness | 0.9750 | **0.8750** | −0.1000 | ✓ |
| Answer Relevancy | 0.7922 | **0.7464** | −0.0458 | ✗ (tiệm cận) |
| Context Precision | 0.9833 | **0.9750** | −0.0083 | ✓ |
| Context Recall | 1.0000 | **1.0000** | 0.0000 | ✓ |

**3/4 metrics đạt ≥ 0.75.** Answer Relevancy = 0.7464 (chỉ thiếu 0.004).

### Latency Breakdown (avg / query)

| Stage | Avg ms |
|-------|--------|
| Search (BM25 + Dense + RRF) | 477.8 ms |
| Rerank (CrossEncoder bge-reranker-v2-m3) | 4686.1 ms |
| Generate (gpt-4o-mini) | 1263.2 ms |
| **Total** | **~6.4 s/query** |

---

## Phân tích kết quả

### Điểm tốt

- **Context Recall = 1.0000**: Pipeline truy xuất đầy đủ 100% context cần thiết. Hierarchical chunking (M1) + hybrid search (M2) không bỏ sót bất kỳ thông tin nào trong 20 queries.
- **Context Precision = 0.9750**: 97.5% context được retrieve là thực sự liên quan — chứng minh reranking (M3) loại bỏ hiệu quả noise chunks.
- **Faithfulness = 0.8750 ✓**: gpt-4o-mini với prompt grounded ("PHẢI trích dẫn con số chính xác") giữ câu trả lời sát context, không bịa đặt.

### Điểm cần cải thiện

- **Answer Relevancy = 0.7464 (tiệm cận 0.75)**: Một số câu trả lời chính xác nhưng verbose — trả lời cả đoạn context thay vì chỉ extract phần trả lời câu hỏi. Cải thiện: thêm instruction "Trả lời cực ngắn, chỉ 1 câu" vào system prompt.
- **Faithfulness giảm so với baseline (−0.10)**: Baseline dùng naive paragraph chunks (sạch), còn production dùng enriched chunks với prefix "[Nội dung]". Khi gpt-4o-mini generate, đôi khi paraphrase enrichment prefix thay vì quote nguyên văn số liệu.
- **Rerank latency 4.7s/query**: CrossEncoder bge-reranker-v2-m3 chạy trên CPU. Với GPU hoặc model nhỏ hơn (ms-marco-MiniLM-L-6-v2) có thể giảm xuống <500ms.

---

## Key Findings

1. **Biggest insight — Baseline đã rất tốt (faithfulness=0.975):** Với dữ liệu gọn, có cấu trúc (HR/IT/Finance docs), naive paragraph chunking + dense-only đã retrieve đúng context. Production pipeline không cải thiện thêm vì không gian cải thiện gần như không còn (context recall=1.0 ở cả hai).

2. **Biggest challenge — BM25 tokenization mismatch với tiếng Việt:**
   `underthesea` tokenize "nghỉ phép" trong câu văn corpus thành `nghỉ_phép` (compound), nhưng query standalone "nghỉ phép" thành 2 token riêng → BM25 score = 0. Fix: token expansion `"nghỉ_phép" → ["nghỉ_phép", "nghỉ", "phép"]` ở cả index và search time. Không fix thì BM25 hoàn toàn vô dụng.

3. **Surprise finding — Enrichment có thể là double-edged sword:**
   M5 enrichment thêm "[Nội dung]", "[Tóm tắt]" prefix vào chunks. LLM đôi khi echo prefix này vào câu trả lời → RAGAS đánh giá câu trả lời ít "relevant" hơn. Transparent enrichment (dùng metadata field, không modify text) sẽ tốt hơn.

---

## Failure Cases (Top 5 worst — Answer Relevancy thấp nhất)

| # | Câu hỏi | Vấn đề | Answer Relevancy |
|---|---------|--------|-----------------|
| 1 | Thưởng tháng 13 được trả khi nào? | Answer verbose, trả lời thêm điều kiện không được hỏi | thấp |
| 2 | Hệ thống VPN dùng giao thức gì? | LLM liệt kê cả AES-256-GCM lẫn Curve25519 dài hơn cần | thấp |
| 3 | Phiên VPN tự động ngắt sau bao lâu? | Trả lời đúng nhưng thêm context về MDM không liên quan | thấp |
| 4 | Mật khẩu thay đổi định kỳ thế nào? | Answer hơi dài, nhắc lại cả quy tắc mật khẩu | thấp |
| 5 | Công ty tài trợ đào tạo bao nhiêu? | Trả lời đúng 20 triệu nhưng thêm điều kiện cam kết 12 tháng | thấp |

**Root cause chung**: System prompt cho phép LLM thêm thông tin liên quan — cần strict hơn với "1 câu duy nhất, chỉ trả lời điều được hỏi".

---

## Presentation Notes (5 phút)

### Slide 1 — RAGAS Scores (1 phút)
- **3/4 metrics đạt ≥ 0.75**: faithfulness ✓, context precision ✓, context recall ✓
- Answer Relevancy = 0.7464 — tiệm cận, cần tune prompt
- Context Recall = 1.0 (perfect): không bỏ sót câu trả lời nào

### Slide 2 — Biggest Win: BM25 Tokenization Fix (1 phút)
- Vấn đề: `underthesea` compound token mismatch giữa corpus và query
- Fix: token expansion trong `_bm25_tokenize()` — 5 dòng code, impact lớn
- Nếu không fix: BM25 luôn score=0, hybrid search = dense-only (giống baseline)

### Slide 3 — Case Study: Answer Relevancy (2 phút)
- Query: "Phiên VPN tự động ngắt sau bao lâu?"
- Ground truth: "Sau 8 giờ không hoạt động."
- Pipeline answer: "Phiên VPN tự động ngắt sau 8 giờ không hoạt động. Ngoài ra, VPN chỉ được dùng trên thiết bị đã cài MDM."
- Vấn đề: câu 2 đúng nhưng không được hỏi → RAGAS penalize answer relevancy
- Fix: thêm "Trả lời bằng đúng 1 câu ngắn nhất có thể" vào system prompt

### Slide 4 — If We Had 1 More Hour (1 phút)
1. **Tune LLM prompt**: strict 1-câu → answer relevancy kỳ vọng tăng lên ≥ 0.80
2. **Transparent enrichment**: không modify chunk text → faithfulness tăng
3. **GPU reranker**: latency giảm từ 4.7s → <0.5s/query
