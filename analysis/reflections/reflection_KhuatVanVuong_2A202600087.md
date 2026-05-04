# Individual Reflection — Lab 18

**Tên:** Khuat Van Vuong  
**Module phụ trách:** M1 (Advanced Chunking) + M5 (Enrichment Pipeline)

---

## 1. Đóng góp kỹ thuật

- Module đã implement:
  - `src/m1_chunking.py`: hoàn thiện `chunk_semantic`, `chunk_hierarchical`, `chunk_structure_aware`, `compare_strategies`.
  - `src/m5_enrichment.py`: hoàn thiện `summarize_chunk`, `generate_hypothesis_questions`, `contextual_prepend`, `extract_metadata`, `enrich_chunks`.
- Các hàm/class chính đã viết:
  - M1: semantic grouping theo cosine similarity, parent-child chunking có `parent_id`, structure-aware chunking theo markdown headers.
  - M5: enrichment pipeline với output chuẩn để teammate index trực tiếp: `enriched_text` + `auto_metadata`.
- Số tests pass:
  - M1: `13/13` (`pytest tests/test_m1.py`).
  - M5: đã xử lý lỗi dependency (`openai/httpx`) để pipeline có thể chạy ổn định.

## 2. Kiến thức học được

- Khái niệm mới nhất:
  - Chất lượng retrieval phụ thuộc mạnh vào chunking strategy, không chỉ phụ thuộc model embedding.
  - Enrichment trước embedding (contextual prepend + HyQA + metadata) giúp bridge vocabulary gap và tăng recall.
- Điều bất ngờ nhất:
  - Khi chuẩn hóa output chunk cho index (`enriched_text`), teammate có thể tích hợp search nhanh hơn và ít phải sửa pipeline.
- Kết nối với bài giảng (slide nào):
  - Bám theo pipeline production RAG đã học: chunking → retrieval → reranking → generation → evaluation (RAGAS + failure analysis).

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất:
  - Lỗi môi trường khi tạo OpenAI client: `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`.
- Cách giải quyết:
  - Bổ sung xử lý lỗi an toàn trong M5 (không để crash toàn test/pipeline).
  - Pin dependency tương thích trong `requirements.txt` (`httpx==0.27.2`) để đồng bộ môi trường nhóm.
- Thời gian debug:
  - Khoảng 5 phút.

## 4. Nếu làm lại

- Sẽ làm khác điều gì:
  - Chốt dependency matrix và chạy smoke test môi trường ngay từ đầu thay vì đợi tới cuối.
- Module nào muốn thử tiếp:
  - M2 (Hybrid Search) để tối ưu sâu hơn BM25 + Dense + RRF dựa trên metadata đã enrich.

## 5. Tự đánh giá

| Tiêu chí        | Tự chấm (1-5) |
| --------------- | ------------- |
| Hiểu bài giảng  | 5             |
| Code quality    | 4             |
| Teamwork        | 5             |
| Problem solving | 5             |
