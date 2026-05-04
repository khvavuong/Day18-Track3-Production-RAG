# Individual Reflection — Lab 18

**Tên:** Huỳnh Văn Nghĩa (MSSV: 2A202600085)  
**Module phụ trách:** M3 (Reranking)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M3: Reranking
- Các hàm/class chính đã viết:
  - `CrossEncoderReranker._load_model()`: Load CrossEncoder model (`BAAI/bge-reranker-v2-m3`).
  - `CrossEncoderReranker.rerank()`: Xử lý danh sách document, predict scores, sắp xếp và format kết quả dưới dạng object `RerankResult`.
  - `FlashrankReranker.rerank()`: Tích hợp thư viện Flashrank làm lightweight alternative reranker.
  - `benchmark_reranker()`: Hàm tính toán độ trễ (latency) trung bình, min và max để benchmarking rerankers.
- Số tests pass: 5/5

## 2. Kiến thức học được

- Khái niệm mới nhất: Hiểu rõ cơ chế Cross-encoder (chấm điểm theo từng cặp Query-Doc) mang lại độ chính xác cao hơn Bi-encoder (chỉ so sánh Vector Similarity), nhưng bù lại trade-off là latency cao hơn khi search ở real-time. Do đó Rerank chỉ nên chạy cho top-K document lấy ra từ bước Retrieve.
- Điều bất ngờ nhất: Khả năng tích hợp thư viện `flashrank` cực nhẹ, cho thời gian rerank nhanh ấn tượng (< 10ms) mà không tốn quá nhiều tài nguyên load model so với PyTorch model truyền thống.
- Kết nối với bài giảng: Ứng dụng thực tế lý thuyết Reranking, Reciprocal Rank Fusion kết nối vào Pipeline Production RAG để tối ưu hóa context retrieval precision.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Lần đầu tiên load weights cho pipeline cross-encoder của sentence-transformers bị bottle-neck thời gian đo đạc cho hàm benchmark khởi tạo ban đầu, khiến metrics latency bị nhiễu do tốn thời gian download.
- Cách giải quyết: Tiến hành tạo warmup stage gọi hàm load model trước vòng lặp test time (`time.perf_counter()`).
- Thời gian debug: Khoảng 20 - 30 phút.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Dùng local caching cho embedding thay vì load lại model. Bổ sung thêm strategy rerank Cohere nếu chi phí token API cho phép.
- Module nào muốn thử tiếp: M5 (Enrichment)

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 5 |
