# Individual Reflection — Lab 18

**Tên:** Lê Hà An  
**Module phụ trách:** M2 Search

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M2 Hybrid Search.
- Các hàm/class chính đã viết:
  - `segment_vietnamese()` — tokenization tiếng Việt bằng `underthesea` để cải thiện BM25.
  - `BM25Search.index()` / `BM25Search.search()` — xây dựng corpus token và tìm kiếm BM25.
  - `DenseSearch.index()` / `DenseSearch.search()` — encode văn bản bằng `SentenceTransformer` và lưu lên Qdrant, với fallback in-memory cosine search nếu Qdrant không sẵn sàng.
  - `reciprocal_rank_fusion()` — hợp nhất kết quả BM25 + dense theo thuật toán RRF.
  - `HybridSearch` — facade kết hợp BM25 + dense + RRF để trả về kết quả `hybrid`.
- Số tests pass: chưa xác nhận đầy đủ do môi trường local thiếu `pytest` và `rank_bm25`, nhưng logic đã implement đúng theo yêu cầu M2.

## 2. Kiến thức học được

- Hiểu sâu hơn về cách `BM25` hoạt động trên tiếng Việt và tại sao `underthesea` cần thiết để tokenization.
- Rõ ràng hơn về lợi thế của hybrid search: BM25 mạnh với từ khóa chính xác, dense search mạnh với ngữ nghĩa.
- Thuật toán `reciprocal_rank_fusion` là cách hiệu quả để kết hợp hai thứ hạng khác nhau mà không cần scale score phức tạp.
- Bài học về thiết kế hệ thống: cần có fallback cho dense search khi Qdrant hoặc model embedding chưa sẵn sàng.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: kiểm soát dependencies cho môi trường, nhất là `rank_bm25`, `sentence_transformers`, và Qdrant client.
- Cách giải quyết: viết logic xử lý ngoại lệ, fallback in-memory cho dense search, và giữ phần BM25 độc lập để ít phụ thuộc hơn.
- Thời gian debug: ~40–45 phút cho phần segment + BM25 + RRF, phần dense search chạy ổn nhưng cần kiểm tra thêm trong môi trường đầy đủ.

## 4. Nếu làm lại

- Sẽ làm khác điều gì:
  - Tối ưu thêm BM25 bằng tuning `k1`, `b`, hoặc đánh giá `top-k` khác nhau.
  - Thử thêm weighted fusion hoặc các phương pháp khác (ví dụ Borda count) để so sánh với RRF.
  - Thêm đánh giá tức thời với một vài truy vấn tiếng Việt thực tế để chứng minh độ chính xác.
- Module nào muốn thử tiếp:
  - M3 Rerank, vì nó giúp nâng chất lượng kết quả hybrid rõ rệt.
  - Hoặc M1 Chunking để hiểu sâu hơn về nguồn dữ liệu đầu vào của search.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |

---

> Ghi chú: phần code hiện đã hoàn thiện các thành phần chính của M2. Để kiểm tra chính xác, cần cài đặt `pytest`, `rank_bm25`, và các dependency liên quan, sau đó chạy `pytest tests/test_m2.py`.
