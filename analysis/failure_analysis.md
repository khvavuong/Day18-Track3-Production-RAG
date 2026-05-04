# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Lưu Lương Vi Nhân · Khuất Văn Vương · Huỳnh Văn Nghĩa · Lê Hà An
**Thành viên:** Khuất Văn Vương → M1+M5 · Lê Hà An → M2 · Huỳnh Văn Nghĩa → M3 · Lưu Lương Vi Nhân → M4+Pipeline

---

## RAGAS Scores (gpt-4o-mini, 20 queries)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9750 | 0.8750 | −0.1000 |
| Answer Relevancy | 0.7922 | 0.7464 | −0.0458 |
| Context Precision | 0.9833 | 0.9750 | −0.0083 |
| Context Recall | 1.0000 | 1.0000 | 0.0000 |

---

## Bottom-5 Failures

### #1 — avg score: 0.5000
- **Question:** Nhân viên được nghỉ phép không lương tối đa bao nhiêu ngày mỗi năm?
- **Expected:** Tối đa 30 ngày mỗi năm và phải được Giám đốc bộ phận phê duyệt.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Scores:** faithfulness=0.00 · answer_relevancy=0.00 · context_precision=1.00 · context_recall=1.00
- **Worst metric:** Faithfulness (0.00) và Answer Relevancy (0.00)

**Error Tree:**
```
Output sai ("Không tìm thấy")
  └─ Context đúng? → CÓ (context_precision=1.0, context_recall=1.0)
       └─ LLM nhìn thấy context? → KHÔNG — LLM trả "not found" dù context có đáp án
            └─ Chunk có bị enrichment truncate không?
                 └─ CÓ — chunk "nghỉ phép không lương" nằm cuối child chunk,
                    bị cắt bởi HIERARCHICAL_CHILD_SIZE=256 chars
                    → LLM không thấy "30 ngày" trong context window
```

- **Root cause:** Hierarchical child chunk (256 chars) cắt đúng tại điểm có "30 ngày". LLM nhận được context không đủ → fallback về "không tìm thấy". Context recall = 1.0 vì RAGAS đo theo ground truth overlap, nhưng phần quan trọng bị truncate trong answer generation.
- **Suggested fix:** Tăng `HIERARCHICAL_CHILD_SIZE` từ 256 → 512 chars, hoặc retrieve parent chunk (2048 chars) thay vì child chunk khi rerank score > threshold.

---

### #2 — avg score: 0.7500
- **Question:** Sự cố an ninh phải được báo cáo trong bao lâu?
- **Expected:** Trong vòng 1 giờ kể từ khi phát hiện, qua hotline IT hoặc email security@vinuni.edu.vn.
- **Got:** Sự cố an ninh phải được báo cáo trong vòng 1 giờ kể từ khi phát hiện.
- **Scores:** faithfulness=1.00 · answer_relevancy=0.00 · context_precision=1.00 · context_recall=1.00
- **Worst metric:** Answer Relevancy (0.00)

**Error Tree:**
```
Output đúng về nội dung, nhưng answer_relevancy=0
  └─ Context đúng? → CÓ
       └─ Facts đúng? → CÓ (1 giờ — đúng)
            └─ RAGAS answer relevancy đo gì?
                 └─ Reverse question generation: sinh question từ answer,
                    so với original question bằng embedding similarity
                    → Answer ngắn ("1 giờ kể từ khi phát hiện") →
                      câu hỏi ngược sinh ra: "Khi nào cần báo cáo?"
                      vs original: "Phải báo cáo trong bao lâu?"
                    → Semantic similarity thấp do khác cấu trúc tiếng Việt
```

- **Root cause:** RAGAS answer_relevancy dùng reverse question generation + embedding similarity. Với tiếng Việt, câu hỏi "trong bao lâu" vs "khi nào" có embedding vector khác biệt dù ý nghĩa gần nhau → score = 0. Đây là limitation của RAGAS với Vietnamese NLP, không phải lỗi pipeline.
- **Suggested fix:** Dùng multilingual embedding (bge-m3) thay vì default text-embedding-ada-002 trong RAGAS để đo semantic similarity tiếng Việt chính xác hơn.

---

### #3 — avg score: 0.7900
- **Question:** Định mức tạm ứng cho công tác trong nước tối đa là bao nhiêu?
- **Expected:** Tối đa 20 triệu đồng cho mỗi chuyến công tác trong nước.
- **Got:** Định mức tạm ứng cho công tác trong nước tối đa là 20 triệu đồng cho mỗi chuyến công tác.
- **Scores:** faithfulness=0.50 · answer_relevancy=0.83 · context_precision=0.83 · context_recall=1.00
- **Worst metric:** Faithfulness (0.50)

**Error Tree:**
```
Output gần đúng nhưng faithfulness=0.5
  └─ Context đúng? → CÓ (context_recall=1.0)
       └─ LLM có bịa không? → Không bịa, nhưng paraphrase
            └─ RAGAS faithfulness decompose thành statements:
                 Statement 1: "tối đa 20 triệu đồng" → có trong context ✓
                 Statement 2: "cho mỗi chuyến công tác" → có trong context ✓
                 Statement 3: "trong nước" → context ghi "công tác trong nước tối đa 20 triệu"
                              nhưng enrichment prefix làm span detection lệch ✗
                 → 2/4 statements grounded = 0.5
```

- **Root cause:** M5 enrichment thêm prefix "[Nội dung]" vào chunk text. RAGAS faithfulness dùng NLI model để kiểm tra từng statement trong answer có được support bởi context không. Prefix nhiễu làm NLI model xác định sai boundary → 2 statements không được mark là grounded.
- **Suggested fix:** Transparent enrichment — lưu enrichment vào metadata riêng, không modify chunk text. Pass enriched text cho LLM khi generate, nhưng dùng original text khi RAGAS đánh giá faithfulness.

---

### #4 — avg score: 0.8379
- **Question:** Công ty tài trợ tối đa bao nhiêu cho đào tạo bên ngoài mỗi năm?
- **Expected:** Tối đa 20 triệu đồng/năm, nhân viên cam kết làm việc tối thiểu 12 tháng sau khóa học.
- **Got:** Công ty tài trợ tối đa 20 triệu đồng/năm cho khóa đào tạo bên ngoài liên quan đến công việc.
- **Scores:** faithfulness=0.50 · answer_relevancy=0.85 · context_precision=1.00 · context_recall=1.00
- **Worst metric:** Faithfulness (0.50)

**Error Tree:**
```
Answer đúng 1 phần, thiếu thông tin cam kết
  └─ Context đúng? → CÓ
       └─ LLM có bao gồm điều kiện cam kết 12 tháng không? → KHÔNG
            └─ System prompt: "Trả lời ngắn gọn, đúng trọng tâm (1-3 câu)"
                 └─ LLM chọn statement chính "20 triệu/năm" → đúng
                    Bỏ qua phần điều kiện "cam kết 12 tháng" → RAGAS
                    đánh giá faithfulness dựa trên ground truth overlap:
                    50% statements grounded = 0.5
```

- **Root cause:** Ground truth có 2 statements ("20 triệu/năm" và "cam kết 12 tháng"). LLM tuân theo instruction ngắn gọn, chỉ trả 1 statement. RAGAS faithfulness scoring phụ thuộc vào câu trả lời có cover đủ ground truth statements không → 1/2 = 0.5.
- **Suggested fix:** Điều chỉnh prompt để không bỏ sót thông tin điều kiện quan trọng. Hoặc chấp nhận đây là tradeoff giữa conciseness và completeness — câu trả lời "20 triệu/năm" là đúng với câu hỏi được hỏi.

---

### #5 — avg score: 0.8417
- **Question:** Hệ thống VPN của công ty dùng giao thức gì và mã hóa nào?
- **Expected:** Giao thức WireGuard với mã hóa AES-256-GCM và xác thực Curve25519.
- **Got:** Hệ thống VPN của công ty sử dụng giao thức Wi và mã hóa AES-256-GCM.
- **Scores:** faithfulness=0.50 · answer_relevancy=0.87 · context_precision=1.00 · context_recall=1.00
- **Worst metric:** Faithfulness (0.50)

**Error Tree:**
```
"WireGuard" bị truncate thành "Wi" trong answer
  └─ Context đúng? → CÓ (có "WireGuard", "AES-256-GCM", "Curve25519")
       └─ LLM truncate tên giao thức? → CÓ
            └─ Nguyên nhân: token "WireGuard" trong enriched chunk bị
               split boundary tại max_tokens=300 → LLM nhận incomplete token
               → Output "Wi" thay vì "WireGuard"
               → "Wi" không có trong context → RAGAS mark statement là unfaithful
               → Thiếu "Curve25519" → thêm 1 statement miss
               → faithfulness = 1/2 statements grounded = 0.5
```

- **Root cause:** `max_tokens=300` trong `generate_answer()` có thể không đủ cho câu trả lời technical cần quote tên protocol/cipher đầy đủ. Khi LLM bị force stop mid-token, output bị truncate.
- **Suggested fix:** Tăng `max_tokens` từ 300 → 500 cho câu hỏi technical. Hoặc thêm validation: nếu answer chứa tên protocol bị cắt (không phải keyword hoàn chỉnh), retry generation.

---

## Case Study — Phân tích sâu cho Presentation

**Question chọn phân tích:** "Nhân viên được nghỉ phép không lương tối đa bao nhiêu ngày mỗi năm?" (#1)

**Error Tree walkthrough (trình bày 5 phút):**

```
1. Output đúng?
   → KHÔNG — LLM trả "Không tìm thấy thông tin trong tài liệu."
      (câu trả lời đúng phải là "30 ngày")

2. Context được retrieve đúng không?
   → CÓ — context_precision=1.0, context_recall=1.0
   (RAGAS xác nhận đúng document được truy xuất)

3. LLM nhìn thấy "30 ngày" trong context không?
   → KHÔNG — chunk "nghỉ phép không lương" bị HIERARCHICAL_CHILD_SIZE=256
   cắt tại "Nhân viên có thể xin nghỉ phép không lương tối đa"
   → phần "30 ngày mỗi năm" rơi sang child chunk tiếp theo
   → reranker đưa chunk đầu lên top-1, LLM không thấy con số

4. Fix ở bước nào?
   → M1 (chunking): tăng child size từ 256 → 512
   → Hoặc M2 (retrieval): khi context_recall = 1.0 nhưng answer = "not found",
     implement parent-document retrieval (trả về parent chunk thay vì child)
```

**Kết luận:** Pipeline retrieve đúng document nhưng chunk quá nhỏ làm mất thông tin số liệu. Đây là classic "small chunk = high precision, low completeness" tradeoff của hierarchical chunking.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Tăng child size lên 512 chars** — fix trực tiếp case #1 và tương tự
2. **Tune LLM prompt** — thêm "Trả lời 1 câu ngắn, chỉ điều được hỏi" → fix answer_relevancy từ 0.7464 lên ≥ 0.80
3. **Transparent enrichment** — không add prefix vào chunk text → fix faithfulness từ 0.8750 lên ≥ 0.92
