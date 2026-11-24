# Datasheet for Datasets: VeriStat Benchmark & Official Vietnamese Statistical Corpus

Tài liệu này mô tả bộ dữ liệu và benchmark **VeriStat** theo 7 mục chuẩn của *Datasheets for Datasets* (Gebru et al., 2021), được xây dựng trực tiếp từ các kết quả kiểm định định lượng trong [data_audit.json](data_audit.json), [data_quality_report.json](data_quality_report.json) và [benchmark_sanity_report.json](benchmark_sanity_report.json).

---

## 1. Motivation (Động lực)

- **Mục đích:** Cung cấp một cơ sở dữ liệu thống kê chính thức có cấu trúc (SQLite) và bộ câu hỏi kiểm chuẩn đa bước (**VeriStat Benchmark**) nhằm đánh giá năng lực suy luận định lượng, sử dụng công cụ (tool-use), và khả năng tự kiểm chứng của các tác tử LLM (ReAct, MCTS + Verifier) trên dữ liệu thống kê tiếng Việt thực tế.
- **Bài toán thực tiễn:** Dữ liệu thống kê chính thức của Việt Nam có nhiều thách thức đặc thù: bảng PDF nhiều trang, dấu thập phân kiểu Việt Nam (dấu phẩy), ký hiệu khuyết thiếu đa dạng (`..`, `-`, `x`), khác biệt đơn vị giữa các bảng (*nghìn người* vs *người*, *tỷ đồng* vs *triệu đồng*), và đặc biệt là sự thay đổi địa giới hành chính từ **63 tỉnh/thành phố** (Niên giám 2025) sang **34 tỉnh/thành phố** sau sáp nhập theo Nghị quyết 202/2025/QH15 (Biểu số liệu tháng 10/2026).
- **Nhóm xây dựng:** Nhóm đồ án **VeriStat** (Thành viên 1 — Long phụ trách số hóa dữ liệu, chuẩn hóa hành chính và sinh benchmark; Thành viên 2 — Khoa phụ trách công cụ & tác tử; Thành viên 3 — Sơn phụ trách Verifier, MCTS và khung đánh giá A.4).

---

## 2. Composition (Thành phần Dữ liệu)

### 2.1. Kho Dữ liệu SQLite (`data/processed/veristat.db`)
- **Tổng số bảng thống kê đã số hóa:** **32 bảng** (`tables_catalog`), phủ đủ **5 lĩnh vực**:
  - `dan_so` (Dân số): 6 bảng
  - `lao_dong` (Lao động): 6 bảng
  - `doanh_nghiep` (Doanh nghiệp): 8 bảng
  - `giao_duc` (Giáo dục): 6 bảng
  - `y_te` (Y tế & Mức sống dân cư): 6 bảng
- **Phân bổ theo nguồn:** 27 bảng từ Niên giám Thống kê 2025 (`pdf_2025`) và 5 bảng từ Biểu số liệu tháng 10/2026 (`excel_2026`).
- **Bảng ánh xạ hành chính (`admin_mapping`):** **93 bản ghi**, ánh xạ đầy đủ 63 tỉnh/thành phố cũ, 6 vùng kinh tế - xã hội, `Cả nước`, và các biến thể tên sang 34 tỉnh/thành phố mới sau sáp nhập năm 2025.
- **Kho tài liệu giải thích (`docs_fts`):** Chỉ mục toàn văn SQLite FTS5 chứa phần giải thích thuật ngữ thống kê chính thức từ 4 chương của Niên giám 2025 và ghi chú sáp nhập 34 tỉnh.
- **Bản xuất trung gian tái lập:** 32 tệp CSV sạch (UTF-8-SIG) tại `data/interim/tables/`.

### 2.2. Bộ Câu hỏi Benchmark (`data/benchmark/`)
- **Tập hiệu chỉnh (`dev_30.jsonl`):** **30 câu** (`growth`: 8, `share`: 8, `ranking`: 7, `correlation`: 7; `easy`: 12, `medium`: 8, `hard`: 10; `uses_admin_mapping`: 8 câu = **26.67%**).
- **Tập kiểm thử chính (`benchmark_400.jsonl`):** **400 câu** (0 trùng lặp ID và 0 trùng lặp văn bản với `dev_30.jsonl`):
  - **Theo nhóm suy luận:** `growth`: 100, `share`: 100, `ranking`: 100, `correlation`: 100.
  - **Theo độ khó:** `easy`: 120 (30.0%), `medium`: 162 (40.5%), `hard`: 118 (29.5%).
  - **Theo kiểu đáp án (`answer_type`):** `number`: 300, `entity_list`: 71, `entity`: 29.
  - **Sử dụng ánh xạ hành chính (`uses_admin_mapping`):** 108/400 câu (**27.0%**).
  - **Phân bố miền giá trị số (`benchmark_400`):**
    - `growth` (%): min `-33.332`, median `4.5445`, max `85.2589`
    - `share` (%): min `0.0553`, median `5.3061`, max `97.4001`
    - `correlation` (Pearson $r$): min `-0.8165`, median `0.6484`, max `0.9939`
- **Mẫu rà soát chéo 20% (`crosscheck_sample.csv`):** **80 câu** lấy mẫu phân tầng (20 câu/nhóm) phục vụ kiểm tra thủ công bởi 3 thành viên.

---

## 3. Collection Process (Quy trình Thu thập)

Dữ liệu được trích xuất hoàn toàn từ **2 nguồn thống kê chính thức** của Cục Thống kê / Tổng cục Thống kê Việt Nam (GSO):
1. **`NG-TOAN-QUOC-2025-1.pdf`** — *Niên giám Thống kê Toàn quốc 2025* (1.557 trang, độ lệch trang bìa `cover_offset = 1`, gồm 14 chương mục; đã quét tự động 173 bảng thuộc 4 chương chính và tuyển chọn có kiểm định 27 bảng).
2. **`2.-Bieu-thang-10.2026.xlsx`** — *Biểu số liệu Kinh tế - Xã hội tháng 10 năm 2026* (20 sheet; tuyển chọn 5 sheet trọng tâm thuộc lĩnh vực Lao động và Doanh nghiệp, đã áp dụng cơ cấu 34 tỉnh/thành phố mới).

Không thu thập dữ liệu cá nhân (PII); toàn bộ số liệu đều là thống kê tổng hợp cấp quốc gia, vùng, tỉnh hoặc ngành kinh tế đã công bố công khai.

---

## 4. Preprocessing, Cleaning, and Labeling (Tiền xử lý, Làm sạch & Gán nhãn)

- **Chuẩn hóa số học:** Chuyển đổi định dạng dấu phẩy thập phân kiểu Việt Nam (ví dụ `102345,3` $\rightarrow$ `102345.3`), loại bỏ chú thích `(*)`, chuẩn hóa các ký hiệu khuyết thiếu (`..`, `...`, `-`, `–`, `x`, `Sơ bộ`) về `NULL`.
- **Phân cấp địa lý (`geo_level`):** Gắn nhãn tường minh `country`, `region`, `province`, `other` cho từng dòng dữ liệu địa phương. Kết quả kiểm tra tự động ghi nhận **`unexpected_other_count = 0`** trên toàn bộ 32 bảng.
- **Tỷ lệ dữ liệu khuyết thiếu (Null rate):**
  - **26/32 bảng** có `null_rate_pct = 0.0%`.
  - **6 bảng** có giá trị `NULL` nguyên bản từ nguồn (không tự suy diễn/điền bù): `NG2025_T331` (0.91%), `NG2025_T332` (2.50%), `NG2025_T340` (3.33%), `NG2025_T339` (4.81%), `NG2025_T335` (6.67%), và `NG2025_T341` (21.33%).
- **Đối chiếu tổng cộng dồn (Additive Reconciliation):** Trên **36/36 phép kiểm tra** tổng 63 tỉnh so với dòng `Cả nước` ở các bảng có tính cộng dồn (`NG2025_T015`, `T017`, `T020`, `T046`, `T053`, `T136`, `T139`, `T151`), **100% (36/36) đạt chuẩn**, sai lệch làm tròn lớn nhất chỉ là **`0.0008%`**.
- **Sinh và xác minh Ground Truth:** Mỗi câu hỏi trong `dev_30.jsonl` và `benchmark_400.jsonl` đều đi kèm mã thực thi `reference_code` chạy trên `veristat.db` ở chế độ chỉ đọc (`mode=ro`). Script `scripts/verify_benchmark.py` tái thực thi độc lập **430/430 câu hỏi** đạt trạng thái **`VERIFIED 100%`**.
- **Rà soát chéo 20%:** Tệp `data/benchmark/crosscheck_sample.csv` (80 câu) được chuẩn bị sẵn các cột `reviewer_long_ok`, `reviewer_khoa_ok`, `reviewer_son_ok` để ghi nhận tỷ lệ đồng thuận thủ công của cả nhóm trước khi chốt báo cáo cuối kỳ.

---

## 5. Uses (Mục đích Sử dụng)

- **Mục đích khuyến nghị:** Đánh giá và so sánh các phương pháp suy luận có công cụ (Direct, ReAct, MCTS + Verifier) trên bài toán hỏi đáp thống kê tiếng Việt; nghiên cứu lỗi suy luận định lượng (C.4 Error Taxonomy).
- **Giới hạn & Không khuyến nghị:** Không dùng để suy luận cá nhân, dự báo kinh tế vĩ mô thời gian thực, hoặc thay thế công bố pháp lý chính thức của Cục Thống kê. Khi so sánh dữ liệu năm 2020–2025 (63 tỉnh) với năm 2026 (34 tỉnh), người dùng bắt buộc phải dùng bảng `admin_mapping`.

---

## 6. Distribution (Phân phối)

- Bộ mã nguồn ETL, bộ sinh benchmark, tệp CSV trung gian và các tập JSONL (`dev_30.jsonl`, `benchmark_400.jsonl`) được lưu trữ và quản lý phiên bản trong kho mã nguồn nội bộ của nhóm đồ án VeriStat.
- Cơ sở dữ liệu `data/processed/veristat.db` có thể được tái tạo hoàn toàn tất định bằng lệnh `python scripts/build_db.py` (từ tệp nguồn gốc hoặc từ `data/interim/tables/*.csv` với cờ `--from-csv`).

---

## 7. Maintenance (Bảo trì)

- **Người phụ trách:** Thành viên 1 (Long — VeriStat Data & Benchmark).
- **Kiểm thử hồi quy tự động:** Mọi thay đổi đối với schema, `admin_mapping`, pipeline ETL, bộ sinh benchmark hoặc bộ công cụ C.4 đều được kiểm tra bằng `pytest tests/data/`, `scripts/check_data_quality.py` và `scripts/verify_benchmark.py`.
- **Cập nhật phiên bản:** Khi có Niên giám Thống kê hoặc Biểu số liệu mới, chỉ cần bổ sung mục tương ứng vào `docs/table_selection.yaml` và chạy lại pipeline ETL + kiểm định.
