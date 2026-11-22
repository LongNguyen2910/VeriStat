# Thiết kế Hệ thống VeriStat — Đề tài 18: Tác tử Phân tích Số liệu Thống kê Kinh tế – Xã hội

**Ngày:** 2026-11-22  
**Quy mô nhóm:** 3 thành viên | **Thời hạn:** 2 tuần | **Thứ tự thực hiện:** Phần A & D trước → Phần B, C & Báo cáo  
**Tài nguyên tính toán:** 1 tài khoản Google Colab Pro (GPU L4 24GB / A100 40GB) + máy cá nhân (CPU)  
**Nguồn dữ liệu chính:** Tự số hóa trực tiếp từ **Niên giám Thống kê Toàn quốc 2025** (`NG-TOAN-QUOC-2025-1.pdf`, ~1.558 trang, chữ vector) và **Báo cáo tình hình kinh tế – xã hội tháng 10/2026** (`2.-Bieu-thang-10.2026.xlsx`, 20 sheet) của cơ quan thống kê Việt Nam.

---

## 1. Mục tiêu và Bối cảnh (Intent & Why)

Nhóm 3 sinh viên cần hoàn thành trọn vẹn **Đề tài 18 (Khối tích hợp)** của môn Học máy cơ bản trong 2 tuần trong điều kiện vừa làm vừa tìm hiểu. Trọng tâm kỹ thuật là xây dựng một tác tử mô hình ngôn ngữ lớn (LLM Agent) có khả năng **sử dụng công cụ, lập kế hoạch và tìm kiếm** để phân tích số liệu thống kê kinh tế – xã hội Việt Nam với **đáp số kiểm chứng được**, kết nối chặt chẽ giữa cài đặt thực nghiệm (Phần A & D) với nền tảng lý thuyết POMDP, bỏ phiếu đa số, bộ kiểm chứng và MCTS/UCT (Phần B & C).

### Mục tiêu (Goals)
- Đáp ứng đầy đủ 100% các mục chấm điểm của Đề tài 18:
  - **Phần A (20đ):** Cài đặt từ đầu bằng Python thuần + NumPy (**không dùng framework agent dựng sẵn** như LangChain/LangGraph/LlamaIndex) cho vòng lặp ReAct (3 công cụ + giới hạn bước/thời gian), Self-Consistency (có biến thể dừng sớm), Best-of-*n* với Verifier, MCTS với quy tắc UCT, và khung đánh giá tác tử.
  - **Phần D (30đ):** Tự số hóa trực tiếp các bảng số liệu từ `NG-TOAN-QUOC-2025-1.pdf` và `2.-Bieu-thang-10.2026.xlsx` thành CSDL phủ đủ 5 lĩnh vực (dân số, lao động, giáo dục, y tế, doanh nghiệp) có siêu dữ liệu và bảng ánh xạ địa giới hành chính; bộ **≥430 câu hỏi** (30 câu `dev` tách biệt để chọn siêu tham số + **≥400 câu `test`**) có đáp số chuẩn tính bằng mã độc lập; so sánh 5 phương pháp suy luận (Lời nhắc đơn, ReAct, Self-Consistency, Best-of-*n* + Verifier, MCTS) dùng LLM mã nguồn mở dưới cùng ngân sách token; đạt yêu cầu an toàn (sandbox không mạng) và minh bạch (kèm mã tính toán và nguồn bảng).
  - **Phần B (25đ) & Phần C (15đ):** Hoàn thành đủ 5 chứng minh lý thuyết chặt chẽ và 4 bài tập (bao gồm phân loại lỗi và gán nhãn 50 vết thực thi thật từ Phần D).
  - **Báo cáo & tính tái lập (10đ):** Báo cáo LaTeX 25–40 trang, kho Git có tệp môi trường và script tái lập, Datasheet, Model Card, và phụ lục khai báo sử dụng AI.
- Tối ưu hóa chi phí tính toán để chạy trọn bộ thực nghiệm trên **1 tài khoản Google Colab Pro** mà không bị mất dữ liệu khi ngắt phiên.

### Ngoài phạm vi (Non-goals)
- Không huấn luyện (fine-tune) mô hình LLM hay mô hình Verifier/PRM riêng (tránh rủi ro compute và tiến độ trong 2 tuần).
- Không thu thập/crawl trực tiếp dữ liệu web theo thời gian thực trong lúc tác tử suy luận (sandbox hoàn toàn không có mạng).
- Không xây dựng ứng dụng web thương mại hoàn chỉnh (chỉ ưu tiên CLI, script tái lập và notebook Colab; giao diện demo nếu có chỉ là tùy chọn phụ sau khi xong mốc chính).

### Điều cần tránh (Anti-goals)
- Kết quả số học do LLM tự suy đoán (ảo giác) không qua tính toán kiểm chứng bằng Python trên dữ liệu thật.
- Tiêu tốn hết compute units của Colab Pro vào việc debug code (phải debug cục bộ trên CPU với MockLLM trước).
- Đánh giá các phương pháp ở ngân sách token không tương đương khiến đường cong *accuracy vs. test-time compute* vô nghĩa.
- Dồn Phần B và LaTeX sang ngày cuối cùng khiến không kịp kiểm tra tính chặt chẽ của chứng minh.

---

## 2. Các Ràng buộc (Constraints)

1. **Tài nguyên tính toán:** Chỉ có 1 tài khoản Google Colab Pro. Mọi thao tác suy luận LLM phải hỗ trợ **batching/async qua vLLM**, **lưu cache theo từng request** và **resume khi ngắt phiên**.
2. **Mô hình mã nguồn mở:** Bắt buộc dùng LLM mã nguồn mở (ví dụ `Qwen2.5-7B-Instruct` chạy FP16/BF16 trên L4/A100, hoặc `Qwen2.5-14B-Instruct-AWQ` trên L4 24GB).
3. **An toàn thực thi:** Mã Python do tác tử sinh ra phải chạy trong môi trường cô lập, **không có kết nối mạng**, có giới hạn bộ nhớ, CPU và thời gian chạy.
4. **Độ chính xác, tách tập và tái lập:** Sai số tương đối < 0,1% cho đáp số số học; siêu tham số (ngưỡng Verifier $\tau$, ngân sách mở rộng MCTS) **chỉ được chọn trên tập `dev` 30 câu tách biệt hoàn toàn với tập `test` ≥400 câu**; cố định hạt giống ngẫu nhiên (seed); báo cáo trung bình ± độ lệch chuẩn (hoặc khoảng tin cậy bootstrap trên ≥5 hạt giống/lần chạy).
5. **Liêm chính học thuật & dữ liệu:** Ghi rõ nguồn công khai chính thức (Niên giám Thống kê Toàn quốc 2025, Báo cáo Biểu tháng 10/2026), quy trình số hóa và kiểm định trong Datasheet; không dùng framework agent dựng sẵn cho Phần A; khai báo phạm vi dùng công cụ AI trong phụ lục báo cáo; cả 3 thành viên phải hiểu rõ code và chứng minh để bảo vệ vấn đáp.

---

## 3. Các Hướng tiếp cận đã cân nhắc (Approaches)

| Hướng tiếp cận | Tóm tắt | Ưu điểm | Đánh đổi / Nhược điểm |
|---|---|---|---|
| **Hướng 1: Hybrid Verifier (Đã chọn)** | Tác tử ReAct + Verifier 2 tầng (luật tất định kiểm tra thực thi/schema/trích dẫn + LLM-as-a-Verifier nhẹ qua cùng server vLLM) cho Best-of-*n* và MCTS. | Không cần fine-tune thêm model; tiết kiệm compute Colab Pro; giảm mạnh FPR so với thuần luật; kịp 2 tuần cho nhóm 3 người. | Chất lượng Verifier tầng 2 phụ thuộc thiết kế prompt rubric và độ hiệu chuẩn xác suất của LLM nền. |
| **Hướng 2: Fine-tuned Verifier (LoRA PRM/ORM)** | Thu thập vết chạy trên tập train riêng (~300 câu) rồi fine-tune model 1.5B–3B làm Verifier chuyên biệt theo *Cobbe et al. (2021)*. | Sát bài báo gốc, ấn tượng về mặt học thuật. | Nhân đôi khối lượng việc và compute Colab Pro (sinh dữ liệu train + train LoRA + eval 400 câu × 5 phương pháp) → rủi ro cao trễ hạn 2 tuần. |
| **Hướng 3: Verifier thuần luật + MCTS nông** | Verifier chỉ kiểm tra code chạy không lỗi, ra số hợp lệ, có trích dẫn bảng; MCTS độ sâu 2–3 bước. | Nhanh, rẻ, dễ debug nhất. | FPR cao (code chạy đúng cú pháp nhưng sai công thức vẫn đậu) → Best-of-*n* và MCTS khó vượt trội Self-Consistency (trùng giới hạn lý thuyết B.3). |

**Lý do chọn Hướng 1:** Cân bằng tối ưu giữa độ tin cậy khoa học (tạo được chênh lệch rõ ràng trên đường cong *accuracy vs. compute* và liên hệ trực tiếp với lý thuyết TPR/FPR ở B.3), ngân sách tính toán của 1 Colab Pro và thời hạn 2 tuần.

---

## 4. Thiết kế Chi tiết

*(Lưu ý: Những quyết định kỹ thuật cụ thể dưới đây được đánh dấu **[Đề xuất kỹ thuật]** là các lựa chọn mặc định do kiến trúc sư đề xuất để nhóm triển khai nhanh; nhóm có thể điều chỉnh nếu muốn.)*

### 4.1. Cấu trúc Kho mã nguồn (`VeriStat/`) — *[Đề xuất kỹ thuật]*
```text
VeriStat/
├── pyproject.toml / requirements.txt   # Khóa phiên bản thư viện để tái lập
├── src/veristat/
│   ├── data/                           # Số hóa PDF/Excel (Niên giám 2025 + Biểu 10/2026) → SQLite + admin_mapping
│   ├── tools/                          # 3 công cụ: query_table, search_docs, run_python (sandbox)
│   ├── llm/                            # Client OpenAI-compatible (vLLM), MockLLM, SQLite cache
│   ├── agents/                         # Tự cài đặt từ đầu (Python+NumPy): single_prompt, react, sc, best_of_n, mcts
│   ├── verifier/                       # Hybrid Verifier (Tier 1 deterministic + Tier 2 LLM judge)
│   └── eval/                           # Khung đánh giá A.4, tính metrics, vẽ đồ thị compute-curve
├── data/
│   ├── raw/                            # Tham chiếu tệp gốc NG-TOAN-QUOC-2025-1.pdf & 2.-Bieu-thang-10.2026.xlsx
│   ├── processed/veristat.db           # CSDL SQLite đã số hóa + bảng metadata + admin_mapping + FTS5
│   └── benchmark/                      # dev_30.jsonl (chọn siêu tham số) + benchmark_400.jsonl (tập test ≥400 câu)
├── scripts/                            # Tập lệnh tái lập: build_db.py, gen_benchmark.py, run_eval.py, plot_results.py
├── notebooks/                          # colab_vllm_runner.ipynb (chạy thực nghiệm trên Colab Pro)
├── tests/                              # Unit tests chạy cục bộ trên CPU (sandbox, tools, MCTS, verifier)
├── docs/                               # DATASHEET.md, MODEL_CARD.md, error_taxonomy_50.csv
└── report/                             # Mã nguồn LaTeX báo cáo 25–40 trang (Phần A, B, C, D, Phụ lục)
```

### 4.2. Dữ liệu & Bộ Đánh giá (Phần D.1 & D.2 — Quy trình Data Science: EDA-First & Data Audit)

**Nguyên tắc triển khai (Hướng A — Đã chọn cho Thành viên 1):** Không viết cứng schema, luật làm sạch hay template câu hỏi trước khi khảo sát dữ liệu thật. Toàn bộ công việc của Thành viên 1 đi theo 4 pha Data Science chuẩn:
1. **Pha 1 — EDA & Data Audit (Notebook-first):** Khảo sát trực tiếp Mục lục (TOC) và trang bảng mẫu của 5 lĩnh vực trong `NG-TOAN-QUOC-2025-1.pdf` cùng 20 sheet của `2.-Bieu-thang-10.2026.xlsx`; kiểm toán cấu trúc bảng PDF (header nhiều tầng, bảng kéo dài nhiều trang, định dạng số/giá trị khuyết, đơn vị tính, độ phủ năm), phát hiện dòng tổng hợp (`Cả nước`, 6 vùng kinh tế) nằm lẫn với tỉnh/thành, và trích xuất danh sách biến thể địa phương thực tế (kể cả thay đổi địa giới giữa Niên giám 2025 và Báo cáo 10/2026 nếu có) để xây `admin_mapping` dựa trên dữ liệu thật.
2. **Pha 2 — Số hóa, Chuẩn hóa CSDL & Data Quality Checks:** Đóng gói quy tắc số hóa đã kiểm chứng từ EDA (`pdfplumber`/`pymupdf` cho PDF + `openpyxl` cho Excel) vào `build_db.py` để dựng `veristat.db` (SQLite + FTS5) kèm cột phân loại cấp đơn vị (`geo_level`: `country` / `region` / `province`) tránh cộng trùng hoặc lọt dòng tổng vào câu xếp hạng; chạy bộ kiểm định chất lượng dữ liệu (tỷ lệ null, độ phủ năm, đối chiếu tổng tỉnh vs dòng `Cả nước`, 0 biến thể tỉnh chưa ánh xạ, kiểm tra chéo ngẫu nhiên với trang PDF gốc, truy vấn FTS5 không lỗi).
3. **Pha 3 — Thiết kế Benchmark theo tư duy Thống kê & Sanity Check:** Thiết kế **≥430 câu hỏi** tách thành **`dev_30.jsonl` (30 câu chọn siêu tham số)** và **`benchmark_400.jsonl` (≥400 câu test độc lập)** có ý nghĩa kinh tế–xã hội thật (chỉ xếp hạng cùng `geo_level`, chỉ tính tương quan giữa các cặp chỉ tiêu có liên hệ kinh tế–xã hội hợp lý, loại mẫu số 0/`NaN`/đồng hạng `top-k`), tính đáp số độc lập, kiểm tra phân phối đáp số & outliers, và rà soát chéo 20% mẫu với 2 thành viên còn lại.
4. **Pha 4 — Datasheet & Phân tích lỗi C.4:** Viết `DATASHEET.md` bằng chính số liệu Data Audit từ Pha 1–2; chuẩn bị khung gán nhãn 50 vết thực thi (C.4) có đo độ đồng thuận Cohen's $\kappa$ giữa 2 người gán nhãn.

1. **CSDL Thống kê (`veristat.db` — SQLite + FTS5)** — *[Đề xuất kỹ thuật chọn SQLite thay vì DuckDB vì nhẹ, có sẵn trong Python chuẩn, tích hợp sẵn FTS5 cho tra cứu tài liệu và dễ mở chế độ read-only `mode=ro`]*:
   - Số hóa có chọn lọc **~20–30 bảng trọng tâm** từ `NG-TOAN-QUOC-2025-1.pdf` và **~5–8 sheet trọng tâm** từ `2.-Bieu-thang-10.2026.xlsx` phủ đủ **5 lĩnh vực bắt buộc**:
     1. Dân số (dân số trung bình, mật độ, tỷ suất sinh/tử, đô thị hóa theo tỉnh/năm)
     2. Lao động (lực lượng lao động, tỷ lệ thất nghiệp, năng suất, thu nhập bình quân)
     3. Giáo dục (số trường/lớp/học sinh, tỷ lệ tốt nghiệp, giáo dục đại học)
     4. Y tế (giường bệnh, bác sĩ, cơ sở y tế theo địa phương)
     5. Doanh nghiệp (số doanh nghiệp đang hoạt động, lao động, vốn, doanh thu theo ngành/tỉnh)
   - **Bảng siêu dữ liệu (`tables_catalog`):** `table_id`, `title`, `domain`, `unit` (đơn vị tính chuẩn hóa, ví dụ: nghìn người, %, tỷ đồng), `reporting_period` (khoảng năm), `scope` (cả nước / 63 tỉnh thành / vùng kinh tế), `columns_schema`, `source_ref`, `retrieved_at`.
   - **Bảng ánh xạ địa giới hành chính (`admin_mapping`):** Chuẩn hóa tên tỉnh/thành phố (ví dụ: `TP. Hồ Chí Minh` / `Thành phố Hồ Chí Minh`, `Thừa Thiên Huế` / `Thành phố Huế`, `Bà Rịa - Vũng Tàu`), ánh xạ sáp nhập lịch sử (`Hà Tây` + `Hà Nội (cũ)` → `Hà Nội` từ 2008; ánh xạ tỉnh ↔ 6 vùng kinh tế–xã hội; và bổ sung bảng quy đổi nếu dữ liệu có cập nhật địa giới mới).
   - **Chỉ mục tra cứu tài liệu (`docs_glossary` với FTS5):** Lưu mô tả chi tiết từng bảng, định nghĩa chỉ tiêu thống kê, ghi chú phương pháp tính và ghi chú thay đổi địa giới để công cụ `search_docs` trả về ngữ cảnh chính xác.

2. **Bộ Đánh giá ≥400 Câu hỏi (`benchmark_400.jsonl`):**
   - Phân bố đều theo **4 nhóm câu hỏi phân tích** yêu cầu trong D.2:
     - **Tăng trưởng (~100 câu):** tốc độ tăng trưởng năm sau so với năm trước, CAGR giai đoạn, chênh lệch tuyệt đối/tương đối.
     - **Tỷ trọng (~100 câu):** cơ cấu phần trăm của một địa phương/ngành trên tổng, so sánh tỷ trọng giữa hai năm.
     - **Xếp hạng (~100 câu):** top-k, giá trị lớn nhất/nhỏ nhất, thứ hạng của một tỉnh theo chỉ tiêu đơn hoặc chỉ tiêu dẫn xuất (ví dụ: bác sĩ trên 10.000 dân).
     - **Tương quan (~100 câu):** hệ số tương quan Pearson/Spearman giữa hai chỉ tiêu trên tập các tỉnh hoặc theo chuỗi thời gian.
   - Phân cấp **3 mức độ khó** theo số bước tính toán:
     - *Dễ (1 bước, ~120 câu):* 1 bảng, lọc và tính toán đơn giản.
     - *Trung bình (2–3 bước, ~160 câu):* 1–2 bảng, tính tỷ số/tăng trưởng nhiều năm hoặc xếp hạng theo chỉ tiêu dẫn xuất.
     - *Khó (≥3 bước, ~120 câu):* ghép nối nhiều bảng khác lĩnh vực (ví dụ: giáo dục + dân số) hoặc bắt buộc tra cứu `admin_mapping` (gộp vùng kinh tế / xử lý thay đổi tên–địa giới).
   - **Kiểu đáp số & quy tắc khớp chuẩn** — *[Đề xuất kỹ thuật]*: Hỗ trợ 3 kiểu đáp số chuẩn hóa (`number`, `entity`, `entity_list`) để nhóm câu **Xếp hạng (`top-k`)** tự nhiên và đúng thực tế:
     - `number`: sai số tương đối $< 0,1\%$ (dùng cho tăng trưởng, tỷ trọng, tương quan, giá trị lớn nhất/nhỏ nhất, thứ hạng số nguyên).
     - `entity` / `entity_list`: chuẩn hóa tên địa phương/ngành qua `admin_mapping` rồi so khớp chính xác (có thứ tự cho `top-k`).
   - **Đáp số chuẩn kiểm chứng độc lập:** Mỗi câu hỏi đi kèm một hàm/script Python độc lập chạy trực tiếp trên `veristat.db` để xuất `(ground_truth_value, answer_type, unit, required_table_ids, reference_code)`. Cả 3 thành viên rà soát chéo một mẫu phân tầng (tối thiểu 15–20% số câu) để loại bỏ câu hỏi mơ hồ hoặc lỗi đơn vị.

### 4.3. Lõi Tác tử, Công cụ & Tìm kiếm (Phần A.1 – A.4)

1. **Ba công cụ trong vòng lặp ReAct (A.1):**
   - `search_docs(query, top_k=5)`: Truy vấn FTS5/BM25 trên `tables_catalog`, `docs_glossary` và ghi chú `admin_mapping`. Trả về `table_id`, tiêu đề, đơn vị, kỳ báo cáo, phạm vi và tên cột.
   - `query_table(sql)`: Thực thi câu lệnh **chỉ đọc (`SELECT` only)** trên `veristat.db` mở bằng URI `mode=ro`. Chặn từ khóa DDL/DML, giới hạn tối đa 200 dòng trả về mỗi lần gọi, luôn đính kèm siêu dữ liệu `unit` của bảng được truy vấn.
   - `run_python(code)`: Thực thi mã Python trong **subprocess cô lập không mạng**:
     - Trước khi chạy: kiểm tra AST chặn toàn bộ lệnh/module nguy hiểm (`socket`, `urllib`, `requests`, `http`, `subprocess`, `os`, `shutil`, `open` ghi file, `eval`/`exec`).
     - Khi chạy: chạy tiến trình con với môi trường sạch (`env={}`), thư mục tạm chỉ đọc dữ liệu bảng cho phép (hoặc truyền dữ liệu qua JSON/read-only SQLite), áp dụng `resource.setrlimit` giới hạn RAM (ví dụ 512MB) và CPU time (ví dụ 5 giây), cùng timeout thực (10 giây). Mọi hành vi bị chặn đều ghi nhận là **vi phạm an toàn**. *(Khi chạy máy local có Docker, hỗ trợ thêm tùy chọn backend `--network=none`.)*
   - **Vòng lặp ReAct:** Chuỗi `Thought → Action → Observation` với giới hạn cứng `max_steps=8` và `max_seconds=60`. Đầu ra cuối bắt buộc có cấu trúc gồm: `answer` (số, tên thực thể hoặc danh sách `top-k` theo `answer_type`), `unit`, `cited_tables` (danh sách `table_id`), và `computation_code` (để bảo đảm minh bạch D.5).

2. **Self-Consistency & Best-of-*n* với Hybrid Verifier (A.2):**
   - **Self-Consistency:** Lấy mẫu $n$ quỹ đạo ReAct độc lập (`temperature=0.7`), chuẩn hóa đáp số (gom cụm số học theo ngưỡng sai số tương đối 0,1%, hoặc chuẩn hóa chuỗi/tuple qua `admin_mapping`) rồi bỏ phiếu đa số. Cài đặt thêm **chiến lược bỏ phiếu tuần tự dừng sớm** (dừng ngay khi một đáp án đạt $> n/2$ phiếu) để kiểm chứng thực nghiệm định lý B.5.
   - **Hybrid Verifier (2 tầng):**
     - *Tầng 1 (Tất định — loại nhanh lỗi kỹ thuật):* Kiểm tra (i) không vi phạm an toàn, (ii) có ít nhất 1 lần gọi `query_table` hoặc `run_python` thành công, (iii) mọi `cited_tables` đều tồn tại và thực sự đã được truy vấn trong vết chạy, (iv) đáp số cuối khớp với kết quả xuất ra từ `run_python` (chống LLM tự bịa số/tên ở bước cuối). Nếu trượt Tầng 1 → điểm Verifier = `0.0`.
     - *Tầng 2 (LLM-as-a-Verifier — giảm FPR lỗi logic/công thức):* Gửi `(câu hỏi, tóm tắt dữ liệu đã truy vấn, mã Python tính toán, kết quả & đơn vị)` cho cùng server vLLM với prompt kiểm chứng nhị phân/rubric (`0.0` đến `1.0` dựa trên xác suất token hoặc điểm chấm 3 tiêu chí: đúng bảng/địa giới, đúng công thức, đúng đơn vị).
     - **Quy tắc chọn của Best-of-*n*** — *[Đề xuất kỹ thuật]*: Dùng `argmax` liên tục trên điểm Hybrid Verifier làm quy tắc chính để tối ưu độ chính xác Phần D; đồng thời ghi nhận thêm chế độ ngưỡng nhị phân $\tau$ (hiệu chỉnh trên tập dev 30 câu; nếu 0 ứng viên đạt ngưỡng thì fallback chọn ngẫu nhiên đều trong $n$ ứng viên) để đo thực nghiệm $(\text{TPR}, \text{FPR})$ và đối chiếu trực tiếp công thức B.3 mà không tốn thêm lượt gọi LLM.

3. **MCTS với quy tắc UCT (A.3):**
   - **Không gian trạng thái cây:** Mỗi nút $s$ lưu lịch sử bước `(Thought, Action, Observation)` từ gốc đến độ sâu $d \le d_{\max}$ (đề xuất $d_{\max}=6$).
   - **4 pha chuẩn MCTS:**
     1. *Selection:* Đi từ gốc theo quy tắc UCT: $a^\star = \arg\max_a \left[\bar{Q}(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}}\right]$ với $c = \sqrt{2}$ (ưu tiên nút chưa thăm $N(s,a)=0$).
     2. *Expansion:* Tại nút lá chưa kết thúc, gọi LLM sinh $k$ hành động ứng viên đa dạng (ví dụ $k=2$ với `temperature=0.7`), thực thi công cụ tương ứng để tạo nút con.
     3. *Evaluation (Simulation/Value)* — *[Đề xuất kỹ thuật]*: **Nút trung gian** chỉ chấm bằng heuristic tất định Tầng 1 $v(s) \in [0,1]$ (thưởng gọi đúng công cụ/ra dữ liệu không rỗng/trích dẫn bảng hợp lệ; phạt lỗi cú pháp, dữ liệu rỗng, lặp hành động) để không đốt 30–50% ngân sách token vào việc gọi LLM chấm từng nút con; **nút kết thúc** được chấm đầy đủ bằng Hybrid Verifier (Tầng 1 + Tầng 2).
     4. *Backpropagation:* Cập nhật $N(s) \leftarrow N(s)+1$ và trung bình trượt $\bar{Q}(s,a)$ dọc đường về gốc.
   - **Kết thúc:** Khi hết ngân sách vòng lặp / ngân sách token, chọn nút kết thúc có điểm Verifier cao nhất (hoặc nút có số lần thăm cao nhất).

4. **Khung Đánh giá, Tái lập Thống kê & So sánh Ngân sách Token (A.4 & D.3–D.4):**
   - So sánh 5 phương pháp trên cùng tập ≥400 câu hỏi:
     1. **Single Prompt (Baseline):** Đưa schema tóm tắt + câu hỏi, 1 lần gọi LLM không dùng công cụ.
     2. **ReAct ($n=1$):** 1 quỹ đạo ReAct.
     3. **Self-Consistency:** $n \in \{3, 5, 8\}$ quỹ đạo ReAct (đo thêm số lần gọi tiết kiệm được khi dùng dừng sớm B.5).
     4. **Best-of-*n* + Hybrid Verifier:** $n \in \{3, 5, 8\}$ quỹ đạo ReAct + chi phí token của Verifier (tính gộp vào tổng ngân sách token).
     5. **MCTS-UCT:** Chạy ở 3 mức ngân sách mở rộng nút đã hiệu chỉnh trên tập dev 30 câu để tổng token thực tế tương đương các mức $n \in \{3, 5, 8\}$ của SC/BoN.
   - **Cách vận hành "cùng ngân sách token"** — *[Đề xuất kỹ thuật]*: Áp dụng kiểm tra trần token cứng trước mỗi lượt sinh/mở rộng mới (nếu chạm trần thì dừng và lấy kết quả tốt nhất hiện có), kết hợp hiệu chỉnh siêu tham số trên tập dev 30 câu và vẽ **đường cong độ chính xác theo tổng token thực chi trung bình** (tính đủ prompt + completion + verifier token).
   - **Tái lập thống kê phù hợp 1 Colab Pro** — *[Đề xuất kỹ thuật]*: Sinh một **bể quỹ đạo dùng chung $N=12$ quỹ đạo ReAct/câu** trên Colab Pro, từ đó lấy mẫu con theo **5 hạt giống độc lập** để báo cáo `trung bình ± độ lệch chuẩn` cho ReAct ($n=1$), Self-Consistency ($n \in \{3,5,8\}$, cả đầy đủ lẫn dừng sớm) và Best-of-*n* ($n \in \{3,5,8\}$) mà không phải nhân 5× chi phí GPU. Với MCTS-UCT, chạy 1 lượt đầy đủ trên cả 400 câu kèm khoảng tin cậy bootstrap (1.000 lần lấy mẫu lại) + lặp 5 hạt giống trên tập con phân tầng 80 câu (hoặc full 400 câu nếu còn dư compute units).
   - **Độ đo bắt buộc:** Độ chính xác (sai số tương đối $< 0,1\%$ với số; khớp chuẩn hóa với `entity`/`entity_list`), tỷ lệ trích dẫn đúng bảng nguồn (`table_id` exact/set match), số lời gọi công cụ trung bình, tổng token (input + output), độ trễ (giây/câu), tỷ lệ vi phạm ràng buộc an toàn, và đường cong độ chính xác theo ngân sách tính toán.
   - **Chiến lược tiết kiệm Colab Pro:** Toàn bộ lời gọi LLM được cache theo `(model, prompt_hash, temperature, seed)` vào SQLite; mỗi lần chạy eval ghi checkpoint theo `question_id` để ngắt phiên có thể chạy tiếp ngay lập tức.

### 4.4. Phần Lý thuyết (B), Bài tập (C) & Báo cáo LaTeX

- **Phần B (5 chứng minh):** Trình bày đầy đủ giả thiết, ký hiệu thống nhất với tài liệu gốc: (B.1) mô hình hóa tác tử là POMDP và dẫn xuất công thức cập nhật niềm tin Bayes; (B.2) cận Hoeffding cho bỏ phiếu đa số nhị phân + điều kiện $p_{\text{correct}} > \max_{j \ne \text{correct}} p_j$ cho đa lớp; (B.3) xác suất chọn đúng của Best-of-*n* với bộ kiểm chứng không hoàn hảo (xét đầy đủ trường hợp không có lời giải nào được chấp nhận) và giới hạn $n\to\infty$; (B.4) cận $O(\ln N(s))$ của UCT từ kết quả UCB1; (B.5) chứng minh tính tương đương đầu ra và bất đẳng thức kỳ vọng số lời gọi của bỏ phiếu tuần tự dừng sớm.
- **Phần C (4 bài tập):**
  - **C.1:** Với $p = 0,6$, tính chính xác xác suất bỏ phiếu đa số đúng cho $n \in \{1, 5, 15\}$ (phân phối nhị thức) và đối chiếu với cận Hoeffding ở B.2 bằng script kiểm chứng.
  - **C.2:** Với $p = 0,3$, $\text{TPR} = 0,9$, $\text{FPR} = 0,1$, tính xác suất chọn đúng theo B.3 cho $n \in \{1, 4, 16\}$ và giới hạn $n \to \infty$ (ghi rõ cả hai quy ước khi 0 lời giải được chấp nhận: fallback ngẫu nhiên hoặc từ chối — cả hai đều hội tụ cùng giới hạn $0,7941$), kèm phân tích giới hạn của test-time compute.
  - **C.3:** Chuẩn bị sẵn khung trình bày tay 2 vòng MCTS ($c=\sqrt{2}$) để điền ngay cây quyết định do giảng viên cung cấp (hoặc dùng cây mẫu minh họa nếu giảng viên chưa gửi).
  - **C.4:** Định nghĩa bảng phân loại lỗi tác tử (chọn sai công cụ, tham số/SQL sai, lặp vô hạn, ảo giác số liệu, sai đơn vị/địa giới) và **gán nhãn thủ công trên 50 vết thực thi thật** lấy ngẫu nhiên phân tầng từ kết quả chạy Phần D.
- **Sản phẩm nộp:** Báo cáo LaTeX 25–40 trang, `DATASHEET.md` (theo Gebru et al., 2021), `MODEL_CARD.md` (theo Mitchell et al., 2019), kho Git tái lập và phụ lục khai báo sử dụng AI.

---

## 5. Lộ trình 2 Tuần & Phân công 3 Thành viên — *[Đề xuất kỹ thuật]*

- **Long (Data & Benchmark Lead):** Số hóa trực tiếp `NG-TOAN-QUOC-2025-1.pdf` & `2.-Bieu-thang-10.2026.xlsx` (theo Hướng A: EDA-First & Data Audit) → `veristat.db`, xây `admin_mapping` & `docs_glossary`, sinh và kiểm chứng bộ ≥430 câu hỏi (`dev_30` + `benchmark_400`), viết `DATASHEET.md`, phụ trách C.4 (cùng nhóm gán nhãn 50 vết).
- **Khoa (Agent & Sandbox Lead):** Xây 3 công cụ (`query_table`, `search_docs`, `run_python` sandbox cô lập), vòng lặp ReAct (tự cài đặt từ đầu), lớp gọi vLLM + cache/resume trên Colab Pro, viết `MODEL_CARD.md`.
- **Sơn (Search, Verifier & Theory Lead):** Tự cài đặt từ đầu Self-Consistency (có dừng sớm), Hybrid Verifier, MCTS-UCT, khung đánh giá & biểu đồ compute-curve; dẫn dắt khung chứng minh Phần B & bài tập C.1–C.3.

| Giai đoạn | Thời gian | Mục tiêu chính |
|---|---|---|
| **Giai đoạn 1: Dữ liệu & Lõi A (cục bộ CPU)** | Ngày 1 – 4 | Hoàn thiện `veristat.db` (số hóa 5 lĩnh vực + `admin_mapping`), bộ ≥430 câu hỏi (`dev_30` + `benchmark_400`) có đáp số chuẩn, và chạy pass 100% unit test cho ReAct/SC/Best-of-*n*/MCTS bằng MockLLM. |
| **Giai đoạn 2: Chạy Thực nghiệm Colab Pro (Phần D)** | Ngày 5 – 8 | Triển khai vLLM trên Colab Pro, hiệu chỉnh prompt/ngưỡng Verifier/ngân sách MCTS trên tập `dev_30` tách biệt, sau đó chạy full ≥400 câu `test` × 5 phương pháp (có cache/resume), xuất bảng kết quả & đồ thị compute-curve. |
| **Giai đoạn 3: Lý thuyết B, Bài tập C & Báo cáo** | Ngày 9 – 12 | Hoàn thiện 5 chứng minh Phần B, giải C.1–C.3, cả 3 thành viên cùng gán nhãn lỗi 50 vết thực thi (C.4), viết báo cáo LaTeX 25–40 trang + Datasheet + Model Card. |
| **Giai đoạn 4: Kiểm tra chéo & Chuẩn bị bảo vệ** | Ngày 13 – 14 | Chạy lại script tái lập từ đầu trên môi trường sạch, rà soát toàn bộ chứng minh và công thức LaTeX, ôn tập vấn đáp chéo giữa 3 thành viên. |

---

## 6. Phần để lại cho giai đoạn lập kế hoạch & cài đặt (Left to the Builder)
- Chi tiết DDL của từng bảng thống kê cụ thể sau khi hoàn thành Pha 1 EDA trên `NG-TOAN-QUOC-2025-1.pdf` và `2.-Bieu-thang-10.2026.xlsx`.
- Prompt template chi tiết cho ReAct, Single Prompt và Hybrid Verifier Tầng 2.
- Danh sách whitelist chi tiết cho AST của Python sandbox và cấu hình batching vLLM tối ưu theo GPU thực tế Colab Pro cấp (L4 vs A100).
- Mẫu LaTeX và kịch bản vẽ biểu đồ matplotlib/seaborn cho báo cáo.
