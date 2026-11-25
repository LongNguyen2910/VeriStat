# Nhật ký Khai báo Sử dụng Công cụ AI Tạo sinh (Phụ lục Báo cáo)

## Bảng nhật ký sử dụng AI

### Long
- **Công cụ sử dụng:** Antigravity CLI (mô hình Gemini bản trả phí - Gemini 4 Neon)

| Ngày | Hạng mục / Tệp liên quan | Phạm vi AI hỗ trợ | Thành viên tự kiểm chứng & chỉnh sửa |
| :--- | :--- | :--- | :--- |
| 2026-11-22 | Khảo sát dữ liệu & chọn bảng (`audit.py`, `table_selection.yaml`) | Gợi ý cấu trúc script quét mục lục PDF/Excel và tổng hợp `data_audit.json`. | Chọn 32 bảng có đủ 5 lĩnh vực, đối chiếu cấu trúc 63 tỉnh (2025) vs 34 tỉnh (2026), chỉnh sửa lại logic đếm trang pdf so với trang sách gốc |
| 2026-11-22 | Schema SQLite & ánh xạ địa giới (`schema.py`, `admin_mapping.py`) | Hỗ trợ viết DDL SQLite/FTS5, chuẩn hóa Unicode/dấu thanh tiếng Việt và nạp bảng `admin_mapping`. | Đối chiếu toàn bộ bảng sáp nhập 63 -> 34 tỉnh theo NQ 202/2025/QH15, check lại hàm `admin_mapping` và test. |
| 2026-11-23 | Bộ số hóa PDF/Excel, ETL & kiểm định (`pdf_parser.py`, `excel_parser.py`, `etl.py`, `build_db.py`, `check_data_quality.py`, `data/README.md`) | Hỗ trợ viết parser theo tọa độ PyMuPDF/openpyxl, hàm ETL, script kiểm định chất lượng và tài liệu tái tạo. | Check logic lại từng hàm, phát hiện logic parsepdf thiếu, sửa lại cột nam trong csdl thành INTEGER (trước đó là FLOAT); yêu cầu xuất 32 file CSV sạch và lược bỏ trích xuất glossary dài từ PDF; đối chiếu kiểm định (`36/36` tổng tỉnh vs `Cả nước`, lệch tối đa `0.0008%`). |
| 2026-11-24 | Thiết kế & kiểm chứng Benchmark (`benchmark_gen.py`, `gen_benchmark.py`, `verify_benchmark.py`, `dev_30.jsonl`, `benchmark_400.jsonl`) | Hỗ trợ viết bộ sinh 430 câu hỏi (4 nhóm × 3 độ khó) kèm `reference_code` độc lập và script kiểm chứng `verify_benchmark.py`. | Đối chiếu yêu cầu Đề tài 18 (D.2, D.4); Kiểm tra lại các cặp tương quan đảm bảo đúng ý nghĩa thực tế; Đảm bảo logic không lặp lại; Kiểm tra thử một số câu benchmark và sửa đổi văn phong giống tự nhiên |

### Sơn
- **Công cụ sử dụng:** Antigravity (mô hình Gemini)

| Ngày | Hạng mục / Tệp liên quan | Phạm vi AI hỗ trợ | Thành viên tự kiểm chứng & chỉnh sửa |
| :--- | :--- | :--- | :--- |
| 2026-11-25 | Thiết kế chi tiết & Hợp đồng Giao diện (`docs/SON_DESIGN_AND_CONTRACT.md`, `src/veristat/agents/types.py`) | Hỗ trợ phân tích kho mã nguồn, đề xuất kiến trúc Step-level Protocol + Offline Pool Replay và viết đặc tả dataclass/Protocol dùng chung với Khoa. | Rà soát từng phần thiết kế (Understanding Lock, 4 phần kỹ thuật, Decision Log D1–D6), tách nhánh riêng `feat/son-search-verifier` và đối chiếu phạm vi với `docs/DESIGN.md`. |
| 2026-11-25 | Self-Consistency (B.5), Hybrid Verifier 2 tầng & Best-of-*n* (`sc.py`, `verifier/`, `best_of_n.py`, `tests/agents/`, `tests/verifier/`) | Hỗ trợ cài đặt thuần Python + NumPy cho gom cụm đáp án, dừng sớm tuần tự, 4 luật Tầng 1, rubric Tầng 2, hiệu chỉnh ngưỡng $\tau$ và 23 unit tests CPU. | Kiểm tra tách biệt khỏi code của Khoa, xác minh 4 luật Tầng 1 chặn gọi LLM Tầng 2, kiểm chứng tính tương đương dừng sớm B.5 và chạy pass 23/23 tests cục bộ. |
| 2026-11-25 | MCTS-UCT, Khung đánh giá A.4, Biểu đồ & Kiểm chứng C.1–C.3 (`mcts.py`, `eval/`, `scripts/run_eval.py`, `plot_results.py`, `verify_theory_c.py`, `tests/eval/`) | Hỗ trợ cài đặt MCTS 4 pha, checkpoint JSONL, replay 5 hạt giống trên pool $N=12$, bootstrap CI, xuất đồ thị compute-curve và script kiểm chứng số học/Monte Carlo C.1–C.3. | Kiểm chứng công thức UCT ($10^{-9}$), kiểm tra công thức B.3 ($p_{\text{rej}}$ khi 0 ứng viên đạt ngưỡng, khớp $n=1 \Rightarrow 0{,}3$ và Monte Carlo 100.000 lần), chạy pass 32/32 tests CPU + 9/9 tests data offline và thử nghiệm end-to-end với `prepare_c4_traces.py`. |
| 2026-11-25 | Rà soát chéo Benchmark (`data/benchmark/crosscheck_sample.csv`) | Hỗ trợ chạy lại độc lập `reference_code` của 80 câu mẫu trên `veristat.db` và quét đồng hạng nội bộ / đơn vị / địa giới. | Đối chiếu trực tiếp 80/80 câu (79 đạt, phát hiện đồng hạng nội bộ `3.97%` ở `TEST_RAN_288`, lỗi lặp từ `TEST_COR_343`, và lưu ý năm giao nhau 2020/2022–2024 ở `TEST_COR_347`, `TEST_COR_315`, `TEST_COR_325`). |

## Cam kết liêm chính
- Toàn bộ số liệu trong `veristat.db` và 32 file CSV được trích xuất trực tiếp từ 2 nguồn công khai chính thức (Niên giám Thống kê Toàn quốc 2025 và Báo cáo KTXH tháng 10/2026), **không dùng AI sinh số liệu giả**.
- Thành viên đã rà soát từng hàm, hiểu rõ logic cài đặt và chịu trách nhiệm bảo vệ vấn đáp trước hội đồng.
