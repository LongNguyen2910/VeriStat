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

## Cam kết liêm chính
- Toàn bộ số liệu trong `veristat.db` và 32 file CSV được trích xuất trực tiếp từ 2 nguồn công khai chính thức (Niên giám Thống kê Toàn quốc 2025 và Báo cáo KTXH tháng 10/2026), **không dùng AI sinh số liệu giả**.
- Thành viên đã rà soát từng hàm, hiểu rõ logic cài đặt và chịu trách nhiệm bảo vệ vấn đáp trước hội đồng.
