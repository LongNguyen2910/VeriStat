# Nhật ký Khai báo Sử dụng Công cụ AI Tạo sinh (Phụ lục Báo cáo)

> **Quy định đề tài:** Khai báo phạm vi sử dụng công cụ AI tạo sinh theo mục *Liêm chính học thuật* (Trang 2, *Danh mục Bài tập lớn Cuối kỳ môn Học máy Cơ bản*). Không sử dụng bản AI miễn phí.

- **Đề tài:** Đề tài 18 — VeriStat (Phần dữ liệu & Benchmark — Thành viên: Long)
- **Công cụ sử dụng:** Antigravity CLI (mô hình Gemini bản trả phí)

## Bảng nhật ký sử dụng AI

| Ngày | Hạng mục / Tệp liên quan | Phạm vi AI hỗ trợ | Thành viên tự kiểm chứng & chỉnh sửa |
| :--- | :--- | :--- | :--- |
| 2026-11-22 | Khảo sát dữ liệu & chọn bảng (`audit.py`, `table_selection.yaml`) | Gợi ý cấu trúc script quét mục lục PDF/Excel và tổng hợp `data_audit.json`. | Tự chọn 32 bảng phủ 5 lĩnh vực, đổi tên khóa `pdf_pages`, đối chiếu cấu trúc 63 tỉnh (2025) vs 34 tỉnh (2026). |
| 2026-11-22 | Schema SQLite & ánh xạ địa giới (`schema.py`, `admin_mapping.py`) | Hỗ trợ viết DDL SQLite/FTS5, chuẩn hóa Unicode/dấu thanh tiếng Việt và nạp bảng `admin_mapping`. | Đối chiếu toàn bộ bảng sáp nhập 63 → 34 tỉnh theo NQ 202/2025/QH15, kiểm tra từng dòng `admin_mapping` (93 dòng) và chạy `pytest`. |
| 2026-11-23 | Bộ số hóa PDF/Excel, ETL & kiểm định (`pdf_parser.py`, `excel_parser.py`, `etl.py`, `build_db.py`, `check_data_quality.py`, `data/README.md`) | Hỗ trợ viết parser theo tọa độ PyMuPDF/openpyxl, hàm ETL, script kiểm định chất lượng và tài liệu tái tạo. | Nghiệm thu từng bước theo TDD; tự phát hiện và yêu cầu sửa lỗi bỏ sót trang 849 (`Chỉ số phát triển` bảng `T332`), năm 2025 bảng `T338`, kiểu `INTEGER` cột `nam`; yêu cầu xuất 32 file CSV sạch và lược bỏ trích xuất glossary dài từ PDF; đối chiếu kiểm định (`36/36` tổng tỉnh vs `Cả nước`, lệch tối đa `0.0008%`). |

## Cam kết liêm chính
- Toàn bộ số liệu trong `veristat.db` và 32 file CSV được trích xuất trực tiếp từ 2 nguồn công khai chính thức (Niên giám Thống kê Toàn quốc 2025 và Báo cáo KTXH tháng 10/2026), **không dùng AI sinh số liệu giả**.
- Thành viên đã rà soát từng hàm, hiểu rõ logic cài đặt và chịu trách nhiệm bảo vệ vấn đáp trước hội đồng.
