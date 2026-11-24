# Hướng dẫn Phân loại Lỗi C.4 (Error Taxonomy Guide)

Tài liệu này quy định **6 nhãn lỗi chuẩn** cho phân tích định tính C.4 trên 50 vết thực thi (execution traces) được lấy mẫu phân tầng từ khung đánh giá A.4 của **VeriStat**, kèm quy trình gán nhãn độc lập bởi 2 thành viên và đo độ đồng thuận bằng hệ số **Cohen's $\kappa$**.

---

## 1. Sáu Nhãn Lỗi Chuẩn (Primary Root Cause)

Mỗi vết thực thi sai chỉ được gán **1 nhãn nguyên nhân gốc** (nếu nhiều lỗi xảy ra dây chuyền, chọn lỗi **đầu tiên** làm lệch quỹ đạo suy luận).

| # | Mã nhãn (`label`) | Định nghĩa ngắn | Ví dụ điển hình trong VeriStat |
|---|---|---|---|
| 1 | `wrong_tool` | Chọn sai công cụ hoặc gọi sai bảng/tài liệu không chứa chỉ tiêu cần tìm. | Cần số liệu *Dân số trung bình* (`NG2025_T017`) nhưng lại truy vấn bảng *Dân số thành thị* (`NG2025_T020`), hoặc dùng `search_docs` thay vì `query_table`. |
| 2 | `bad_args_or_sql` | Chọn đúng công cụ/bảng nhưng viết sai cú pháp SQL, sai tên cột, thiếu `WHERE geo_level='province'`, hoặc truyền sai tham số. | Quên lọc `geo_level='province'` khiến truy vấn `ORDER BY gia_tri DESC` trả về `"Cả nước"` hoặc vùng kinh tế; hoặc viết sai tên bảng `stat_ng2025_t17` thay vì `stat_ng2025_t017`. |
| 3 | `infinite_loop` | Agent lặp đi lặp lại cùng một hành động/truy vấn hoặc chạm giới hạn bước tối đa (`max_steps`) mà không hội tụ ra đáp án. | Gọi lại cùng câu SQL bị lỗi 3–5 lần liên tiếp mà không sửa tham số, hoặc hết ngân sách bước trước khi xuất `Final Answer`. |
| 4 | `number_hallucination` | Tự bịa số liệu, đọc nhầm số từ kết quả bảng, hoặc trích dẫn giá trị không tồn tại trong kết quả công cụ trả về. | Kết quả SQL trả về `101343.8` nhưng trong lập luận cuối agent viết `103143.8` hoặc tự ước lượng dân số Hà Nội mà không gọi `query_table`. |
| 5 | `unit_or_admin_mismatch` | Nhầm đơn vị đo (nghìn người vs người, tỷ đồng vs triệu đồng, % vs ‰) hoặc sai ánh xạ hành chính 63 tỉnh (2025) $\leftrightarrow$ 34 tỉnh sau sáp nhập (NQ 202/2025/QH15). | Chia số lao động doanh nghiệp (đơn vị *người*, `NG2025_T139`) cho lao động có việc làm (đơn vị *nghìn người*, `NG2025_T053`) mà quên nhân `1000`; hoặc tính *TP. Hồ Chí Minh mới* nhưng chỉ lấy TP.HCM cũ mà quên gộp Bình Dương và Bà Rịa - Vũng Tàu. |
| 6 | `wrong_formula_or_logic` | Lấy đúng số liệu, đúng đơn vị và đúng địa giới nhưng áp dụng sai công thức toán/thống kê hoặc đảo chiều logic. | Tính tốc độ tăng trưởng bằng `(y1 - y0) / y1` thay vì `(y1 - y0) / y0`; tính hệ số tương quan sai tập giao 63 tỉnh; hoặc xếp hạng tăng dần khi câu hỏi yêu cầu giảm dần. |

---

## 2. Quy trình Gán nhãn & Tính Cohen's $\kappa$ (Ngày 9–10)

1. **Lấy mẫu phân tầng 50 vết** (ưu tiên vết sai, phủ đủ các phương pháp `Direct` / `ReAct` / `MCTS` và độ khó):
   ```bash
   python scripts/prepare_c4_traces.py \
       --traces results/eval_traces.jsonl \
       --out docs/error_taxonomy_50.csv \
       --n 50 --seed 42
   ```
2. **Gán nhãn độc lập (Blind Annotation):**
   - Thành viên 1 điền cột `annotator_1` (ẩn/không xem cột `annotator_2`).
   - Thành viên 2 điền cột `annotator_2` độc lập trên cùng 50 dòng.
   - Chỉ sử dụng đúng 6 mã nhãn ở bảng trên (viết thường, đúng chính tả).
3. **Đo độ đồng thuận Cohen's $\kappa$:**
   ```bash
   python scripts/prepare_c4_traces.py --evaluate docs/error_taxonomy_50.csv
   ```
   - Mức mục tiêu: $\kappa \ge 0.75$ (đồng thuận tốt / substantial–almost perfect theo Landis & Koch, 1977).
4. **Hòa giải (Adjudication):**
   - Với các dòng mà `annotator_1 != annotator_2`, hai thành viên thảo luận (có thể tham khảo thành viên thứ 3) để chốt `final_label` và ghi ngắn gọn lý do vào cột `notes`.
