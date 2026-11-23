# Hướng dẫn Tái tạo CSDL VeriStat (`veristat.db`) từ 32 file CSV sạch

Thư mục `data/interim/tables/` đã lưu sẵn **32 file CSV sạch** (`utf-8-sig`) được số hóa và chuẩn hóa từ:
- **Niên giám Thống kê Toàn quốc 2025** (`NG-TOAN-QUOC-2025-1.pdf`) — 27 bảng (cơ cấu 63 tỉnh + 6 vùng + Cả nước và các bảng cấp quốc gia).
- **Báo cáo tình hình kinh tế – xã hội tháng 10/2026** (`2.-Bieu-thang-10.2026.xlsx`) — 5 bảng (cơ cấu 34 tỉnh sau sáp nhập 2025 + Cả nước và các bảng doanh nghiệp 10 tháng năm 2026).

Đồng đội có thể dựng lại hoàn chỉnh `data/processed/veristat.db` **trong ~1 giây mà không cần tải file PDF/Excel gốc**.

---

## 1. Cấu trúc thư mục

```text
data/
├── README.md
├── interim/
│   └── tables/           # 32 file CSV sạch (đã commit trong repo)
│       ├── NG2025_T015.csv
│       ├── ...
│       └── B1026_S10.csv
└── processed/
    └── veristat.db       # SQLite + FTS5 (tạo ra sau khi chạy build_db.py)
```

---

## 2. Tái tạo `veristat.db` từ 32 file CSV sạch (Khuyên dùng)

### Bước 1: Cài đặt môi trường
```bash
uv sync
# Hoặc nếu dùng venv sẵn có:
source .venv/bin/activate
```

### Bước 2: Dựng `data/processed/veristat.db` từ CSV
```bash
python scripts/build_db.py \
  --selection docs/table_selection.yaml \
  --out data/processed/veristat.db \
  --from-csv data/interim/tables \
  --no-csv
```

Kết quả mong đợi:
```json
{
  "db_path": "data/processed/veristat.db",
  "admin_mapping_rows": 93,
  "tables_ingested": 32,
  "total_data_rows": 6968,
  "csv_dir": null
}
```

### Bước 3: Chạy kiểm định chất lượng dữ liệu
```bash
python scripts/check_data_quality.py \
  --db data/processed/veristat.db \
  --report docs/data_quality_report.json
```

Kết quả mong đợi:
- `Status: PASS` (exit code `0`)
- Phủ đủ 5 lĩnh vực: `dan_so` (6), `lao_dong` (6), `giao_duc` (6), `y_te` (6), `doanh_nghiep` (8) → **32 bảng**
- `unexpected_other_count: 0`
- `additive_checks_passed: 36 / 36` (`max_additive_rel_diff_pct: 0.0008%`)

### Bước 4: Chạy unit tests kiểm tra schema, `admin_mapping`, ETL & FTS5
```bash
pytest tests/data/test_schema_and_admin.py tests/data/test_digitization_etl.py -v
```

---

## 3. Tái tạo lại từ đầu từ PDF & Excel gốc *(tùy chọn)*

Nếu máy bạn có sẵn `NG-TOAN-QUOC-2025-1.pdf` và `2.-Bieu-thang-10.2026.xlsx` ở thư mục gốc dự án, chạy lệnh sau để số hóa lại từ đầu, xuất đè 32 file CSV và dựng `veristat.db`:

```bash
python scripts/build_db.py \
  --selection docs/table_selection.yaml \
  --out data/processed/veristat.db \
  --csv-dir data/interim/tables
```

---

## 4. Hướng dẫn dùng `veristat.db` cho Agent / Phân tích

### Mở CSDL ở chế độ chỉ đọc (`mode=ro`) & tra cứu bảng
```python
from veristat.data.schema import open_readonly
from veristat.data.etl import search_docs_fts

conn = open_readonly("data/processed/veristat.db")

# 1. Tìm bảng phù hợp bằng FTS5 (đã lọc ký tự đặc biệt an toàn)
hits = search_docs_fts(conn, "dân số trung bình", top_k=5)

# 2. Xem danh mục toàn bộ 32 bảng
tables = conn.execute(
    "SELECT table_id, sql_table, domain, unit, reporting_period FROM tables_catalog"
).fetchall()
```

### Lưu ý quan trọng khi viết SQL
1. **Luôn lọc theo `geo_level` trên các bảng cấp tỉnh** (`country` / `region` / `province` / `other`):
   - Khi xếp hạng `TOP-k` tỉnh hoặc tính `SUM` theo tỉnh, bắt buộc thêm `WHERE geo_level = 'province'` để không bị lọt dòng `Cả nước` hoặc 6 vùng kinh tế vào kết quả.
2. **Sáp nhập địa giới 63 → 34 tỉnh từ 01/07/2025 (NQ 202/2025/QH15):**
   - Các bảng `stat_ng2025_*` (Niên giám 2025) dùng cơ cấu **63 tỉnh**.
   - Bảng `stat_b1026_s05` (Biểu tháng 10/2026) dùng cơ cấu **34 tỉnh**.
   - Khi cần đối chiếu giữa 2 giai đoạn, `JOIN` với bảng `admin_mapping` (cột `canonical_name` → `canonical_2026`).
