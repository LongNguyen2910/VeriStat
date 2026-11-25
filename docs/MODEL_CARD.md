# Thẻ Mô hình (Model Card): Qwen2.5-7B-Instruct trong Hệ thống VeriStat

Tài liệu này đặc tả việc sử dụng mô hình ngôn ngữ lớn `Qwen/Qwen2.5-7B-Instruct` làm "bộ não" lập luận trong hệ thống **VeriStat** (Đề tài 18: Tác tử Phân tích Số liệu Thống kê Kinh tế – Xã hội Việt Nam có Đáp số Kiểm chứng được). Cấu trúc tài liệu tuân thủ chặt chẽ 9 mục chuẩn của Mitchell et al. (2019) (*Model Cards for Model Reporting*), phân định rạch ròi giữa thông số kỹ thuật do nhà phát triển công bố với kết quả thực nghiệm thực tế của nhóm đề tài.

---

## 1. Chi tiết Mô hình (Model Details)

- **Tên mô hình:** `Qwen/Qwen2.5-7B-Instruct`
- **Nhà phát triển:** Alibaba Cloud (Nhóm tác giả Qwen)
- **Kiến trúc mô hình:** Transformer tự hồi quy (Autoregressive Transformer) kết hợp cơ chế mã hóa vị trí xoay RoPE (Rotary Position Embeddings), chú ý nhóm truy vấn GQA (Grouped Query Attention), hàm kích hoạt SwiGLU và chuẩn hóa RMSNorm.
- **Quy mô tham số:** Khoảng 7,61 tỷ tham số (~7.61B).
- **Độ dài ngữ cảnh (Context Length):**
  - *Cấu hình mặc định trong VeriStat:* 32.768 tokens (32K).
  - *Khả năng mở rộng của kiến trúc:* Hỗ trợ tối đa lên đến 131.072 tokens (128K) tùy thuộc vào phiên bản và cấu hình cấp phát bộ nhớ KV cache khi triển khai.
  - *Lưu ý triển khai trong VeriStat:* Ngữ cảnh hoạt động được giới hạn tối ưu trên vLLM nhằm cân bằng bộ nhớ KV cache GPU khi sinh song song nhiều quỹ đạo suy luận trong Best-of-*n* và MCTS-UCT.
- **Định dạng độ chính xác số học:** BF16 / FP16.
- **Công cụ phục vụ suy luận (Inference Engine):** vLLM (giao diện tương thích chuẩn OpenAI API).
- **Môi trường phần cứng triển khai:** NVIDIA L4 24GB hoặc A100 40GB (Google Colab Pro).
- **Giấy phép bản quyền:** Apache 2.0.
- **Nguồn mô hình:** Kho lưu trữ chính thức trên Hugging Face (`Qwen/Qwen2.5-7B-Instruct`).

---

## 2. Mục đích Sử dụng (Intended Use)

### 2.1. Mục đích Sử dụng Chính
- Đóng vai trò hạt nhân lập kế hoạch (planner) và điều phối công cụ trong vòng lặp ReAct của VeriStat.
- Sinh các câu lệnh truy vấn SQL chỉ đọc (`SELECT`) hợp lệ để thực thi qua công cụ `query_table`.
- Sinh các đoạn mã Python tính toán độc lập để thực thi trong môi trường cô lập `run_python`.
- Sinh các hành động ứng viên phục vụ bỏ phiếu đa số trong Self-Consistency, xây dựng bể ứng viên cho Best-of-*n* và mở rộng nút cây trong MCTS-UCT.

### 2.2. Các Trường hợp Sử dụng Nằm ngoài Phạm vi và Bị nghiêm cấm (Out-of-Scope)
- **Tính nhẩm trực tiếp không qua công cụ:** Tuyệt đối không dựa vào xác suất sinh từ ngẫu nhiên của mô hình để đưa ra đáp số số học phức tạp mà không qua kiểm chứng bằng mã Python.
- **Tự suy diễn số liệu thống kê:** Không đưa ra các sự kiện hoặc con số thống kê nằm ngoài dữ liệu gốc của Cục Thống kê (GSO).
- **Coi lập luận của mô hình là chân lý:** Không chấp nhận chuỗi suy nghĩ (chain-of-thought) do mô hình sinh ra là bằng chứng xác thực nếu chưa qua đối chiếu vết thực thi công cụ.
- **Thực thi mã ngoài sandbox:** Nghiêm cấm chạy trực tiếp mã Python do mô hình sinh ra trên máy chủ thật mà không thông qua môi trường sandbox cô lập không mạng.

---

## 3. Các Yếu tố Chi phối (Factors)

Hiệu năng và độ tin cậy của mô hình trong hệ thống VeriStat chịu sự chi phối trực tiếp của các yếu tố đặc thù:
1. **Thuật ngữ Thống kê Tiếng Việt:** Xử lý các khái niệm chuyên ngành kinh tế – xã hội (ví dụ: *Tổng sản phẩm trên địa bàn (GRDP)*, *tỷ suất sinh thô*, *hệ số tương quan Pearson*, *tốc độ tăng trưởng kép hàng năm CAGR*).
2. **Biến động Địa giới Hành chính:** Yêu cầu chuyển đổi và liên kết giữa cơ cấu 63 tỉnh/thành phố cũ (Niên giám Thống kê 2025) và cơ cấu 34 tỉnh/thành phố mới sau hợp nhất (Báo cáo tháng 10/2026, Nghị quyết 202/2025/QH15) thông qua bảng ánh xạ `admin_mapping`.
3. **Quy chuẩn Định dạng Bảng biểu và Số liệu:** Xử lý dấu phẩy thập phân kiểu Việt Nam, dấu chấm phân cách hàng nghìn, các ký hiệu quy ước khuyết số liệu (`..`, `-`, `x`), và sự chênh lệch thứ nguyên đơn vị đo lường (*nghìn người* so với *người*, *tỷ đồng* so với *triệu đồng*).
4. **Chất lượng Số hóa Dữ liệu:** Mức độ toàn vẹn của 32 bảng số liệu đã số hóa lưu trong `veristat.db` so với bản in PDF và tệp Excel gốc nhiều trang.

---

## 4. Bộ Chỉ số và Tiêu chí Đánh giá (Metrics)

Toàn bộ các chỉ số được định lượng hóa và tính toán tường minh trên các tập kiểm chuẩn:

| Chỉ số (Metric) | Công thức & Định nghĩa Tường minh | Tiêu chí Đạt / Mục tiêu |
|---|---|---|
| **Độ chính xác Số học (Numerical Accuracy)** | Tỷ lệ câu hỏi dạng số đạt sai số tương đối $E_r = \frac{\|y_{\text{dự đoán}} - y_{\text{chuẩn}}\|}{\|y_{\text{chuẩn}}\|} \times 100\% < 0,1\%$. Trường hợp $y_{\text{chuẩn}} = 0$, áp dụng sai số tuyệt đối $\|y_{\text{dự đoán}} - y_{\text{chuẩn}}\| < 10^{-4}$. | Chỉ số tối ưu hóa cốt lõi |
| **Độ chính xác Thực thể (Entity Accuracy)** | Tỷ lệ khớp chính xác chuỗi tên địa phương/ngành kinh tế so với đáp án chuẩn sau khi đã chuẩn hóa qua `admin_mapping`. | $100\%$ trên nhóm câu xếp hạng |
| **Độ chính xác Trích dẫn (Citation Accuracy)** | Tỷ lệ câu trả lời có toàn bộ `table_id` trích dẫn tồn tại trong `tables_catalog` và thực sự hỗ trợ tính toán đáp số. | $\ge 95\%$ |
| **Tỷ lệ Vi phạm An toàn (Sandbox Violation Rate)** | Tỷ số giữa số lần mã nguồn cố ý vi phạm chính sách sandbox (import cấm, ghi đĩa, truy cập mạng, vượt trần RAM/CPU) trên tổng số lượt yêu cầu thực thi. | **Mục tiêu thiết kế: 0,0%** |
| **Tỷ lệ Ra đáp số Cuối (Final Answer Rate)** | Tỷ lệ câu hỏi hoàn thành chuỗi suy luận và đưa ra cấu trúc đáp án cuối cùng (`finish`) trong giới hạn $\le 8$ bước. | $\ge 95\%$ |
| **Chi phí Token (Token Cost)** | Số lượng token đầu vào (prompt tokens), token đầu ra (completion tokens) và tổng token trung bình trên mỗi câu hỏi. | Đánh giá trên đường cong compute-curve |
| **Độ trễ Thực thi (Latency)** | Thời gian chạy trung bình (tính bằng giây) từ khi nhận câu hỏi đến khi trả kết quả cuối cùng. | Đo lường riêng cho từng phương pháp |

---

## 5. Dữ liệu Đánh giá (Evaluation Data)

- **Tập Hiệu chỉnh (`data/benchmark/dev_30.jsonl`):** Gồm 30 câu hỏi phân tầng (8 tăng trưởng, 8 tỷ trọng, 7 xếp hạng, 7 tương quan; 12 dễ, 8 trung bình, 10 khó). Dành riêng cho việc tinh chỉnh prompt mẫu và hiệu chuẩn ngưỡng Verifier $\tau$.
- **Tập Kiểm thử Benchmark (`data/benchmark/benchmark_400.jsonl`):** Gồm 400 câu hỏi độc lập phủ đủ 5 lĩnh vực kinh tế – xã hội:
  - 100 câu Tăng trưởng (YoY, CAGR, chênh lệch tuyệt đối và tương đối)
  - 100 câu Tỷ trọng (cơ cấu phần trăm, biến động tỷ trọng qua các năm)
  - 100 câu Xếp hạng (top-$k$, giá trị cực đại/cực tiểu, chỉ tiêu dẫn xuất)
  - 100 câu Tương quan (hệ số tương quan Pearson $r$, Spearman)
- **Ràng buộc Liêm chính Dữ liệu:** Không có bất kỳ sự rò rỉ dữ liệu nào giữa tập `dev_30` và `benchmark_400`. Nghiêm cấm sử dụng tập `benchmark_400` để điều chỉnh prompt hoặc siêu tham số lặp đi lặp lại.

---

## 6. Dữ liệu Huấn luyện & Tiền xử lý (Training Data)

- **Mô hình Nền và Tinh chỉnh Chỉ thị:** Dữ liệu huấn luyện của `Qwen2.5-7B-Instruct` thuộc quyền sở hữu và công bố của Alibaba Cloud, dựa trên tập ngữ liệu đa ngôn ngữ quy mô lớn kết hợp tinh chỉnh có giám sát (SFT) và căn chỉnh trực tiếp (DPO/PPO).
- **Nguồn tài liệu tham chiếu:** Trích xuất từ Báo cáo Kỹ thuật chính thức của Qwen. Nhóm VeriStat không đưa ra bất kỳ ước lượng phỏng đoán nào về tỷ lệ văn bản tiếng Việt hoặc tỷ lệ số liệu GSO trong tập huấn luyện gốc của Alibaba.
- **Phạm vi của VeriStat:** **Nhóm nghiên cứu không thực hiện bất kỳ thao tác fine-tune hoặc cập nhật trọng số nào trên mô hình.** Mô hình nền được giữ cố định nguyên bản làm chính sách suy luận (frozen policy).

---

## 7. Phân tích Thực nghiệm Định lượng (Quantitative Analyses)

*(Ghi chú: Toàn bộ bảng kết quả định lượng dưới đây sẽ được Thành viên 3 - Sơn cập nhật trực tiếp sau khi hoàn tất đợt chạy thực nghiệm test-time compute trên GPU Google Colab Pro. Tuyệt đối không điền trước các số liệu giả định chưa qua kiểm chứng.)*

| Phương pháp Suy luận | Độ chính xác Số học (%) | Độ chính xác Thực thể (%) | Độ chính xác Trích dẫn (%) | Tỷ lệ Ra Đáp số (%) | Token Trung bình / Câu | Độ trễ Trung bình (s) |
|---|---|---|---|---|---|---|
| **1. Lời nhắc Đơn (Single Prompt - Baseline)** | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* |
| **2. Vòng lặp ReAct ($n=1$)** | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* |
| **3. Bỏ phiếu Đa số Self-Consistency ($n=5$)** | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* |
| **4. Best-of-$n$ + Bộ Kiểm chứng (Verifier)** | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* |
| **5. Tìm kiếm Cây MCTS-UCT** | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* | *[Chờ cập nhật]* |

---

## 8. Khía cạnh Đạo đức & An toàn Hệ thống (Ethical Considerations)

1. **Tính Minh bạch & Truy vết Nguồn gốc:** Mỗi câu trả lời bắt buộc phải trích dẫn mã định danh bảng `table_id` tương ứng từ Niên giám Thống kê GSO, cho phép đối chiếu trực tiếp với tài liệu gốc đã công bố công khai.
2. **Kiểm soát Hiện tượng Ảo giác (Hallucination):** Không công nhận suy luận thuần túy của mô hình làm đáp số cuối cùng; đáp số bắt buộc phải được tính toán thông qua việc thực thi mã Python trên bảng dữ liệu SQLite đã được thẩm định.
3. **Bảo vệ Quyền riêng tư & Tuyệt đối không có PII:** Toàn bộ kho dữ liệu là số liệu vĩ mô tổng hợp cấp quốc gia, vùng, tỉnh và ngành kinh tế, hoàn toàn không chứa dữ liệu định danh cá nhân (PII).
4. **An toàn Thực thi Mã nguồn:** Sandbox thực thi mã Python ở mức tiến trình con độc lập của hệ điều hành (không kết nối mạng, môi trường sạch `env={}`, thư mục tạm cô lập, giới hạn cứng RAM 512MB và CPU time 5s). Sandbox chỉ chịu trách nhiệm bảo đảm an toàn hệ thống, còn tính đúng đắn của phép toán do bộ kiểm chứng Hybrid Verifier thẩm định.

---

## 9. Khuyến nghị & Giới hạn Vận hành (Caveats & Recommendations)

- **Bắt buộc Tra cứu Ánh xạ Địa giới:** Khi đối chiếu số liệu theo chuỗi thời gian hoặc so sánh giữa các năm trước và sau 2025, bắt buộc phải truy vấn bảng `admin_mapping` để xử lý sáp nhập địa giới (ví dụ: Thừa Thiên Huế $\to$ Thành phố Huế, hoặc 63 tỉnh $\to$ 34 tỉnh), tránh lỗi cộng trùng hoặc sai lệch mẫu số.
- **Khuyến nghị Vận hành Cache:** Luôn kích hoạt bộ nhớ đệm SQLite (`data/processed/llm_cache.db`) trong suốt quá trình chạy thực nghiệm trên Colab Pro để tái sử dụng các lượt gọi giống nhau, tiết kiệm tối đa compute units.
- **Tách biệt Checkpoint và Trạng thái:** Tiến độ đánh giá benchmark được lưu trữ qua tệp JSONL (`results_<method>.jsonl`) sử dụng bộ khóa tổng hợp `(dataset_version, method, config_hash, question_id)` nhằm chống mất dữ liệu khi ngắt phiên Colab; cơ chế này hoàn toàn độc lập với việc lưu trạng thái nội bộ của từng câu hỏi.
- **Công bố Chi tiết Phần cứng:** Mọi bảng kết quả và báo cáo khoa học phải ghi rõ cấu hình phần cứng thực tế (L4 vs A100), phiên bản driver NVIDIA và phiên bản vLLM được sử dụng.
