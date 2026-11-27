# Bộ dữ liệu trích xuất từ Niên giám và báo cáo kinh tế

## Motivation
- Cung cấp dữ liệu có cấu trúc được số hóa từ niên giám thống kê 2025 và báo cáo do cục Thống kê biên soạn nhằm đánh giá năng lực suy luận định lượng, sử dụng công cụ (tool-use), và khả năng tự kiểm chứng của các tác tử LLM (ReAct, MCTS + Verifier) trên dữ liệu thống kê tiếng Việt thực tế.
- Khắc phục khoảng trống dữ liệu tiếng Việt chuẩn hóa khi dữ liệu thống kê chính thức phát hành dưới dạng báo cáo PDF hoặc Excel nhiều tầng header, không sẵn sàng cho truy vấn SQL hoặc đánh giá tự động tác tử AI.
- Dữ liệu do nhóm 3 xây dựng.

## Composition
- Bộ dữ liệu gồm 32 bảng dữ liệu (tổng 6.968 dòng) thống kê của 5 lĩnh vực: dân số (6 bảng), lao động (6 bảng), doanh nghiệp(8 bảng), giáo dục (6 bảng) và y tế (6 bảng). 
- Bộ dữ liệu gồm 27 bảng từ Niên giám Thống kê 2025 và 5 bảng từ Biểu số liệu tháng 10/2026.
- Bộ dữ liệu không bao phủ hết toàn bộ tiêu chí thống kê và linh vực trong bộ niên giám và báo cáo.
- 6 bảng có giá trị thiếu: NG2025_T331, NG2025_T332, NG2025_T340, NG2025_T339, NG2025_T335, NG2025_T341 do ấn phẩm gốc không công bố.

## Collection process
- Dữ liệu được trích xuất tự động nhờ lệnh script của Python hoàn toàn từ 2 file thống kê chính thức của Cục Thống kê Việt Nam:
1. Niên giám Thống kê Toàn quốc 2025 (pdf, gồm 1.557 trang, gồm 14 chương mục; quét tự động 173 bảng thuộc 4 chương chính và chọn có kiểm định 27 bảng).
2. Biểu số liệu Kinh tế - Xã hội tháng 10 năm 2026 (bảng excel, gồm 20 sheet; chọn 5 sheet trọng tâm thuộc lĩnh vực Lao động và Doanh nghiệp, đã áp dụng cơ cấu 34 tỉnh/thành phố mới).
- Dữ liệu được thu thập từ ngày 22/11, file Biểu số liệu Kinh tế - Xã hội tháng 10 năm 2026 được tham chiếu vào tháng 10 năm 2026 và đăng vào ngày 3 tháng 11, còn file Niên giám thống kê 2025 được đăng vào ngày 30 tháng 6

## Preprocessing/cleaning/labeling
- Chuẩn hóa số thực: Chuyển đổi định dạng dấu phẩy thập phân (ví dụ 102345,thành 102345.3), loại bỏ chú thích (*), loại bỏ các ký hiệu khuyết thiếu (.., ..., -, –, x, Sơ bộ).
- Phân cấp địa lý : Gắn nhãn country, region, province, other cho từng dòng dữ liệu địa phương. 
- Tạo bảng ánh xạ địa phương mới

## Uses
- Bộ dữ liệu này chỉ được xây dựng phục vụ tạo môi trường kiểm chuẩn thực tế cho tác tử LLM suy luận.
- Bộ dữ liệu chỉ được công bố trong phạm vi dự án github này

## Maintenance
- Bộ dữ liệu này sẽ không hỗ trợ cập nhật, phiên bản này sẽ được giữ trên repository của dự án
