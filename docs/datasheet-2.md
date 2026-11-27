# Bộ dữ liệu 430 câu hỏi phân tích có đáp số kiểm chứng

## Motivation
- Cung cấp một bộ câu hỏi - đáp nhằm phân tích định lượng tiếng Việt có đáp số chuẩn kiểm chứng được bằng mã thực thi độc lập, dùng để đánh giá và so sánh công bằng các phương pháp suy luận có sử dụng công cụ của LLM (Single Prompt, ReAct, Self-Consistency, Best-of-n + Verifier, MCTS-UCT) dưới cùng ngân sách token trên cơ sở dữ liệu thống kê chính thức
- Đưa ra kết quả kiểm chứng dựa trên số liệu mới nhất, sau khi thay đổi địa giới hành chính thực tế (sát nhập từ 63 tỉnh sang 34 tỉnh)

## Composition
- Mỗi instance là một đối tượng JSON biểu diễn một bài toán phân tích thống kê tiếng Việt gồm 11 trường có cấu trúc: question_id, question, split, category, difficulty, answer_type, ground_truth, unit, required_table_ids, reference_code, uses_admin_mapping
- Có tổng cộng 430 câu hỏi (30 câu hỏi cho việc tinh chỉnh tham số, và 400 câu hỏi cho việc kiểm tra) được chia làm 3 nhóm câu hỏi: dạng tăng trưởng, tỷ trọng, xếp hạng và mối tương quan, mỗi nhóm câu hỏi lại được chia ra làm 3 mức độ dễ (1 bước, 1 bảng), trung bình (2–3 bước, 1–2 bản) và khó (≥ 3 bước / đa bảng / admin_mapping). Mỗi đáp án thuộc 3 kiểu, số, danh sách thực thể, thực thể.
- Các câu hỏi đều 

## Collection process
- Câu hỏi được sinh tự động theo mỗi dạng câu hỏi, độ khó và theo mẫu được cung cấp nhờ lệnh script của Python có thể tái lập được với random seed.
- Được người sinh kiểm chứng ngẫu nhiên kết quả dựa vào pdf gốc, thay đổi câu hỏi sao cho tự nhiên

## Uses
- Đánh giá và so sánh các chiến lược suy luận ở thời điểm suy diễn (Single Prompt, ReAct, Self-Consistency, Best-of-n + Verifier, MCTS) trên bài toán phân tích số liệu tiếng Việt có công cụ.
- Vì câu hỏi được sinh từ bộ mẫu có kiểm soát, đa dạng cú pháp ngôn ngữ tự nhiên có thể hẹp hơn câu hỏi tự do ngoài thực tế; không dùng làm chuẩn đánh giá khả năng hội thoại mở.

## Maintenance
- Bộ dữ liệu này sẽ không hỗ trợ cập nhật, phiên bản này sẽ được giữ trên repository của dự án