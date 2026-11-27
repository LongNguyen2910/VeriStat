# Thiết kế Chi tiết & Hợp đồng Giao diện — Phần của Sơn (VeriStat, Đề tài 18)

**Phụ trách:** Sơn (Search, Verifier & Theory Lead)  
**Ngày:** 2026-11-25 (Ngày 4/14)  
**Trạng thái:** Đã chốt thiết kế (Validated) — sẵn sàng chia sẻ với Khoa (Agent & Sandbox Lead) và Long (Data & Benchmark Lead)  
**Tài liệu nền tảng:** `docs/DESIGN.md`

---

## 1. Tóm tắt Hiểu biết (Understanding Summary)

### Phạm vi do Sơn phụ trách chính
- Cài đặt **từ đầu bằng Python thuần + NumPy** (không sử dụng framework agent dựng sẵn như LangChain, LangGraph, LlamaIndex):
  - **Phần A.2:** Self-Consistency (`sc.py`, gồm biến thể bỏ phiếu tuần tự dừng sớm phục vụ Định lý B.5), Hybrid Verifier 2 tầng (`tier1_rules.py`, `tier2_llm.py`, `hybrid.py`), và Best-of-*n* (`best_of_n.py`).
  - **Phần A.3:** Cây tìm kiếm Monte Carlo với quy tắc UCT (`mcts.py`).
  - **Phần A.4 & D.3–D.4:** Khung đánh giá tác tử (`metrics.py`, `runner.py`, `scripts/run_eval.py`), tái lập thống kê và kịch bản vẽ biểu đồ compute-curve (`scripts/plot_results.py`).
- **Phần B (25đ):** Hoàn thiện 5 chứng minh lý thuyết chặt chẽ (B.1 POMDP & Bayes; B.2 cận Hoeffding nhị phân & điều kiện đa lớp; B.3 Best-of-*n* với Verifier không hoàn hảo và giới hạn $n \to \infty$; B.4 cận $O(\ln N(s))$ của UCT từ UCB1; B.5 tương đương đầu ra & kỳ vọng số lời gọi của bỏ phiếu tuần tự dừng sớm).
- **Phần C (C.1–C.3):** Giải chi tiết và viết script Python kiểm chứng số học độc lập (`scripts/verify_theory_c.py`) cho 3 bài tập C.1, C.2, C.3.

### Hạng mục phối hợp chung với nhóm
- Rà soát chéo 80 câu hỏi mẫu phân tầng trong `data/benchmark/crosscheck_sample.csv` (điền cột `reviewer_son_ok` và `review_note`).
- Chốt Hợp đồng Giao diện (`src/veristat/agents/types.py`) với Khoa và tích hợp với `StepEnvironment` / `LLMClient` trên Colab Pro (Ngày 5–6).
- Tham gia gán nhãn mù 50 vết thực thi thật (C.4) theo 6 mã lỗi trong `docs/error_taxonomy_guide.md`, đạt độ đồng thuận Cohen's $\kappa \ge 0.75$ và hòa giải.
- Viết phần nội dung của Sơn trong báo cáo LaTeX (`report/`, 25–40 trang toàn nhóm) gồm A.2–A.4, toàn bộ B.1–B.5, C.1–C.3, và D.3–D.4.
- Cập nhật mục `### Sơn` trong `docs/AI_USAGE_LOG.md`, kiểm tra tái lập trên môi trường sạch và chuẩn bị bảo vệ vấn đáp (Ngày 13–14).

### Ngoài phạm vi (Non-goals)
- Không fine-tune mô hình LLM hay mô hình Verifier/PRM riêng.
- Không xây dựng ứng dụng web thương mại hoàn chỉnh.
- Không chỉnh sửa dữ liệu hay bộ benchmark đã kiểm định của Long (chỉ ghi chú và báo cáo nếu phát hiện sai sót khi rà soát chéo).

### Hợp đồng dữ liệu có sẵn từ phần của Long (Tái sử dụng trực tiếp)
1. **`veristat.data.schema`:** `BenchmarkQuestion`, `AnswerType` (`"number" | "entity" | "entity_list"`), `open_readonly` (mở SQLite ở chế độ `mode=ro`).
2. **`veristat.data.admin_mapping`:** `normalize_entity`, `normalize_entity_list` (chuẩn hóa tên địa phương cho gom cụm Self-Consistency và chấm điểm).
3. **`veristat.data.benchmark_gen`:** `check_answer_match(pred, gt, answer_type, rel_tol=1e-3)` (sai số tương đối $< 0{,}1\%$ cho `number`, khớp chuẩn hóa cho `entity` / `entity_list`).
4. **`veristat.data.c4_taxonomy` & `scripts/prepare_c4_traces.py`:** `results/eval_traces.jsonl` do khung đánh giá xuất ra phải chứa đủ các trường `question_id`, `method`, `category`, `difficulty`, `is_correct`, `predicted_answer`, `ground_truth`, `trace`.

---

## 2. Giả định Tường minh & Yêu cầu Phi chức năng (Assumptions & NFRs)

1. **Ngân sách tính toán (1 tài khoản Google Colab Pro):**
   - Toàn bộ lệnh gọi LLM/Verifier đi qua client hỗ trợ batching/async, cache request vào SQLite và ghi checkpoint JSONL theo `question_id` để khôi phục ngay khi ngắt phiên.
   - Dùng cơ chế **bể quỹ đạo dùng chung $N=12$ quỹ đạo ReAct/câu** và lấy mẫu con ngoại tuyến theo 5 hạt giống (`42..46`) cho ReAct, Self-Consistency và Best-of-*n* nhằm báo cáo `trung bình ± độ lệch chuẩn` mà không nhân $5\times$ chi phí GPU.
2. **Tách tập nghiêm ngặt (`dev_30` vs `benchmark_400`):**
   - Ngưỡng Verifier $\tau^\star$ (chế độ nhị phân) và 3 mức `token_budget` cho MCTS-UCT **chỉ được hiệu chỉnh trên `data/benchmark/dev_30.jsonl`**.
   - `data/benchmark/benchmark_400.jsonl` chỉ được dùng cho bước chạy đánh giá cuối cùng.
3. **An toàn & Tính tái lập:**
   - Mã của Sơn không bao giờ `exec`/`eval` mã Python do LLM sinh ra; mọi thực thi mã đi qua `StepEnvironment.execute_action` trong sandbox cô lập không mạng của Khoa.
   - Chỉ truy vấn `data/processed/veristat.db` bằng `open_readonly` (`mode=ro`).
   - Cố định random seed ở mọi bước lấy mẫu, gom cụm, chọn MCTS và bootstrap 1.000 lần.
4. **Tách biệt module & Khả năng bảo trì:**
   - Tách rõ giao diện giữa Khoa (`LLMClient`, `StepEnvironment`) và Sơn (`sc`, `verifier`, `best_of_n`, `mcts`, `eval`) thông qua `src/veristat/agents/types.py`.
   - Toàn bộ unit test của Sơn chạy độc lập trên CPU bằng `MockLLMClient` và `MockStepEnvironment` trong dưới 5 giây.
5. **Điểm lưu ý hiện trạng kho mã nguồn cần phối hợp xử lý:**
   - `data/benchmark/crosscheck_sample.csv` hiện chưa có xác nhận của cả 3 thành viên.
   - `pyproject.toml` cần bổ sung `matplotlib` (phục vụ `scripts/plot_results.py`) và thư viện client gọi vLLM (ví dụ `openai` hoặc `httpx`).
   - Cần nhắn Long chỉnh vài lỗi đánh máy nhỏ trước khi nộp: `docs/datasheet-2.md` dòng 9–10 ("3 nhóm câu hỏi" → 4 nhóm, câu bỏ dở `"Các câu hỏi đều "`) và chú thích dòng 1 của `docs/table_selection.yaml` ("31 bảng" → 32 bảng).
   - Lĩnh vực `giao_duc` chỉ có 8 tham chiếu bảng trong `benchmark_400` và 1 trong `dev_30`: cần giữ tầng `giao_duc` trong tập con phân tầng 80 câu cho phần lặp 5 seed của MCTS.

---

## 3. Nhật ký Quyết định (Decision Log)

| # | Quyết định | Các phương án đã xét | Lý do chọn |
|---|---|---|---|
| **D1** | Chủ động đề xuất Interface Contract cho Khoa và triển khai thiết kế A.2–A.4 ngay từ Ngày 4. | (A) Chủ động đề xuất contract + thiết kế ngay; (B) Chờ code của Khoa; (C) Ưu tiên Phần B trước. | Đã là Ngày 4/14 nhưng `main` chưa có code của Khoa; chốt contract sớm giúp hai thành viên làm song song không bị nghẽn. |
| **D2** | Chọn **Phương án 1: Step-level Protocol + Offline Pool Replay**. | PA1 (Step-level Protocol dùng chung + Replay ngoại tuyến); PA2 (Black-box ReAct + MCTS tự viết prompt/parser riêng); PA3 (Rollout MCTS đầy đủ + Verifier chỉ dùng `logprobs`). | Đảm bảo ReAct, SC, BoN và MCTS dùng chung 100% prompt, công cụ và parser (so sánh compute-curve công bằng), không trùng lặp logic và phù hợp ngân sách 1 Colab Pro. |
| **D3** | Chốt `src/veristat/agents/types.py` gồm 5 dataclass bất biến và 2 Protocol (`LLMClient`, `StepEnvironment`), bổ sung `queried_tables` và `computed_values` vào `StepObservation`. | Có hoặc không truyền cấu trúc `queried_tables` và `computed_values` từ tool về observer. | Giúp Verifier Tầng 1 kiểm tra trích dẫn bảng và đối chiếu chống bịa số liệu chính xác, không phụ thuộc regex đoán lại chuỗi output thô. |
| **D4** | Chốt thiết kế **Self-Consistency** (gom cụm số $<0{,}1\%$ lấy median, chuẩn hóa thực thể qua `admin_mapping`, phá hòa xuất hiện trước, dừng sớm $m = \lfloor n/2 \rfloor + 1$ theo B.5), **Hybrid Verifier 2 tầng** (4 luật Tầng 1 chặn gọi LLM Tầng 2; Tầng 2 JSON rubric 3 tiêu chí `0.4 / 0.4 / 0.2` kèm fallback `0.5` khi lỗi JSON và `logprobs` tùy chọn), và **Best-of-*n*** xuất đồng thời `argmax` + ngưỡng $\tau$. | Verifier chỉ dùng `logprobs` vs JSON rubric; Best-of-*n* chỉ chạy `argmax` vs xuất song song `argmax` và $\tau$. | Giảm FPR trên lỗi công thức/địa giới, xuất trực tiếp số liệu $(\text{TPR}, \text{FPR})$ cho B.3 & C.2 và ghi nhận `reason` hỗ trợ C.4 mà không tốn thêm lượt gọi LLM. |
| **D5** | Chốt thiết kế **MCTS-UCT** ($d_{\max}=6$, $k=2$, $c=\sqrt{2}$, khử trùng `action_key`, cờ `exhausted`, cắt tỉa ngay khi `safety_violation`, nút trung gian dùng heuristic Tầng 1, nút kết thúc dùng Hybrid Verifier, trần `token_budget` cứng hiệu chỉnh trên `dev_30` theo $n \in \{3, 5, 8\}$). | Rollout LLM đầy đủ ở mỗi nút lá vs heuristic Tầng 1 ở nút trung gian; giới hạn mềm vs trần token cứng. | Tiết kiệm 30–50% ngân sách token so với rollout đầy đủ, chống kẹt chọn lại nhánh kết thúc và bảo đảm công bằng ngân sách token. |
| **D6** | Chốt **Khung đánh giá A.4** tách 2 pha (Colab checkpoint JSONL $\to$ CPU phân tích 5 seed trên pool $N=12$ + bootstrap 1.000 lần cho MCTS 400 + 5 seed trên tập con 80 câu + xuất `eval_traces.jsonl` chuẩn C.4), bộ test CPU dùng Mock và `scripts/verify_theory_c.py` cho C.1–C.3. | Chạy lại GPU 5 lần độc lập vs dùng pool $N=12$ + replay ngoại tuyến; chạy 5 seed MCTS full 400 câu vs 1 lần 400 câu (bootstrap) + 5 seed tập con 80 câu. | Kiểm soát rủi ro hết compute units trên 1 Colab Pro mà vẫn bảo đảm đầy đủ kiểm định thống kê (`mean ± std` và 95% CI). |

---

## 4. Hợp đồng Giao diện Khoa ↔ Sơn (`src/veristat/agents/types.py`)

Phần dưới đây là đặc tả Python chuẩn dùng làm hợp đồng tích hợp trực tiếp giữa phần của **Khoa** (`tools/`, `llm/`, `agents/react.py`) và phần của **Sơn** (`agents/sc.py`, `agents/best_of_n.py`, `agents/mcts.py`, `verifier/`, `eval/`):

```python
"""Shared immutable data contracts and protocols between Khoa (Tools/LLM/ReAct)
and Sơn (SC/Verifier/Best-of-n/MCTS/Eval).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from veristat.data.schema import BenchmarkQuestion

ToolName = Literal["search_docs", "query_table", "run_python", "final_answer"]
TerminatedReason = Literal[
    "final_answer",
    "max_steps",
    "timeout",
    "token_budget",
    "safety_violation",
    "error",
]


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by `LLMClient.generate`."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    logprobs: dict[str, float] | None = None
    cached: bool = False


@dataclass(frozen=True)
class ToolAction:
    """Single Thought + Action parsed from one LLM step."""

    thought: str
    tool_name: ToolName
    tool_args: dict[str, Any]
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass(frozen=True)
class StepObservation:
    """Structured observation produced by executing one `ToolAction`."""

    output: Any
    ok: bool
    error: str | None = None
    safety_violation: bool = False
    queried_tables: tuple[str, ...] = ()
    computed_values: tuple[Any, ...] = ()
    latency_s: float = 0.0


@dataclass(frozen=True)
class ReActStep:
    """One Thought -> Action -> Observation transition in a trajectory."""

    step_idx: int
    action: ToolAction
    observation: StepObservation | None = None


@dataclass(frozen=True)
class FinalAnswer:
    """Structured final output extracted from a `final_answer` action."""

    answer: float | str | list[str] | None
    unit: str
    cited_tables: tuple[str, ...]
    computation_code: str
    parse_ok: bool


@dataclass(frozen=True)
class Trajectory:
    """Complete execution trace for a single question."""

    question_id: str
    steps: tuple[ReActStep, ...]
    final: FinalAnswer | None
    terminated_reason: TerminatedReason
    total_prompt_tokens: int
    total_completion_tokens: int
    total_latency_s: float
    seed: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def tool_calls_count(self) -> int:
        return sum(1 for s in self.steps if s.action.tool_name != "final_answer")

    @property
    def has_safety_violation(self) -> bool:
        return self.terminated_reason == "safety_violation" or any(
            s.observation is not None and s.observation.safety_violation
            for s in self.steps
        )

    @property
    def all_queried_tables(self) -> tuple[str, ...]:
        seen: list[str] = []
        for step in self.steps:
            if step.observation is None:
                continue
            for tid in step.observation.queried_tables:
                if tid not in seen:
                    seen.append(tid)
        return tuple(seen)


class LLMClient(Protocol):
    """OpenAI-compatible / vLLM client (with SQLite cache) implemented by Khoa."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        logprobs: bool = False,
    ) -> LLMResponse:
        ...


class StepEnvironment(Protocol):
    """Step-level ReAct + 3-tool environment shared by ReAct, SC, BoN, and MCTS."""

    def propose_actions(
        self,
        question: BenchmarkQuestion,
        history: tuple[ReActStep, ...],
        *,
        k: int = 1,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> list[ToolAction]:
        ...

    def execute_action(
        self,
        action: ToolAction,
        history: tuple[ReActStep, ...] = (),
    ) -> StepObservation:
        ...

    def extract_final_answer(self, action: ToolAction) -> FinalAnswer:
        ...

    def run_react(
        self,
        question: BenchmarkQuestion,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_steps: int = 8,
        max_seconds: float = 60.0,
        max_tokens: int | None = None,
    ) -> Trajectory:
        ...
```

### Quy ước trách nhiệm giữa Khoa và Sơn
- **Khoa cài đặt:** `LLMClient` (OpenAI-compatible/vLLM + SQLite cache) và `StepEnvironment` bọc 3 công cụ (`search_docs`, `query_table`, `run_python` sandbox cô lập). Trong `execute_action`:
  - Khi gọi `query_table`: tự điền `queried_tables` chứa danh sách `table_id` thực sự được truy vấn (ánh xạ từ `sql_table` trong `tables_catalog`).
  - Khi gọi `run_python`: tự điền `computed_values` chứa các giá trị số/thực thể xuất ra từ sandbox (và đặt `safety_violation=True` nếu AST/rlimit/mạng bị chặn).
  - Hàm `run_react` chỉ cần khoảng ~30 dòng ghép vòng lặp gọi `propose_actions(k=1)` $\to$ `execute_action` $\to$ `extract_final_answer` khi `tool_name == "final_answer"` hoặc dừng khi đạt `max_steps=8`, `max_seconds=60.0`, `max_tokens` hoặc `safety_violation`.
- **Sơn cài đặt:** Dùng trực tiếp `StepEnvironment` cho `sc.py`, `best_of_n.py`, `mcts.py`, `eval/` và dùng `LLMClient` cho `tier2_llm.py`, cùng bộ `MockLLMClient` / `MockStepEnvironment` phục vụ unit test CPU.

---

## 5. Thiết kế Chi tiết các Module của Sơn

### 5.1. Self-Consistency & Dừng sớm (`src/veristat/agents/sc.py`)
- **Lọc quỹ đạo hợp lệ:** Bỏ qua các quỹ đạo có `final is None`, `not final.parse_ok`, `final.answer is None`, hoặc `has_safety_violation`. Nếu tất cả đều không hợp lệ, trả kết quả rỗng và ghi nhận thất bại.
- **Gom cụm đáp án (Clustering):**
  - `answer_type == "number"`: Gom cụm các số có sai số tương đối $< 0{,}1\%$ (tái sử dụng `check_answer_match(..., rel_tol=1e-3)`, xử lý giá trị `0.0` với `abs <= 1e-6`). Giá trị đại diện của cụm thắng cuộc là **median** của các phần tử trong cụm.
  - `answer_type == "entity"`: Chuẩn hóa bằng `normalize_entity` trước khi đếm phiếu.
  - `answer_type == "entity_list"`: Chuẩn hóa từng phần tử theo thứ tự bằng `normalize_entity_list` trước khi đếm phiếu.
  - **Phá hòa (Tie-breaking):** Tất định theo cụm xuất hiện sớm hơn trong chuỗi quỹ đạo.
- **Chiến lược bỏ phiếu tuần tự dừng sớm (Định lý B.5):**
  - Với $n \in \{3, 5, 8\}$, ngưỡng quá bán là $m = \lfloor n / 2 \rfloor + 1$ (tương ứng $2, 3, 5$).
  - Hỗ trợ cả chế độ sinh trực tiếp từ `StepEnvironment` lẫn chế độ **offline replay** trên danh sách $n$ quỹ đạo lấy mẫu từ bể $N=12$: đọc tuần tự từng quỹ đạo, dừng ngay khi một cụm đạt $\ge m$ phiếu (hoặc chạm `token_budget`), ghi nhận `calls_used`, `calls_saved = n - calls_used`, và `tokens_saved`.

### 5.2. Hybrid Verifier 2 tầng (`src/veristat/verifier/`)
- **`tier1_rules.py` (Tầng 1 tất định — 0 token LLM):**
  - Kiểm tra 4 điều kiện cứng trên một `Trajectory`:
    1. Không có vi phạm an toàn (`not traj.has_safety_violation`).
    2. Có ít nhất 1 bước `query_table` hoặc `run_python` thực thi thành công (`obs.ok`).
    3. `final` hợp lệ (`parse_ok=True`), `cited_tables` không rỗng, mọi `table_id` trong `cited_tables` tồn tại trong `tables_catalog` (`open_readonly`) và đều nằm trong `traj.all_queried_tables`.
    4. Giá trị `final.answer` khớp với ít nhất một kết quả trong `computed_values` (từ `run_python`, hoặc từ `query_table` nếu là câu đọc trực tiếp) nhằm chặn LLM tự bịa số liệu ở bước cuối.
  - Nếu trượt bất kỳ điều kiện nào $\to$ `tier1_passed = False`, `score = 0.0`, **dừng ngay không gọi LLM Tầng 2**.
  - Cung cấp thêm hàm `score_intermediate_state(steps) -> float` $\in [0, 1]$ cho nút trung gian MCTS: thưởng khi `search_docs` / `query_table` / `run_python` thành công và có dữ liệu khác rỗng; phạt khi lỗi cú pháp SQL/Python, kết quả rỗng, lặp lại hành động giống hệt bước trước, hoặc vi phạm an toàn.
- **`tier2_llm.py` (Tầng 2 LLM-as-a-Verifier):**
  - Chỉ chạy khi `tier1_passed == True`.
  - Prompt nhận: `(câu hỏi, danh sách bảng & tóm tắt observation đã truy vấn, computation_code, answer, unit)`.
  - Yêu cầu đầu ra JSON gồm 3 tiêu chí (mỗi tiêu chí nhận điểm trong $\{0.0, 0.5, 1.0\}$) và giải thích ngắn:
    - `table_and_geo` (trọng số `0.4`): đúng bảng nguồn, đúng năm, đúng lọc `geo_level` (`country`/`region`/`province`) và ánh xạ sáp nhập 63 $\leftrightarrow$ 34 tỉnh nếu có.
    - `formula` (trọng số `0.4`): đúng công thức thống kê (tăng trưởng, CAGR, tỷ trọng, tương quan Pearson/Spearman, xếp hạng).
    - `unit_and_type` (trọng số `0.2`): khớp đơn vị tính và định dạng `answer_type`.
    - `reason`: lý do ngắn gọn phục vụ phân tích lỗi C.4.
  - Hỗ trợ đọc tùy chọn `logprobs` làm tín hiệu phụ khi cần; nếu JSON lỗi cú pháp sau regex fallback $\to$ gán điểm bảo thủ `0.5` kèm cờ `parse_fallback=True` để không làm gãy luồng eval.
- **`hybrid.py`:** Kết hợp Tầng 1 + Tầng 2 thành `VerificationResult(score, tier1_passed, tier1_reasons, tier2_scores, reason, verifier_prompt_tokens, verifier_completion_tokens, latency_s)`.

### 5.3. Best-of-*n* (`src/veristat/agents/best_of_n.py`)
- Chạy `HybridVerifier` đúng 1 lần trên $n$ quỹ đạo ứng viên ($n \in \{3, 5, 8\}$), cộng toàn bộ `verifier_prompt_tokens + verifier_completion_tokens` vào tổng token tiêu thụ.
- Xuất đồng thời kết quả cho 2 quy tắc chọn từ cùng 1 lần chấm điểm:
  1. **`argmax` liên tục (chính cho Phần D):** Chọn quỹ đạo có `score` cao nhất (phá hòa theo quỹ đạo xuất hiện sớm hơn).
  2. **Ngưỡng nhị phân $\tau$ (đối chiếu B.3 & C.2):** Chọn các ứng viên có `score >= tau` ($\tau^\star$ hiệu chỉnh duy nhất trên `dev_30`); ghi nhận cả hai biến thể khi 0 ứng viên đạt ngưỡng (fallback ngẫu nhiên đều trong $n$ ứng viên và từ chối trả lời `abstain`) để tính thực nghiệm $(\text{TPR}, \text{FPR})$.

### 5.4. MCTS với quy tắc UCT (`src/veristat/agents/mcts.py`)
- **Cấu trúc nút `MCTSNode`:** Lưu `parent`, `children`, `step: ReActStep | None`, `depth` ($0 \le d \le d_{\max}=6$), `visits` ($N$), `value_sum` ($W$, $\bar{Q} = W / N$), `is_terminal`, `terminal_score`, `exhausted`, và `action_key` (chuẩn hóa `tool_name` + tham số để chống sinh nhánh trùng lặp).
- **4 pha chuẩn MCTS (Python thuần + NumPy):**
  1. **Selection:** Bắt đầu từ gốc, chọn con chưa `exhausted` tối đa hóa UCT:
     $$\text{UCT}(s, a) = \bar{Q}(s, a) + c\sqrt{\frac{\ln N(s)}{N(s, a)}}, \quad c = \sqrt{2}$$
     Ưu tiên tuyệt đối nút con chưa thăm ($N(s,a)=0$).
  2. **Expansion:** Tại lá chưa kết thúc và $d < d_{\max}$, gọi `env.propose_actions(question, history, k=2, temperature=0.7, seed=...)`. Khử trùng lặp theo `action_key` với các con hiện có, rồi gọi `env.execute_action` cho mỗi ứng viên mới. Nhánh nào có `safety_violation=True` được cắt tỉa ngay (`is_terminal=True`, `terminal_score=0.0`, `exhausted=True`).
  3. **Evaluation:**
     - **Nút trung gian** ($d < d_{\max}$, chưa gọi `final_answer`): chấm bằng heuristic tất định Tầng 1 `score_intermediate_state` (**0 token LLM**, tiết kiệm 30–50% ngân sách token).
     - **Nút kết thúc** (`final_answer` hoặc chạm $d_{\max}$): tạo `Trajectory` từ đường dẫn gốc $\to$ lá và chấm đầy đủ bằng `HybridVerifier` (cộng đủ token Verifier vào tổng chi phí).
  4. **Backpropagation:** Cập nhật `visits += 1` và `value_sum += v` dọc đường về gốc; lan truyền `exhausted = True` lên cha nếu toàn bộ con của cha đã `exhausted`.
- **Kiểm soát ngân sách & Kết thúc:**
  - Trước mỗi vòng lặp mở rộng, kiểm tra trần cứng `tokens_used >= token_budget` (tính gộp prompt + completion + verifier), `iterations >= max_iterations`, hoặc `elapsed_s >= max_seconds`.
  - Ba mức `token_budget` được hiệu chỉnh **chỉ trên `dev_30`** sao cho tổng token thực chi trung bình của MCTS khớp với 3 mốc $n \in \{3, 5, 8\}$ của SC/Best-of-*n*.
  - Chọn nút kết thúc có `terminal_score` cao nhất (phá hòa theo `visits` cao hơn). Nếu chạm trần token trước khi có nút kết thúc hợp lệ, trả về `terminated_reason="token_budget"` với kết quả tốt nhất hiện có (`final=None` nếu chưa có).

### 5.5. Khung Đánh giá & Biểu đồ (`src/veristat/eval/`, `scripts/run_eval.py`, `scripts/plot_results.py`)
- **Kiến trúc 2 pha:**
  - *Pha 1 — Thu thập thực nghiệm (Colab Pro, checkpoint JSONL theo `question_id`):*
    - Chạy Single Prompt baseline (`results/single_{split}.jsonl`).
    - Sinh bể $N=12$ quỹ đạo ReAct/câu (`temperature=0.7`) và chấm sẵn Hybrid Verifier cho cả 12 quỹ đạo (`results/pool_12_{split}.jsonl`).
    - Chạy MCTS-UCT tại 3 mức `token_budget` (`results/mcts_{budget_name}_{split}.jsonl`).
  - *Pha 2 — Tổng hợp thống kê ngoại tuyến (CPU, 0 tốn GPU):*
    - Lấy mẫu con theo **5 hạt giống (`42..46`)** từ `pool_12` để tính `mean ± std` cho ReAct ($n=1$), SC ($n \in \{3,5,8\}$, bản đủ và bản dừng sớm B.5), và Best-of-*n* ($n \in \{3,5,8\}$, `argmax` và ngưỡng $\tau^\star$).
    - Cho MCTS-UCT trên 400 câu: tính khoảng tin cậy **bootstrap 95% (1.000 lần lấy mẫu lại)** và chạy bổ sung lặp 5 seed trên tập con phân tầng 80 câu (giữ nguyên tỷ lệ 4 nhóm × 3 độ khó và đảm bảo có đủ câu thuộc lĩnh vực `giao_duc`).
- **Độ đo xuất ra `results/eval_summary.json`:** Độ chính xác tổng & phân rã (theo `category`, `difficulty`, `domain`, `uses_admin_mapping`), tỷ lệ trích dẫn đúng `required_table_ids` (exact & set match), số lời gọi công cụ TB, tổng token TB (prompt + completion + verifier), độ trễ (giây/câu), tỷ lệ vi phạm an toàn, $(\text{TPR}, \text{FPR})$ của Verifier, và số lời gọi tiết kiệm của SC dừng sớm.
- **Xuất vết lỗi cho C.4 (`results/eval_traces.jsonl`):** Lưu mỗi dòng một bản ghi chứa đủ `question_id`, `method`, `category`, `difficulty`, `is_correct`, `predicted_answer`, `ground_truth`, `trace` để truyền trực tiếp vào `scripts/prepare_c4_traces.py`.
- **Biểu đồ (`scripts/plot_results.py`):**
  1. Đường cong **Accuracy vs. Tổng token trung bình** so sánh cả 5 phương pháp dưới cùng ngân sách token.
  2. Biểu đồ cột phân rã Accuracy theo độ khó (`easy`, `medium`, `hard`) và theo nhóm câu hỏi (`growth`, `share`, `ranking`, `correlation`).
  3. Đồ thị đối chiếu thực nghiệm với lý thuyết: SC thực nghiệm vs cận B.2, Best-of-*n* theo $\tau^\star$ vs đường cong B.3, tỷ lệ gọi tiết kiệm của dừng sớm vs B.5.

### 5.6. Chiến lược Kiểm thử CPU (`tests/agents/`, `tests/verifier/`, `tests/eval/`)
- Xây dựng `MockLLMClient` và `MockStepEnvironment` tất định theo seed để chạy toàn bộ suite kiểm thử cục bộ trên CPU (< 5 giây):
  - `test_sc_bon.py`: gom cụm số `<0.1%`, xử lý biên `gt=0.0`, chuẩn hóa tên tỉnh qua `admin_mapping`, chứng minh dừng sớm B.5 cho đáp án trùng khớp tuyệt đối với chạy đủ $n$, Best-of-*n* chọn đúng cả `argmax` lẫn $\tau$.
  - `test_verifier.py`: kiểm tra đủ 4 luật Tầng 1 và xác minh Tầng 2 không bị gọi khi Tầng 1 trượt; kiểm tra parse JSON lỗi không crash và cộng đủ token verifier.
  - `test_mcts.py`: kiểm tra công thức UCT trên cây đồ chơi chính xác tới $10^{-9}$, cập nhật backpropagation đúng, cắt tỉa nhánh `safety_violation`, không chọn lại nhánh `exhausted`, và dừng đúng tại `token_budget`.
  - `test_eval_harness.py`: chạy end-to-end giả lập trên `dev_30`, kiểm tra checkpoint/resume không trùng `question_id`, và kiểm tra file `eval_traces.jsonl` sinh ra nạp thành công qua `sample_traces_for_c4`.

### 5.7. Lý thuyết Phần B & Bài tập C.1–C.3 (`scripts/verify_theory_c.py`, `report/sections/`)
- **Phần B (`report/sections/part_b_theory.tex`):** Trình bày đầy đủ giả thiết, định nghĩa ký hiệu thống nhất và 5 chứng minh B.1–B.5.
- **Phần C (`scripts/verify_theory_c.py` & `report/sections/part_c_exercises.tex`):**
  - **C.1:** Tính chính xác phân phối nhị thức cho $p=0{,}6$, $n \in \{1, 5, 15\}$ và đối chiếu cận Hoeffding từ B.2.
  - **C.2:** Tính xác suất chọn đúng theo B.3 với $p=0{,}3$, $\text{TPR}=0{,}9$, $\text{FPR}=0{,}1$ cho $n \in \{1, 4, 16\}$ theo cả 2 quy ước (fallback ngẫu nhiên đều và từ chối), chứng minh giới hạn $n \to \infty$ hội tụ về $\frac{0{,}27}{0{,}34} \approx 0{,}7941$, kèm phân tích giới hạn test-time compute.
  - **C.3:** Cài đặt hàm tính tay từng bước UCT ($c=\sqrt{2}$) cho 2 vòng MCTS kèm bảng mẫu trình bày LaTeX sẵn sàng điền cây quyết định do giảng viên cung cấp.

---

## 6. Lộ trình Thực hiện Chi tiết cho Sơn (Ngày 4 → Ngày 14)

### Bước 0 — Thiết lập môi trường & Vòng kiểm định nền (Ngày 4)
```powershell
uv sync
uv run python scripts/build_db.py --selection docs/table_selection.yaml --out data/processed/veristat.db --from-csv data/interim/tables --no-csv
uv run python scripts/check_data_quality.py --db data/processed/veristat.db --report docs/data_quality_report.json
uv run python scripts/verify_benchmark.py --db data/processed/veristat.db --dev data/benchmark/dev_30.jsonl --test data/benchmark/benchmark_400.jsonl
uv run pytest -v
```

### Lịch trình các giai đoạn tiếp theo

| Giai đoạn | Thời gian | Nội dung công việc chính | Tiêu chí hoàn thành |
|---|---|---|---|
| **Giai đoạn 1: Contract & Lõi A.2–A.4 trên CPU** | Ngày 4 – Ngày 5 | Gửi Section 4 (`types.py`) cho Khoa chốt; rà soát chéo 80 câu `crosscheck_sample.csv`; cài đặt `sc.py`, `verifier/`, `best_of_n.py`, `mcts.py`, `eval/`, `scripts/run_eval.py`, `scripts/plot_results.py` + Mock tests. | Pass 100% `uv run pytest tests/agents tests/verifier tests/eval -v` trên CPU; chạy thử eval giả lập trên `dev_30` + resume thành công; 80/80 dòng có `reviewer_son_ok`. |
| **Giai đoạn 2: Tích hợp Khoa & Thực nghiệm Colab Pro** | Ngày 6 – Ngày 8 | Ghép với `StepEnvironment` và `LLMClient` của Khoa; hiệu chỉnh prompt Tầng 2, $\tau^\star$ và 3 mức ngân sách MCTS trên `dev_30`; chạy `benchmark_400` (Single Prompt, bể $N=12$, MCTS 3 mức ngân sách); vẽ đồ thị compute-curve. | Đủ 400/400 câu có checkpoint; bảng so sánh 5 phương pháp (`mean ± std` 5 seed & bootstrap 95% CI); `results/eval_traces.jsonl` sẵn sàng cho C.4. |
| **Giai đoạn 3: Lý thuyết B, Bài tập C & Báo cáo LaTeX** | Ngày 8 – Ngày 12 | Viết 5 chứng minh B.1–B.5 và chạy `scripts/verify_theory_c.py` cho C.1–C.3 (bắt đầu ngay trong lúc chờ GPU Colab); cùng Long/Khoa gán nhãn 50 vết C.4 ($\kappa \ge 0{,}75$); viết các mục của Sơn trong `report/` và cập nhật `docs/AI_USAGE_LOG.md`. | 100% số liệu C.1–C.3 khớp script kiểm chứng; C.4 đạt $\kappa \ge 0{,}75$; LaTeX biên dịch PDF sạch, số liệu khớp tuyệt đối giữa bảng/hình/text. |
| **Giai đoạn 4: Kiểm tra tái lập & Bảo vệ vấn đáp** | Ngày 13 – Ngày 14 | Chạy tái lập toàn bộ pipeline trên môi trường sạch từ 32 CSV $\to$ `veristat.db` $\to$ `pytest` $\to$ bảng & đồ thị từ kết quả cache; tập vấn đáp chéo cả 3 thành viên. | Pipeline tái lập chạy thông suốt không lỗi; cả 3 thành viên nắm vững kiến trúc và chứng minh. |
