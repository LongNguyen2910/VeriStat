# Thiết kế Kỹ thuật Phân hệ Tác tử, Công cụ, Sandbox & LLM Client (Khoa - VeriStat)

**Dự án:** VeriStat — Đề tài 18 (Tác tử Phân tích Số liệu Thống kê Kinh tế – Xã hội Việt Nam)  
**Tác giả thiết kế:** Khoa (Agent & Sandbox Lead)  
**Người tích hợp kế thừa:** Sơn (Search, Verifier & Theory Lead)  
**Ngày phê duyệt:** 2026-11-25  
**Trạng thái:** ĐÃ KHÓA THIẾT KẾ (Final Validated Design)

---

## 1. Biên bản Khóa Hiểu biết (Understanding Summary)

- **Mục tiêu xây dựng:**
  1. **Bộ 3 công cụ:** `query_table` (SQL read-only trên SQLite `veristat.db`), `search_docs` (FTS5 BM25 search), `run_python` (sandbox cô lập đa tầng ở mức tiến trình OS).
  2. **Vòng lặp ReAct thuần:** Tự cài đặt từ đầu bằng Python + NumPy (không dùng LangChain/LangGraph/LlamaIndex), kiến trúc State-Centric với `AgentState` bất biến, hỗ trợ `clone()` và chuyển trạng thái đơn bước `step()` phục vụ MCTS-UCT.
  3. **Lớp LLM Client & Cache 2 tầng:** Kết nối `vLLM` (OpenAI-compatible) trên Colab Pro (GPU L4/A100) và `MockLLM` trên CPU local; SQLite request cache băm canonical JSON; TaskCheckpoint JSONL khôi phục theo bộ khóa tổng hợp `task_key`.
  4. **Tài liệu `docs/MODEL_CARD.md`:** Soạn thảo khung chuẩn 9 mục của Mitchell et al. (2019) cho `Qwen/Qwen2.5-7B-Instruct`.
- **Mục đích tồn tại:** Đóng vai trò động cơ suy luận và tính toán số học có kiểm chứng cho Đề tài 18, loại bỏ ảo giác số liệu bằng cách bắt buộc LLM phải chạy code thực nghiệm trên dữ liệu thống kê thật của GSO.
- **Ràng buộc cốt lõi:**
  - Không sử dụng framework agent dựng sẵn.
  - Phân định rạch ròi 3 trụ cột của `run_python`: (1) AST Linting (lọc tĩnh), (2) Resource Rationing (cắt CPU $\le 5$s, RAM $\le 512$MB), (3) OS-level Isolation (cô lập tiến trình, không mạng, môi trường `env={}` sạch, thư mục tạm cô lập).
  - Tương thích song song kiểm thử cục bộ (Windows CPU) và thực nghiệm GPU (Colab Pro Linux).
  - Giới hạn cứng ReAct: $\le 8$ bước, thời gian $\le 60$s/câu; sai số số học mục tiêu $< 0,1\%$.
- **Mục tiêu loại trừ (Non-goals):** Không fine-tune mô hình; không crawl web trực tiếp khi suy luận; không xây web UI thương mại.

---

## 2. Các Giả định Kỹ thuật (Assumptions)

1. CSDL `veristat.db` (do Long số hóa) sẵn sàng với 32 bảng thống kê, bảng siêu dữ liệu `tables_catalog`, bảng FTS5 `docs_fts`, và bảng ánh xạ địa giới `admin_mapping`.
2. Mô hình phục vụ thực nghiệm chính là `Qwen/Qwen2.5-7B-Instruct` (chạy BF16/FP16 trên GPU L4/A100 qua vLLM).
3. JSON Schema là hợp đồng hình dạng dữ liệu cho công cụ, không thay thế kiểm tra nghiệp vụ và ranh giới bảo mật.
4. SQLite request cache lưu tại `data/processed/llm_cache.db`; TaskCheckpoint lưu tại `data/benchmark/results_<method>.jsonl`.
5. JSONL checkpoint bị đứt đoạn dòng cuối do ngắt phiên Colab phải được loader tự động bỏ qua dòng lỗi, không coi là câu hoàn tất.

---

## 3. Nhật ký Quyết định (Decision Log)

| # | Hạng mục | Quyết định đã chọn | Các phương án đã cân nhắc | Lý do lựa chọn |
|---|---|---|---|---|
| 1 | **Kiến trúc Tác tử & Trạng thái** | Hướng 1: State-Centric với `AgentState` bất biến/cloneable, chuyển trạng thái qua `step(state)` | Hướng 2: Monolithic Agent (dict tự do); Hướng 3: IPC Sandbox Daemon | Bảo đảm tính phân nhánh độc lập cho cây tìm kiếm MCTS-UCT của Sơn mà không làm biến dạng trạng thái gốc. |
| 2 | **Cơ chế An toàn Sandbox** | Phân định 3 trụ cột: (1) AST Linting tĩnh, (2) Điều tiết tài nguyên, (3) Cô lập OS tiến trình con | Thuần AST; Docker Container bắt buộc | AST chỉ là heuristic lọc sớm; ranh giới bảo mật cốt lõi phải nằm ở OS (không mạng, filesystem tạm, timeout/rlimit). Docker quá nặng và khó chạy trên Colab. |
| 3 | **Định dạng Hành động & Xử lý lỗi** | Ưu tiên JSON Schema chuẩn + Regex/Markdown fallback; lỗi công cụ nạp vào `Observation` | Strict JSON không sửa lỗi; One-shot fail immediately | Tối ưu tính bền bỉ, giúp mô hình tự nhận diện lỗi (SQL sai cú pháp, chia cho 0) và tự sửa sai trong tối đa 8 bước. |
| 4 | **Lớp LLM Client & Cache** | Client tách biệt + SQLite cache canonical JSON + JSONL TaskCheckpoint có khóa tổng hợp | File-based cache; In-memory cache dump | Tái sử dụng tối đa token khi chạy lại thực nghiệm, chống mất dữ liệu khi ngắt phiên Colab Pro. |
| 5 | **Đặc tả Model Card** | Chuẩn 9 mục Mitchell et al. (2019) tách bạch thông tin nhà phát hành với kết quả đo lường thật | Tự điền số liệu giả định; Viết gộp so sánh nhiều model | Đảm bảo tính liêm chính học thuật và độ chính xác khoa học cho báo cáo Đề tài 18. |

---

## 4. Thiết kế Chi tiết Hệ thống (Detailed Design)

### 4.1. Cấu trúc Mô-đun Mã nguồn (`src/veristat/`)
```text
src/veristat/
├── tools/
│   ├── base.py              # BaseTool, ToolResult, ToolRegistry
│   ├── table_tool.py        # QueryTableTool (SQLite mode=ro, LIMIT 200, unit metadata)
│   ├── docs_tool.py         # SearchDocsTool (FTS5 BM25 search)
│   └── sandbox/
│       ├── ast_guard.py     # AST static safety analyzer (linting cấm import/exec)
│       └── runner.py        # Subprocess OS-level sandbox & resource limits (Linux/Win)
├── agents/
│   ├── contracts.py         # VerificationResult, ToolCall, AgentAction
│   ├── state.py             # AgentState, StepRecord (serializable, deep cloneable)
│   ├── parser.py            # JSON schema parser + Regex fallback
│   ├── prompts.py           # System prompts & ReAct templates
│   └── react.py             # VeriStatReActAgent (step, run_trajectory, run)
├── llm/
│   ├── base.py              # LLMClient (interface), LLMResponse, LLMMessage
│   ├── vllm_client.py       # VLLMClient (OpenAI-compatible HTTP, conditional retry)
│   ├── mock_client.py       # MockLLMClient (deterministic pattern/fixture matching)
│   ├── cache.py             # SQLiteLLMCache (canonical JSON SHA-256 key)
│   └── checkpoint.py        # TaskCheckpoint (compound task_key JSONL resume manager)
└── cli.py                   # CLI chạy thử nghiệm đơn câu hỏi hoặc batch
```

### 4.2. Đặc tả Bộ 3 Công cụ & Sandbox Cô lập

#### 1. Giao diện Chung `ToolResult`
```python
@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    success: bool
    output: str                   # Dữ liệu bảng (CSV/Markdown) hoặc stdout tính toán
    error: str | None = None      # Thông điệp lỗi chi tiết
    error_type: str | None = None # "SQL_SYNTAX", "AST_FORBIDDEN", "RESOURCE_LIMIT", "SECURITY_VIOLATION", "RUNTIME_ERROR"
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### 2. `query_table(sql: str) -> ToolResult`
- **URI kết nối:** `file:veristat.db?mode=ro`.
- **Kiểm soát câu lệnh:** Chặn tất cả từ khóa DDL/DML (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `PRAGMA`).
- **Phân trang an toàn:** Tự động ép `LIMIT 200`.
- **Làm giàu siêu dữ liệu:** Gắn `table_id`, `unit`, `scope` vào `metadata` của `ToolResult`.

#### 3. `search_docs(query: str, top_k: int = 5) -> ToolResult`
- **Truy vấn:** FTS5 BM25 trên bảng ảo `docs_fts`.
- **Đầu ra:** Định dạng bảng tóm tắt Markdown gồm `table_id`, tiêu đề, cột, đơn vị, kỳ báo cáo và đoạn trích giải thích.

#### 4. `run_python(code: str) -> ToolResult` (Ba trụ cột độc lập)
- **Trụ cột 1 — AST Linting tĩnh (`ast_guard.py`):** Lọc sớm mã độc, cấm node `Import`/`ImportFrom` (`socket`, `requests`, `urllib`, `http`, `subprocess`, `os`, `shutil`, `sys`, `pathlib`), cấm `open` ghi file, cấm `eval`/`exec`.
- **Trụ cột 2 — Điều tiết tài nguyên (`runner.py`):**
  - Linux/Colab: `resource.setrlimit(RLIMIT_CPU, (5, 5))` và `resource.setrlimit(RLIMIT_AS, (512MB, 512MB))`.
  - Windows: Giám sát bộ nhớ tiến trình con + timeout cứng `subprocess.run(timeout=10)`.
- **Trụ cột 3 — Cô lập bảo mật OS (`runner.py`):** Tiến trình con độc lập `python -I -E -s`, môi trường rỗng `env={}`, thư mục làm việc là `TemporaryDirectory` độc lập, chặn socket ở cấp runtime.
- **Phân loại lỗi:** Vi phạm chính sách trả về `error_type="SECURITY_VIOLATION"` hoặc `"RESOURCE_LIMIT"`.

---

### 4.3. Đặc tả Trạng thái Tác tử (`AgentState`) & Vòng lặp ReAct

#### 1. Hợp đồng Trạng thái (`agents/state.py` & `agents/contracts.py`)
```python
@dataclass
class StepRecord:
    step_idx: int
    thought: str
    action_name: str | None
    action_input: dict[str, Any] | None
    observation: str
    tool_error: str | None = None
    execution_time: float = 0.0

@dataclass
class AgentState:
    question_id: str
    question: str
    current_step: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    cited_tables: set[str] = field(default_factory=set)
    is_terminal: bool = False
    final_answer: dict[str, Any] | None = None  # {"value": Any, "answer_type": str, "unit": str, "computation_code": str}
    status: str = "RUNNING"                     # "RUNNING", "SUCCESS", "MAX_STEPS_REACHED", "SAFETY_HALT"

    def clone(self) -> AgentState:
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

#### 2. Nguyên tử hóa Vòng lặp ReAct (`agents/react.py`)
- `step(state: AgentState) -> AgentState`: Chuyển trạng thái đơn bước. Sinh prompt từ lịch sử `state.steps`, gọi LLM, bóc tách JSON action (fallback regex), thực thi công cụ hoặc kết thúc, ghi nhận `StepRecord` mới và trả về trạng thái kế tiếp.
- `run(question: str, max_steps: int = 8) -> AgentState`: Thực hiện vòng lặp cho đến khi `is_terminal` hoặc đạt `max_steps`.
- **Tự sửa sai (Self-correction):** Lỗi công cụ được nạp có cấu trúc vào `Observation: [ErrorType] <detail>` để mô hình tự sửa trong các bước còn lại.

---

### 4.4. Đặc tả Lớp LLM Client, Cache & Khôi phục Tiến trình

#### 1. Hợp đồng Client (`llm/base.py`)
```python
@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    cached: bool = False
    model: str = ""
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        ...
```

#### 2. VLLMClient & Thử lại có Điều kiện (`llm/vllm_client.py`)
- Cấu hình qua `VLLMConfig(base_url, model, timeout_seconds=60.0, max_retries=3)`.
- **Chính sách retry:** Chỉ retry tối đa 3 lần với exponential backoff và jitter đối với HTTP 429, 5xx, lỗi kết nối mạng hoặc timeout. Tuyệt đối **không retry** lỗi 400, 401, 403 hoặc payload sai quy cách.

#### 3. SQLiteLLMCache với Canonical JSON Key (`llm/cache.py`)
- **Tệp CSDL:** `data/processed/llm_cache.db`.
- **Khóa băm SHA-256 Canonical:**
  ```python
  cache_payload = {
      "model": model,
      "messages": messages,
      "temperature": temperature,
      "seed": seed,
      "max_tokens": max_tokens,
      "top_p": top_p,
      "tools": tool_schemas,
      "prompt_version": prompt_version,
  }
  canonical = json.dumps(cache_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
  cache_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
  ```
- **Quy tắc:** Chỉ cache các response thành công và hợp lệ; dùng transaction và `busy_timeout=5000` chống khóa tệp SQLite.

#### 4. TaskCheckpoint & Resume trên Colab Pro (`llm/checkpoint.py`)
- **Tệp lưu vết:** `data/benchmark/results_<method>.jsonl`.
- **Khóa tổng hợp định danh tác vụ:** `task_key = (dataset_version, method, config_hash, question_id)`.
- **Xử lý sự cố ghi dở:** Loader tự động bỏ qua dòng cuối nếu bị lỗi JSON parse do đứt nguồn Colab, chỉ nạp các bản ghi `SUCCESS` vào `completed_ids`. Ghi dữ liệu bằng `f.write(line); f.flush(); os.fsync(f.fileno())`.

---

### 4.5. Đặc tả Khung Tài liệu `docs/MODEL_CARD.md`
- Tuân thủ 9 mục chuẩn của Mitchell et al. (2019) cho mô hình `Qwen/Qwen2.5-7B-Instruct`.
- Phân biệt rõ context length mặc định (32K) với khả năng mở rộng (128K).
- Định nghĩa công thức tường minh cho các độ đo:
  - **Numerical Accuracy:** $E_r = \frac{|y_{\text{pred}} - y_{\text{true}}|}{|y_{\text{true}}|} \times 100\% < 0,1\%$ (quy định riêng sai số tuyệt đối khi $y_{\text{true}} = 0$).
  - **Entity Accuracy:** Khớp chính xác chuỗi chuẩn hóa qua `admin_mapping`.
  - **Citation Accuracy:** $100\%$ `table_id` trích dẫn phải nằm trong danh mục hỗ trợ kết luận.
  - **Sandbox Violation Rate:** Số vi phạm / tổng số lượt chạy (mục tiêu $0\%$).
- Phân tách rõ ràng: Sandbox là kiểm soát thực thi mã, không phải bộ lọc loại bỏ ảo giác toán học. Chỉ điền kết quả thực nghiệm sau khi chạy benchmark thật trên Colab Pro.

---

## 5. Lộ trình Triển khai 5 Bước (Implementation Roadmap)

1. **Bước 1 (Foundation Contracts & Tools Core):**
   - Triển khai `agents/contracts.py`, `agents/state.py` và `tools/base.py` (`BaseTool`, `ToolResult`, `ToolRegistry`).
   - Viết unit tests cho `AgentState.clone()`, serialization và branch isolation.
2. **Bước 2 (Metadata & Data Access Tools):**
   - Khoa phối hợp cùng Long xác nhận schema 32 bảng trong `veristat.db`.
   - Triển khai `tools/table_tool.py` (`query_table`) và `tools/docs_tool.py` (`search_docs`).
3. **Bước 3 (Isolated Python Sandbox):**
   - Triển khai `tools/sandbox/ast_guard.py` và `tools/sandbox/runner.py`.
   - Viết bộ unit test riêng biệt kiểm chứng trên Windows (local) và Linux (giả lập / Colab).
4. **Bước 4 (LLM Client, Cache & ReAct Agent):**
   - Triển khai `llm/base.py`, `llm/vllm_client.py`, `llm/mock_client.py`, `llm/cache.py` và `llm/checkpoint.py`.
   - Triển khai `agents/parser.py`, `agents/prompts.py` và `agents/react.py`.
   - Viết 8 ca kiểm thử biên (cache hit/miss, canonical key, conditional retry, resume JSONL hỏng dòng cuối).
5. **Bước 5 (Tài liệu Model Card & Bàn giao Handoff):**
   - Khởi tạo tệp `docs/MODEL_CARD.md` chuẩn 9 mục để Sơn điền kết quả benchmark.
   - Bàn giao `ToolRegistry` và `VeriStatReActAgent` cho Sơn triển khai Self-Consistency, Verifier và MCTS-UCT.
