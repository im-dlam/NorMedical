# NorMedical (RaumaMedical q3AI)

Hệ thống hỗ trợ tư vấn lâm sàng đa lượt và tra cứu Dược thư Quốc gia Việt Nam vận hành hoàn toàn cục bộ (On-Premise), xây dựng theo kiến trúc **SOLID**, kết hợp mô hình ngôn ngữ lớn tinh chỉnh (**Qwen3-VL-4B-Instruct QLoRA / GGUF**) và công cụ truy xuất tri thức **BM25 RAG**.

---

## 1. Kiến trúc hệ thống (SOLID)

Hệ thống được tách lớp rõ ràng để dễ bảo trì, kiểm thử và thay thế thành phần (ví dụ: chuyển từ LM Studio sang Ollama/vLLM, hoặc thay BM25 bằng FAISS Hybrid Search mà không cần sửa tầng nghiệp vụ):

```text
q3AI/
├── config.json                     # Cấu hình tập trung: API, decoding, RAG, prompt, từ điển chuẩn hóa
├── keys.example.json               # File mẫu cấu hình khóa API (copy thành keys.json khi chạy)
├── requirements.txt                # Danh sách thư viện Python cần thiết
├── main.py                         # Composition Root (Dependency Injection) cho CLI & Gradio
├── web/
│   ├── app.py                      # Máy chủ FastAPI SSE Streaming (cổng mặc định 8386)
│   └── static/
│       └── index.html              # Giao diện cổng tư vấn lâm sàng 2 cột (render Markdown offline)
├── src/
│   ├── config/
│   │   └── settings.py             # Trình nạp cấu hình dạng frozen dataclass & resolve đường dẫn
│   ├── domain/
│   │   ├── models.py               # KnowledgeDocument, ChatMessage, ConsultationTurn
│   │   └── interfaces.py           # ITextNormalizer, IRagRetriever, ILLMClient
│   ├── preprocessing/
│   │   └── normalizer.py           # Chuẩn hóa lỗi chính tả tiếng Việt, nhận diện ý định & mở rộng từ khóa thuốc
│   ├── rag/
│   │   ├── bm25_retriever.py       # Công cụ truy xuất BM25 + Bigram + tiền tố hoạt chất ASCII
│   │   └── data/
│   │       ├── drugs_knowledge_base.jsonl     # 1.754 chuyên luận Dược thư Quốc gia Việt Nam
│   │       └── diseases_knowledge_base.jsonl  # 673 bài bệnh học lâm sàng Vinmec
│   ├── llm/
│   │   └── lmstudio_client.py      # Client OpenAI-compatible SSE Streaming kết nối LM Studio
│   ├── services/
│   │   └── consultation_service.py # Bộ điều phối hội thoại đa lượt, định tuyến 4 chế độ & RAG
│   └── ui/
│       ├── cli.py                  # Giao diện dòng lệnh (Terminal Streaming)
│       └── web_gradio.py           # Giao diện Gradio 2 cột (tùy chọn)
└── tests/
    └── test_edge_cases.py          # Bộ kiểm thử tự động 12 tình huống đặc biệt
```

---

## 2. Cơ chế hoạt động chính

### 2.1. Bộ định tuyến 4 chế độ lâm sàng
Mỗi tin nhắn đầu vào được chuẩn hóa (xử lý lỗi gõ thiếu dấu phổ biến như `nước đa -> nước đá`, `đau học -> đau họng`) và định tuyến vào 1 trong 4 chế độ trong `ConsultationService`:

1. **Cờ đỏ cấp cứu (Red Flag):** Phát hiện triệu chứng nguy hiểm (*đau thắt ngực, khó thở dữ dội, nôn/ho ra máu, co giật, ngất xỉu, liệt nửa người...*) để cảnh báo gọi cấp cứu 115 ngay lập tức.
2. **Tra cứu Dược lý (Drug Lookup):** Khi người dùng hỏi trực tiếp về thuốc hoặc hoạt chất, hệ thống chuyển sang `system_drug_lookup`, mở rộng từ đồng nghĩa triệu chứng/biệt dược phổ biến (*Panadol -> Paracetamol, Smecta -> Diosmectit, Oresol -> ORS...*) và trích xuất tối đa 3 chuyên luận Dược thư (1.500 ký tự/tài liệu) để trình bày nhóm dược lý, công dụng, liều dùng tham khảo và chống chỉ định.
3. **Khai thác triệu chứng đa lượt (Symptom Probing):** Hỏi từng câu một theo trình tự lâm sàng (tính chất/thời gian -> triệu chứng đi kèm -> yếu tố kích phát) thay vì kết luận vội ở lượt đầu.
4. **Kết luận & Miễn trừ y khoa (Conclusion):** Tự động dừng hỏi khi đủ dữ kiện hoặc khi người dùng trả lời phủ định ngắn (`không`, `không á`), tổng hợp nguyên nhân khả dĩ, hướng chăm sóc an toàn và tự động gắn tuyên bố miễn trừ trách nhiệm y khoa ở cuối các lượt có tư vấn thuốc/điều trị.

### 2.2. Kho tri thức RAG cục bộ (`2.427` tài liệu)
- **Dược thư Quốc gia Việt Nam:** `1.754` chuyên luận hoạt chất.
- **Bệnh học lâm sàng Vinmec:** `673` bài bệnh học.
- Chạy hoàn toàn trên RAM CPU bằng **BM25 tùy biến** (Unigram + Bigram + lọc từ dừng `query_stopwords` + loại trừ danh sách từ đời thường trùng tên hoạt chất `common_false_drugs`), không chiếm VRAM GPU.

---

## 3. Yêu cầu & Cài đặt

### Bước 1: Cài đặt thư viện Python
```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình khóa API (`keys.json`)
Tạo file `keys.json` từ file mẫu `keys.example.json`:
```bash
cp keys.example.json keys.json
```
*(File `keys.json` đã được đưa vào `.gitignore` để không lộ khóa khi đẩy mã nguồn lên Git.)*

### Bước 3: Bật máy chủ suy luận cục bộ trên LM Studio
1. Mở phần mềm **LM Studio** và nạp mô hình GGUF (`qwen3-vl-4b-medical` hoặc tương đương).
2. Đặt **Context Length = `4096`** để đảm bảo đủ bộ nhớ cho các chuyên luận Dược thư.
3. Chuyển sang mục **Developer** (`<-->`) và bật **Start Server** (cổng mặc định `http://localhost:1234/v1/chat/completions`, có thể thay đổi trong `config.json`).

---

## 4. Hướng dẫn chạy hệ thống

### 4.1. Chạy giao diện Web FastAPI (Khuyến nghị — cổng mặc định `8386`)
Di chuyển vào thư mục `web/` và chạy `uvicorn`:
```bash
cd web
uvicorn app:app --port 8386
```
Sau đó truy cập trên trình duyệt tại địa chỉ: `http://127.0.0.1:8386`

- `GET /` — Giao diện cổng tư vấn lâm sàng 2 cột (hỗ trợ Streaming SSE, trạng thái `Thinking...`, hiển thị nguồn RAG song song).
- `POST /api/chat` — Endpoint SSE Streaming nhận `{"message": "..."}`.
- `POST /api/reset` — Khởi tạo phiên khám mới.
- `GET /api/health` — Kiểm tra trạng thái máy chủ và số lượng tài liệu RAG đã nạp.

### 4.2. Chạy trên dòng lệnh (CLI Streaming)
```bash
python main.py --mode cli
```
Gõ `reset` để làm mới bệnh án, gõ `exit` để thoát.

### 4.3. Chạy bộ kiểm thử tự động (12 tình huống đặc biệt)
```bash
python tests/test_edge_cases.py
```

---

## 5. Miễn trừ trách nhiệm y khoa

Hệ thống NorMedical (RaumaMedical) được xây dựng nhằm mục đích nghiên cứu học thuật và hỗ trợ tra cứu thông tin tham khảo theo Dược thư Quốc gia Việt Nam. Các nội dung phản hồi không phải là đơn thuốc và không thay thế cho việc thăm khám, chẩn đoán hoặc chỉ định điều trị trực tiếp của bác sĩ chuyên khoa.
