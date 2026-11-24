# NorMedical (RaumaMedical q3AI)

Hệ thống hỗ trợ tư vấn lâm sàng đa lượt và tra cứu Dược thư Quốc gia Việt Nam vận hành hoàn toàn cục bộ (On-Premise), xây dựng theo kiến trúc **SOLID**, kết hợp mô hình ngôn ngữ lớn tinh chỉnh (**Qwen3-VL-4B-Instruct QLoRA / GGUF**) và công cụ truy xuất tri thức **BM25 RAG**.

---

## 1. Tổng quan & Cơ sở Nghiên cứu Thực tiễn

Hệ thống được thiết kế từ kết quả khảo sát thực tế trên hai tập mẫu độc lập:

### 1.1. Khảo sát chuyên sâu (`N = 120`: 70% người dân, 20% sinh viên Y-Dược, 10% Bác sĩ/Dược sĩ)
- **81.7%** (98/120) tự tra cứu triệu chứng và tên thuốc trên mạng trước khi đi khám.
- **68.3%** (82/120) từng dùng AI tổng quát (ChatGPT, Gemini) để hỏi về sức khỏe, nhưng chỉ **23.3%** hài lòng về độ an toàn và tin cậy.
- **94.2%** kỳ vọng công cụ dẫn nguồn chính thống từ cơ quan y tế trong nước.
- **3 rào cản lớn của AI tổng quát:** Trả lời chung chung dài dòng (**41.7%**), không nhận diện đúng thuốc/ngữ cảnh Việt Nam (**35.0%**), thiếu tin cậy về liều lượng và an toàn (**23.3%**).
- **3 nỗi đau lớn nhất:** Sợ sai liều / tương tác thuốc (**62.5%**), sợ bỏ sót dấu hiệu cấp cứu (**22.5%**), lo ngại rò rỉ dữ liệu bệnh án lên đám mây (**15.0%**).
- **Ngưỡng thời gian chờ chấp nhận được:** **76.7%** kỳ vọng phản hồi dưới 5 giây (**38.3%** chọn `< 3 giây`, **45.0%** chọn `3 - 5 giây`, chỉ **3.4%** chấp nhận `> 7 giây`).

### 1.2. Khảo sát cộng đồng độc lập (`N = 164`, thang Likert 1-5)
- Mức đánh giá hiện trạng AI y tế nói chung còn thấp: Tốc độ `3.09` (SD `1.45`), Diễn đạt dễ hiểu `2.93` (SD `1.39`), Chi tiết `3.05` (SD `1.47`), Giúp yên tâm `3.10` (SD `1.39`), Hài lòng tổng thể `3.21` (SD `1.35`).
- Lo ngại lớn nhất của cộng đồng: Lộ bệnh sử riêng tư (**29.27%**), câu trả lời chung chung (**25.00%**), không rõ nguồn Bộ Y tế (**18.90%**), sợ dùng nhầm thuốc (**15.85%**).

---

## 2. Dữ liệu, Kỹ thuật Huấn luyện & Tối ưu Suy luận

### 2.1. Làm sạch Dữ liệu & Xây dựng Bộ dữ liệu Y khoa Tiếng Việt
- **Phân tích lỗi dữ liệu thô (`ViHealthQA`):** Khảo sát cho thấy câu hỏi thô thường rất ngắn (trung vị `63` ký tự dạng tiêu đề SEO), trong khi câu trả lời nhắc tới bệnh án ẩn hoặc lặp văn mẫu khuyên đi khám chung chung.
- **Bộ dữ liệu đã chuẩn hóa:**
  - **Tập SFT Đơn lượt sạch:** `7.439` mẫu huấn luyện (`train_sft_clean.csv`) + `500` mẫu kiểm định (`val_sft_clean.csv`).
  - **Tập Hội thoại Đa lượt (Multi-turn SFT):** `8.063` hội thoại đơn và đa lượt (`train_sft_multiturn_messages.jsonl`) tổng hợp và lọc sạch từ `ViHealthQA`, `Vinmec`, `BookingCare`, `Medical-o1-reasoning-vi`.
  - **Kho Tri thức RAG (`2.427` tài liệu chính thống):** `1.754` chuyên luận **Dược thư Quốc gia Việt Nam** (`drugs_knowledge_base.jsonl`) + `673` bài **Bệnh học lâm sàng Vinmec** (`diseases_knowledge_base.jsonl`).

### 2.2. Cấu hình & Kết quả Huấn luyện (Fine-tuning)

| Tham số / Hạng mục | Giá trị Thực tế |
| :--- | :--- |
| **Mô hình nền (Base Model)** | `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` |
| **Kỹ thuật tinh chỉnh** | QLoRA 4-bit (Unsloth) trên GPU Tesla T4 (Google Colab) |
| **Cấu hình LoRA** | `r = 16`, `lora_alpha = 16`, `lora_dropout = 0.05`, `bias = "none"` |
| **Lớp tinh chỉnh** | `finetune_language_layers = True`, đóng băng nhánh Vision (`finetune_vision_layers = False`) |
| **Hàm mất mát (Loss Masking)** | `train_on_responses_only` (chỉ tính loss trên lượt trả lời `<\|im_start\|>assistant`) |
| **Lịch học & Số bước** | `lr = 3e-5`, `cosine` scheduler, `504 steps` (1 full epoch trên 8.062 hội thoại) |
| **Kết quả Hội tụ Loss** | Training Loss giảm mượt từ **`2.454`** xuống **`~1.51 - 1.55`** |
| **Dung lượng Adapter LoRA** | `132 MB` (`adapter_model.safetensors`, trọn bộ 7 file ~`137 MB`) |
| **Định dạng triển khai cục bộ** | GGUF lượng tử hóa `Q4_K_M` (~`2.4 - 3.2 GB VRAM`), vận hành trên Laptop **RTX 3050 (4GB VRAM)** |

### 2.3. Các Phát hiện Kỹ thuật khi Tối ưu Suy luận (Inference Engineering)
1. **Hệ số `LoRA Scaling = 0.25`:** Thực nghiệm cho thấy ở mức mặc định `scaling = 1.0`, trọng số LoRA học từ văn bản Q&A có xu hướng ghi đè tư duy sinh lý bệnh gốc của mô hình (*Catastrophic Forgetting*). Giảm hệ số về `0.25` khôi phục hoàn toàn năng lực suy luận lâm sàng gốc của `Qwen3-4B` đồng thời duy trì văn phong bác sĩ Việt Nam.
2. **Tối ưu giải mã (Decoding) cho Tiếng Việt:** Loại bỏ `no_repeat_ngram_size = 4` (vốn gây hỏng dấu/từ ghép trên tokenizer âm tiết tiếng Việt, ví dụ `"uể oải"` bị biến dạng thành `"uể oại"`). Cấu hình chuẩn hóa trong `config.json`: `temperature = 0.30`, `top_p = 0.88`, `top_k = 40`, `repeat_penalty = 1.12`, `presence_penalty = 0.35`, `frequency_penalty = 0.25`, `max_tokens = 750`.

---

## 3. Kiến trúc Hệ thống (SOLID)

Hệ thống được thiết kế theo nguyên lý **SOLID** (tách biệt cấu hình, khóa bảo mật, domain interface, tầng truy xuất RAG, client LLM và giao diện):

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

## 4. Cơ chế Định tuyến 4 Chế độ & RAG Cục bộ

1. **Cờ đỏ cấp cứu (Red Flag):** Phát hiện triệu chứng nguy hiểm (*đau thắt ngực, khó thở dữ dội, nôn/ho ra máu, co giật, ngất xỉu, liệt nửa người...*) để cảnh báo gọi cấp cứu 115 ngay lập tức.
2. **Tra cứu Dược lý (Drug Lookup):** Dùng riêng `system_drug_lookup`, mở rộng từ đồng nghĩa triệu chứng và biệt dược phổ biến (*Panadol -> Paracetamol, Smecta -> Diosmectit, Oresol -> ORS...*), trích xuất tối đa 3 chuyên luận Dược thư (`1.500` ký tự/tài liệu) để trình bày nhóm dược lý, công dụng, liều dùng tham khảo và chống chỉ định.
3. **Khai thác triệu chứng đa lượt (Symptom Probing):** Tự động sửa lỗi gõ thiếu dấu phổ biến (`nước đa -> nước đá`, `đau học -> đau họng`), hỏi cuốn chiếu 1 câu tự nhiên mỗi lượt theo trình tự lâm sàng.
4. **Kết luận & Miễn trừ y khoa (Conclusion):** Tự động dừng hỏi khi đủ dữ kiện hoặc khi bệnh nhân trả lời phủ định ngắn (`không`, `không á`), tránh vòng lặp của mô hình 4B, và gắn tuyên bố miễn trừ trách nhiệm y khoa ở cuối các lượt có tư vấn thuốc/điều trị.

### Kết quả Kiểm thử Tự động
- **Bộ kiểm thử 12 tình huống đặc biệt (`tests/test_edge_cases.py`):** Đạt **12/12 PASS** (tra cứu thuốc, cờ đỏ cấp cứu, chuẩn hóa thiếu dấu, chống lặp hội thoại đa lượt).
- **Kiểm định truy xuất RAG trên 20 câu hỏi thuốc thực tế:** Đạt **20/20** trích xuất trúng chuyên luận Dược thư Quốc gia.

---

## 5. Ưu điểm, Giới hạn & Lộ trình Phát triển

### Ưu điểm đạt được
- Hỏi bệnh đa lượt tự nhiên, giữ được tư duy y khoa nhờ `LoRA Scaling = 0.25`.
- Tra cứu Dược thư chạy hoàn toàn trên CPU (`< 10 ms`), không chiếm VRAM, hiển thị nguồn đối chứng song song.
- Chạy 100% cục bộ (On-Premise) trên GPU phổ thông `4GB VRAM` (RTX 3050), bảo vệ tuyệt đối dữ liệu bệnh án.
- Tích hợp Streaming SSE trên cả FastAPI Web và CLI, giảm thời gian chờ cảm nhận.

### Giới hạn hiện tại & Hướng nâng cấp
- **Kho biệt dược thương mại:** Hiện phủ `1.754` hoạt chất gốc Dược thư Quốc gia; giai đoạn tiếp theo tích hợp danh mục biệt dược Bộ Y tế (DAV) và so khớp mờ `RapidFuzz`.
- **Truy xuất ngữ nghĩa dài:** Bổ sung tìm kiếm lai Hybrid Search (`BM25 + FAISS Vector`) song song cho các mô tả triệu chứng dân dã.
- **An toàn cứng & Thị giác:** Xây dựng module Python tính liều cứng theo cân nặng (kg), kiểm tra tương tác đa thuốc và kích hoạt nhánh Vision sẵn có của `Qwen3-VL` cho bài toán OCR đơn thuốc / vỏ hộp.

---

## 6. Yêu cầu & Hướng dẫn Chạy

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Tạo file khóa cấu hình
```bash
cp keys.example.json keys.json
```
*(File `keys.json` đã được đưa vào `.gitignore` để không đẩy lên Git.)*

### Bước 3: Bật LM Studio Local Server
1. Mở **LM Studio** và nạp mô hình GGUF (`qwen3-vl-4b-medical` hoặc tương đương).
2. Đặt **Context Length = `4096`**.
3. Vào mục **Developer** (`<-->`) -> bật **Start Server** (mặc định `http://localhost:1234/v1/chat/completions`, có thể cấu hình trong `config.json`).

### Bước 4: Khởi chạy hệ thống
- **Giao diện Web FastAPI (cổng mặc định `8386`):**
  ```bash
  cd web
  uvicorn app:app --port 8386
  ```
  Truy cập trình duyệt tại: `http://127.0.0.1:8386`
  - `GET /` — Giao diện cổng tư vấn lâm sàng 2 cột (Streaming SSE, `Thinking...`, render Markdown offline).
  - `POST /api/chat` — Endpoint SSE Streaming nhận `{"message": "..."}`.
  - `POST /api/reset` — Khởi tạo phiên khám mới.
  - `GET /api/health` — Kiểm tra trạng thái máy chủ và số lượng tài liệu RAG.

- **Giao diện dòng lệnh (CLI Streaming):**
  ```bash
  python main.py --mode cli
  ```

- **Chạy bộ kiểm thử tự động:**
  ```bash
  python tests/test_edge_cases.py
  ```

---

## 7. Tài liệu Tham khảo & Miễn trừ Y khoa

- Bộ Y tế Việt Nam (2022): *Dược thư Quốc gia Việt Nam lần xuất bản thứ ba*, NXB Y học (1.754 chuyên luận hoạt chất).
- Hệ thống Y tế Vinmec: *Cẩm nang Bệnh học Lâm sàng* (673 bài chuyên khoa).
- Qwen Team (Alibaba Cloud, 2025): *Qwen3 & Qwen3-VL Technical Report*.
- Dettmers et al. (NeurIPS 2023): *QLoRA: Efficient Finetuning of Quantized LLMs*.
- Hu et al. (ICLR 2022): *LoRA: Low-Rank Adaptation of Large Language Models*.
- Lewis et al. (NeurIPS 2020): *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- Robertson & Zaragoza (2009): *The Probabilistic Relevance Framework: BM25 and Beyond*.

> **Lưu ý y khoa:** Hệ thống NorMedical (RaumaMedical) được xây dựng cho mục đích nghiên cứu học thuật và hỗ trợ tra cứu thông tin tham khảo theo Dược thư Quốc gia Việt Nam. Các nội dung phản hồi không phải là đơn thuốc và không thay thế cho việc thăm khám, chẩn đoán hoặc chỉ định điều trị trực tiếp của bác sĩ chuyên khoa.
