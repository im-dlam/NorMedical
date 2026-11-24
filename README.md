# NorMedical

**Hệ thống Hỗ trợ Tư vấn Lâm sàng Đa lượt & Tra cứu Dược thư Quốc gia Việt Nam vận hành Cục bộ (On-Premise)**

<p>
  <a href="https://huggingface.co/imdlam/q3medical"><img src="https://img.shields.io/badge/HuggingFace-imdlam%2Fq3medical-0f3d5e?style=flat-square" height="18" alt="HuggingFace" /></a>
  <a href="https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"><img src="https://img.shields.io/badge/Base-Qwen3--VL--4B-1e40af?style=flat-square" height="18" alt="Base Model" /></a>
  <a href="https://huggingface.co/imdlam/q3medical"><img src="https://img.shields.io/badge/GGUF-Q4__K__M-0e7490?style=flat-square" height="18" alt="GGUF" /></a>
  <a href="#6-hướng-dẫn-cài-đặt--khởi-chạy"><img src="https://img.shields.io/badge/FastAPI-SSE-15803d?style=flat-square" height="18" alt="FastAPI" /></a>
  <a href="#3-kho-tri-thức--cơ-chế-truy-xuất-rag"><img src="https://img.shields.io/badge/RAG-2%2C427%20docs-334155?style=flat-square" height="18" alt="RAG" /></a>
</p>

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Cơ sở Nghiên cứu & Khảo sát Thực tế](#2-cơ-sở-nghiên-cứu--khảo-sát-thực-tế)
3. [Kho Tri thức & Cơ chế Truy xuất RAG](#3-kho-tri-thức--cơ-chế-truy-xuất-rag)
4. [Huấn luyện Mô hình & Tối ưu Suy luận](#4-huấn-luyện-mô-hình--tối-ưu-suy-luận)
5. [Kiến trúc Hệ thống & Cấu trúc Mã nguồn](#5-kiến-trúc-hệ-thống--cấu-trúc-mã-nguồn)
6. [Hướng dẫn Cài đặt & Khởi chạy](#6-hướng-dẫn-cài-đặt--khởi-chạy)
7. [API Reference](#7-api-reference)
8. [Kiểm thử & Đánh giá](#8-kiểm-thử--đánh-giá)
9. [Giới hạn Hiện tại & Lộ trình Phát triển](#9-giới-hạn-hiện-tại--lộ-trình-phát-triển)
10. [Tài liệu Tham khảo & Miễn trừ Y khoa](#10-tài-liệu-tham-khảo--miễn-trừ-y-khoa)

---

## 1. Tổng quan

**NorMedical** (tên trợ lý: **RaumaMedical**) là hệ thống tư vấn sức khỏe sơ bộ và tra cứu dược lý tiếng Việt chạy hoàn toàn cục bộ trên phần cứng phổ thông (**GPU 4GB VRAM**). Dự án giải quyết ba hạn chế phổ biến của các hệ thống hỏi đáp y tế hiện nay:

- **Hỏi bệnh đa lượt có định hướng:** Chủ động khai thác bệnh sử từng câu một (tính chất, thời gian, triệu chứng đi kèm) trước khi phân tích nguyên nhân, tránh kết luận vội vàng ngay từ lượt đầu.
- **Giảm ảo giác dược lý bằng nguồn chính thống:** Truy xuất trực tiếp chuyên luận từ **Dược thư Quốc gia Việt Nam** và **Bệnh học lâm sàng Vinmec**, hiển thị song song nguồn đối chứng theo thời gian thực.
- **Bảo mật dữ liệu bệnh án 100% On-Premise:** Toàn bộ quá trình suy luận và truy xuất tri thức chạy trên máy cục bộ, không gửi thông tin sức khỏe của người dùng ra máy chủ bên ngoài.

---

## 2. Cơ sở Nghiên cứu & Khảo sát Thực tế

Thiết kế hệ thống được xây dựng dựa trên số liệu khảo sát thực địa ở hai nhóm mẫu độc lập:

### Khảo sát Chuyên sâu (`N = 120`)
*Cơ cấu mẫu: 70% người dân • 20% sinh viên Y-Dược • 10% Bác sĩ & Dược sĩ đang công tác.*

| Chỉ tiêu Khảo sát | Tỷ lệ | Hàm ý Thiết kế Hệ thống |
| :--- | :---: | :--- |
| Tự tra cứu triệu chứng & thuốc trên mạng trước khi khám | **81.7%** | Nhu cầu thực tế cao cho công cụ sàng lọc ban đầu |
| Từng dùng AI tổng quát nhưng chỉ hài lòng về độ an toàn | **68.3%** / **23.3%** | Cần mô hình tinh chỉnh chuyên biệt cho y khoa Việt Nam |
| Yêu cầu công cụ dẫn nguồn chính thống trong nước | **94.2%** | Tích hợp trực tiếp Dược thư Quốc gia & hiển thị nguồn đối chứng |
| Lo ngại sai liều / ngộ độc / tương tác thuốc | **62.5%** | Bắt buộc trích xuất liều lượng & chống chỉ định từ Dược thư |
| Kỳ vọng thời gian phản hồi dưới 5 giây | **76.7%** | Triển khai giao thức truyền dòng Streaming SSE (`Thinking...`) |

### Khảo sát Cộng đồng Độc lập (`N = 164`, Thang Likert 1–5)
- Điểm hài lòng hiện trạng với các công cụ AI y tế nói chung: Tốc độ `3.09` (SD `1.45`), Diễn đạt dễ hiểu `2.93` (SD `1.39`), Độ chi tiết `3.05` (SD `1.47`), Giúp yên tâm `3.10` (SD `1.39`), Tổng thể `3.21` (SD `1.35`).
- Lo ngại hàng đầu: Lộ thông tin bệnh sử riêng tư (**29.27%**), câu trả lời chung chung (**25.00%**), không rõ nguồn Bộ Y tế (**18.90%**), sợ dùng nhầm thuốc (**15.85%**).

---

## 3. Kho Tri thức & Cơ chế Truy xuất RAG

Hệ thống vận hành bộ truy xuất **BM25 tùy biến** chạy hoàn toàn trên RAM CPU (độ trễ `< 10 ms`, không chiếm VRAM GPU) trên **2.427 tài liệu chuẩn hóa**:

| Nguồn Dữ liệu | Tệp Lưu trữ | Quy mô | Nội dung Chuẩn hóa |
| :--- | :--- | :---: | :--- |
| **Dược thư Quốc gia Việt Nam** | `src/rag/data/drugs_knowledge_base.jsonl` | `1.754` chuyên luận | Hoạt chất, nhóm dược lý, dạng bào chế, chỉ định, liều dùng, chống chỉ định, ADR |
| **Bệnh học Lâm sàng Vinmec** | `src/rag/data/diseases_knowledge_base.jsonl` | `673` bài bệnh học | Tên bệnh, nguyên nhân, triệu chứng điển hình, hướng xử trí & chăm sóc |

**Các tối ưu truy xuất đã triển khai:**
- **Tokenization lai (Unigram + Bigram + Tiền tố hoạt chất ASCII):** Tăng độ chính xác khi tìm cụm từ y khoa và chịu được lỗi gõ thiếu ký tự tên hoạt chất tiếng Anh.
- **Ánh xạ triệu chứng & biệt dược phổ biến:** Tự động mở rộng từ khóa đời thường và tên thương mại sang hoạt chất Dược thư (`Panadol -> Paracetamol`, `Smecta -> Diosmectit`, `Oresol -> ORS`, `ho -> Dextromethorphan / Acetylcystein / Guaifenesin`...).
- **Lọc nhiễu & chống khớp giả:** Loại bỏ từ dừng y tế (`query_stopwords`), chặn từ đời thường trùng tên hoạt chất (`common_false_drugs`: *nước, muối, đường, sắt, kẽm, gừng, mật ong...*) và tự động ưu tiên chuyên luận giàu dữ liệu liều dùng.

---

## 4. Huấn luyện Mô hình & Tối ưu Suy luận

### 4.1. Tiền xử lý Dữ liệu Huấn luyện
Khảo sát dữ liệu thô `ViHealthQA` phát hiện phần lớn câu hỏi quá ngắn (trung vị `63` ký tự dạng tiêu đề SEO), trong khi câu trả lời nhắc tới bệnh án ẩn hoặc lặp văn mẫu. Hệ thống đã lọc sạch và tổng hợp đa nguồn (`ViHealthQA`, `Vinmec`, `BookingCare`, `Medical-o1-reasoning-vi`) thành:
- **Tập SFT Đơn lượt sạch:** `7.439` mẫu huấn luyện (`train_sft_clean.csv`) và `500` mẫu kiểm định (`val_sft_clean.csv`).
- **Tập Hội thoại Đa lượt (Multi-turn SFT):** `8.063` hội thoại (`train_sft_multiturn_messages.jsonl`).

### 4.2. Thông số Huấn luyện QLoRA

| Tham số | Giá trị Thực tế |
| :--- | :--- |
| **Mô hình nền** | `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` |
| **Kho trọng số công khai** | [huggingface.co/imdlam/q3medical](https://huggingface.co/imdlam/q3medical) |
| **Kỹ thuật tinh chỉnh** | QLoRA 4-bit (Unsloth) trên Tesla T4 GPU |
| **Cấu hình LoRA** | `r = 16`, `lora_alpha = 16`, `lora_dropout = 0.05`, `bias = "none"` |
| **Phạm vi lớp train** | Tinh chỉnh lớp ngôn ngữ (`finetune_language_layers = True`), đóng băng nhánh Vision |
| **Hàm mất mát** | `train_on_responses_only` (chỉ tính loss trên phản hồi `<\|im_start\|>assistant`) |
| **Lịch học & Thời lượng** | `lr = 3e-5`, `cosine` scheduler, `504 steps` (1 epoch trên 8.062 hội thoại) |
| **Độ hội tụ Loss** | Giảm mượt từ **`2.454`** xuống **`~1.51 - 1.55`** |
| **Dung lượng Adapter** | `132 MB` (`adapter_model.safetensors`, trọn bộ 7 tệp ~`137 MB`) |
| **Triển khai cục bộ** | GGUF `Q4_K_M` (~`2.4 - 3.2 GB VRAM`), vận hành ổn định trên Laptop **RTX 3050 (4GB VRAM)** |

### 4.3. Phát hiện Kỹ thuật khi Tối ưu Suy luận
- **Điều chỉnh `LoRA Scaling = 0.25`:** Thực nghiệm cho thấy mức `scaling = 1.0` dễ gây hiện tượng suy giảm khả năng suy luận sinh lý bệnh gốc (*Catastrophic Forgetting*). Hạ `scaling = 0.25` bảo toàn trọn vẹn năng lực suy luận của `Qwen3-4B` nhưng vẫn duy trì văn phong y khoa tiếng Việt.
- **Tối ưu giải mã (Decoding) Tiếng Việt:** Bỏ `no_repeat_ngram_size = 4` (vốn làm biến dạng từ ghép tiếng Việt trên tokenizer âm tiết). Sử dụng `temperature = 0.30`, `top_p = 0.88`, `top_k = 40`, `repeat_penalty = 1.12`, `presence_penalty = 0.35`, `frequency_penalty = 0.25`, `max_tokens = 750`.

---

## 5. Kiến trúc Hệ thống & Cấu trúc Mã nguồn

### Luồng Xử lý 4 Chế độ Lâm sàng

```text
  [Người dùng]
       │
       ▼
  VietnameseMedicalNormalizer  ──► Chuẩn hóa lỗi gõ thiếu dấu ("nước đa" -> "nước đá", "đau học" -> "đau họng")
       │
       ▼
  ConsultationService (Router)
       ├──► Chế độ 1: Cờ đỏ Cấp cứu      ──► Cảnh báo gọi 115 ngay lập tức
       ├──► Chế độ 2: Tra cứu Dược lý    ──► Mở rộng từ khóa -> BM25 RAG (Top 3 Dược thư, 1.500 ký tự) -> system_drug_lookup
       ├──► Chế độ 3: Khai thác Đa lượt  ──► Hỏi cuốn chiếu 1 câu/lượt (tính chất -> triệu chứng kèm -> bối cảnh)
       └──► Chế độ 4: Kết luận           ──► Tổng hợp 3 phần + Tự động gắn Miễn trừ y khoa khi có tư vấn thuốc/điều trị
       │
       ▼
  LMStudioSSEClient (OpenAI-compatible SSE Streaming) ──► FastAPI Web (:8386) / CLI
```

### Cấu trúc Thư mục

```text
q3AI/
├── config.json                     # Cấu hình hệ thống: API, decoding, RAG, prompt, từ điển chuẩn hóa
├── keys.example.json               # Mẫu tệp khóa bảo mật (copy thành keys.json khi chạy)
├── requirements.txt                # Thư viện Python phụ thuộc
├── main.py                         # Điểm khởi chạy CLI & Gradio
├── web/
│   ├── app.py                      # Máy chủ FastAPI SSE Streaming (cổng mặc định 8386)
│   └── static/
│       └── index.html              # Giao diện cổng tư vấn lâm sàng 2 cột (Markdown renderer offline)
├── src/
│   ├── config/settings.py          # Nạp cấu hình dạng frozen dataclass & tự động chuẩn hóa đường dẫn
│   ├── domain/
│   │   ├── models.py               # KnowledgeDocument, ChatMessage, ConsultationTurn
│   │   └── interfaces.py           # Hợp đồng trừu tượng: ITextNormalizer, IRagRetriever, ILLMClient
│   ├── preprocessing/normalizer.py # Chuẩn hóa tiếng Việt, phân loại ý định & mở rộng từ khóa thuốc
│   ├── rag/
│   │   ├── bm25_retriever.py       # Bộ lập chỉ mục & truy xuất BM25 + Bigram
│   │   └── data/                   # Kho tri thức Dược thư Quốc gia & Bệnh học Vinmec (.jsonl)
│   ├── llm/lmstudio_client.py      # Client truyền dòng SSE chuẩn OpenAI API
│   ├── services/consultation_service.py # Điều phối hội thoại đa lượt, định tuyến 4 chế độ & RAG
│   └── ui/                         # Giao diện Terminal CLI & Gradio
└── tests/
    └── test_edge_cases.py          # Bộ kiểm thử tự động 12 tình huống đặc biệt
```

---

## 6. Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Cài đặt thư viện Python
```bash
git clone https://github.com/im-dlam/NorMedical.git
cd NorMedical
pip install -r requirements.txt
```

### Bước 2: Khởi tạo tệp khóa cấu hình
```bash
cp keys.example.json keys.json
```
*(Tệp `keys.json` được khai báo trong `.gitignore` nhằm đảm bảo an toàn khóa truy cập.)*

### Bước 3: Tải mô hình từ Hugging Face & Bật LM Studio Local Server
- **Kho mô hình:** https://huggingface.co/imdlam/q3medical

1. Mở **LM Studio**, vào mục tìm kiếm (**Discover / Search**) và nhập `imdlam/q3medical`, sau đó tải bản **GGUF (`Q4_K_M`)** và nạp mô hình.
2. *(Tùy chọn tải qua CLI):*
   ```bash
   pip install huggingface_hub
   huggingface-cli download imdlam/q3medical --local-dir ./models/q3medical
   ```
3. Đặt **Context Length = `4096`** trong LM Studio để đủ bộ nhớ cho các chuyên luận Dược thư.
4. Vào mục **Developer (`<-->`)** và bật **Start Server** (cổng mặc định `http://localhost:1234/v1/chat/completions`).

### Bước 4: Khởi chạy Ứng dụng

- **Giao diện Web FastAPI (Cổng mặc định `8386`):**
  ```bash
  cd web
  uvicorn app:app --port 8386
  ```
  Mở trình duyệt tại: `http://127.0.0.1:8386`

- **Giao diện Dòng lệnh (CLI Streaming):**
  ```bash
  python main.py --mode cli
  ```
  *(Gõ `reset` để khởi tạo phiên khám mới, `exit` để thoát.)*

---

## 7. API Reference

Máy chủ FastAPI tại `web/app.py` cung cấp các endpoint sau (mặc định `http://127.0.0.1:8386`):

| Method | Endpoint | Mô tả |
| :---: | :--- | :--- |
| `GET` | `/` | Phục vụ giao diện cổng tư vấn lâm sàng 2 cột |
| `GET` | `/api/health` | Trả về trạng thái hệ thống, số tài liệu RAG đã nạp và địa chỉ LM Studio |
| `POST` | `/api/chat` | Nhận `{"message": "..."}` và trả về luồng **Server-Sent Events** (`event: rag`, `event: token`, `event: error`, `event: done`) |
| `POST` | `/api/reset` | Xóa lịch sử hội thoại hiện tại để bắt đầu ca khám mới |

---

## 8. Kiểm thử & Đánh giá

Chạy bộ kiểm thử tự động tích hợp sẵn:
```bash
python tests/test_edge_cases.py
```

**Kết quả kiểm định hiện tại:**
- **Bộ kiểm thử 12 tình huống đặc biệt (`tests/test_edge_cases.py`):** **12/12 PASS** — bao phủ tra cứu thuốc theo triệu chứng, tên hoạt chất, dấu hiệu cờ đỏ cấp cứu 115, sửa lỗi thiếu dấu (`nước đa -> nước đá`, `đau học -> đau họng`) và chống lặp hội thoại đa lượt.
- **Kiểm định truy xuất RAG trên 20 câu hỏi thuốc thực tế** *(ho, đau đầu, sốt, đau răng, đau dạ dày, trào ngược ợ chua, nghẹt mũi, buồn nôn, viêm họng, dị ứng, tiêu chảy, Panadol, Smecta, Oresol, Omeprazole, Amoxicillin, Ibuprofen, Berberin...)*: Đạt **20/20** trích xuất trúng chuyên luận Dược thư Quốc gia.

---

## 9. Giới hạn Hiện tại & Lộ trình Phát triển

- **Mở rộng danh mục biệt dược:** Hiện phủ `1.754` hoạt chất gốc Dược thư Quốc gia; giai đoạn tiếp theo tích hợp cơ sở dữ liệu biệt dược lưu hành (Cục Quản lý Dược - DAV) kết hợp so khớp mờ `RapidFuzz`.
- **Truy xuất ngữ nghĩa lai (Hybrid Search):** Kết hợp `BM25` với tìm kiếm vector ngữ nghĩa (`FAISS`) cho các mô tả triệu chứng dài bằng ngôn ngữ địa phương.
- **Lớp an toàn dược lý độc lập & Thị giác:** Bổ sung module tính liều theo cân nặng (kg) bằng Python thuần, kiểm tra tương tác đa thuốc và kích hoạt nhánh Vision sẵn có của `Qwen3-VL` cho bài toán nhận diện đơn thuốc / vỏ hộp (OCR).

---

## 10. Tài liệu Tham khảo & Miễn trừ Y khoa

### Tài liệu Tham khảo
- Bộ Y tế Việt Nam (2022). *Dược thư Quốc gia Việt Nam lần xuất bản thứ ba*. Nhà xuất bản Y học, Hà Nội.
- Hệ thống Y tế Vinmec. *Cẩm nang Bệnh học Lâm sàng*.
- Qwen Team, Alibaba Cloud (2025). *Qwen3 & Qwen3-VL Technical Report*.
- Dettmers, T. et al. (NeurIPS 2023). *QLoRA: Efficient Finetuning of Quantized LLMs*.
- Hu, E. J. et al. (ICLR 2022). *LoRA: Low-Rank Adaptation of Large Language Models*.
- Lewis, P. et al. (NeurIPS 2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*.

### Miễn trừ Trách nhiệm Y khoa
> Hệ thống **NorMedical** được phát triển cho mục đích nghiên cứu học thuật và hỗ trợ tra cứu thông tin tham khảo theo Dược thư Quốc gia Việt Nam. Mọi phản hồi từ hệ thống **không phải là đơn thuốc** và không thay thế cho việc thăm khám, chẩn đoán hoặc chỉ định điều trị trực tiếp của bác sĩ chuyên khoa.
