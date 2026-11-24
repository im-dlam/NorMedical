import gradio as gr
from src.domain.models import KnowledgeDocument
from src.services.consultation_service import ConsultationService


def _format_rag_panel(docs: list[KnowledgeDocument]) -> str:
    if not docs:
        return "### 🔍 Trạng thái Lâm sàng: Khai thác & Phân tích Bệnh sử\n*(Đang hỏi khai thác triệu chứng — Chưa cần trích xuất Dược thư)*"
    cards = []
    for doc in docs:
        badge = "💊 DƯỢC THƯ QUỐC GIA" if doc.doc_type == "drug" else "🩺 PHÁC ĐỒ BỆNH HỌC"
        cards.append(f"#### {badge}: {doc.title}\n```text\n{doc.rag_text[:850]}\n```")
    return "\n\n".join(cards)


def run_web_ui(service: ConsultationService) -> None:
    def chat_handler(user_message: str, chat_history: list):
        if not user_message.strip():
            yield "", chat_history, "*(Hãy nhập triệu chứng hoặc tên thuốc cần tra cứu)*"
            return

        docs, token_stream = service.stream_turn(user_message)
        rag_md = _format_rag_panel(docs)
        chat_history = chat_history + [(user_message, "")]
        partial = ""

        try:
            for token in token_stream:
                partial += token
                chat_history[-1] = (user_message, partial)
                yield "", chat_history, rag_md
        except ConnectionError as err:
            chat_history[-1] = (user_message, f"⚠️ {err}")
            yield "", chat_history, rag_md

    def reset_handler():
        service.reset_session()
        return [], "### 📚 Đã làm mới bệnh án."

    with gr.Blocks(theme=gr.themes.Soft(), title="RaumaMedical q3AI (SOLID + Streaming)") as demo:
        gr.Markdown("# 🏥 RaumaMedical q3AI — Bác Sĩ AI Tư Vấn Lâm Sàng & Tra Cứu Dược Thư Quốc Gia")
        with gr.Row():
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(height=520, label="Phòng Khám Lâm Sàng RaumaMedical (Multi-Turn Streaming)")
                msg = gr.Textbox(
                    placeholder="Mô tả triệu chứng (vd: Sáng dậy tôi bị đau họng) hoặc hỏi thuốc (vd: Liều dùng Paracetamol)...",
                    label="Bệnh nhân nhập thông tin",
                )
                with gr.Row():
                    send_btn = gr.Button("Gửi thông tin", variant="primary")
                    clear_btn = gr.Button("Khám ca mới (Reset)")
            with gr.Column(scale=5):
                rag_box = gr.Markdown(
                    value=f"### 📚 Kho Tri Thức RAG ({service.knowledge_count} Tài liệu Dược thư & Bệnh học)\n*Nguồn dẫn chứng tự động hiển thị tại đây.*"
                )

        msg.submit(chat_handler, [msg, chatbot], [msg, chatbot, rag_box])
        send_btn.click(chat_handler, [msg, chatbot], [msg, chatbot, rag_box])
        clear_btn.click(reset_handler, None, [chatbot, rag_box])

    demo.launch(inbrowser=True)
