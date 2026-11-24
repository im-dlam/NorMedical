from typing import Iterator
from src.config.settings import Settings
from src.domain.interfaces import ILLMClient, IRagRetriever, ITextNormalizer
from src.domain.models import ChatMessage, KnowledgeDocument


class ConsultationService:
    """
    Application Service (Dependency Inversion):
    Chỉ phụ thuộc vào các interface (ITextNormalizer, IRagRetriever, ILLMClient).
    """

    def __init__(
        self,
        settings: Settings,
        normalizer: ITextNormalizer,
        retriever: IRagRetriever,
        llm_client: ILLMClient,
    ):
        self._settings = settings
        self._normalizer = normalizer
        self._retriever = retriever
        self._llm = llm_client
        self._history: list[ChatMessage] = []

    @property
    def knowledge_count(self) -> int:
        return self._retriever.document_count

    def reset_session(self) -> None:
        self._history.clear()

    def _select_rag_docs(self, normalized_query: str) -> list[KnowledgeDocument]:
        dav_hits = self._retriever.search_dav_brands(normalized_query, top_k=2)
        has_drug = self._normalizer.has_drug_intent(normalized_query) or bool(dav_hits)
        is_confirm = len(self._history) >= 2 and self._normalizer.is_short_confirmation(normalized_query)
        is_long_desc = len(normalized_query.split()) >= self._settings.rag.min_words_for_auto_rag

        if not (has_drug or is_confirm or is_long_desc):
            return []

        if is_confirm:
            user_turns = [m.content for m in self._history if m.role == "user"]
            search_query = " ".join(user_turns + [normalized_query])
        else:
            search_query = normalized_query

        if has_drug:
            search_query = self._normalizer.expand_drug_query(search_query)

        # Liên kết chéo Hoạt chất chính từ DAV 55k sang chuyên luận Dược thư Quốc gia
        if dav_hits:
            primary_ingredient = dav_hits[0][1]
            kb_docs = (
                self._retriever.search(primary_ingredient, top_k_drugs=2, top_k_diseases=0)
                if primary_ingredient
                else []
            )
            merged: list[KnowledgeDocument] = [dav_hits[0][0]]
            seen_titles = {dav_hits[0][0].title}
            for doc in kb_docs:
                if doc.title not in seen_titles:
                    merged.append(doc)
                    seen_titles.add(doc.title)
            return merged[:3]

        top_k_drugs = self._settings.rag.top_k_drugs if has_drug else 1
        top_k_diseases = 0 if has_drug else self._settings.rag.top_k_diseases
        return self._retriever.search(
            search_query,
            top_k_drugs=top_k_drugs,
            top_k_diseases=top_k_diseases,
        )

    def _build_phase_directive(self, normalized_query: str, docs: list[KnowledgeDocument]) -> str:
        """
        Bộ định tuyến 4 chế độ lâm sàng (Red Flag / Drug Lookup / Symptom Probing / Conclusion):
        - Gặp dấu hiệu cấp cứu -> Cảnh báo gọi 115 ngay lập tức.
        - Hỏi trực tiếp về thuốc -> Liệt kê ngay hoạt chất/thuốc phù hợp từ Dược thư RAG.
        - Khai thác triệu chứng -> Hỏi cuốn chiếu 1 câu tự nhiên/lượt (cấm nhãn 'Hỏi thêm:').
        - Đủ dữ kiện hoặc trả lời phủ định ngắn -> Dừng hỏi, chốt tư vấn 3 phần.
        """
        if self._normalizer.is_red_flag_emergency(normalized_query):
            return (
                "\n\n[TRẠNG THÁI CỜ ĐỎ CẤP CỨU]: "
                "Bệnh nhân có dấu hiệu nguy hiểm đe dọa tính mạng! "
                "Hãy cảnh báo khẩn cấp ngay đầu câu, yêu cầu gọi cấp cứu 115 hoặc đến khoa Cấp cứu gần nhất NGAY LẬP TỨC, "
                "và hướng dẫn 2-3 bước sơ cứu an toàn trong lúc chờ cấp cứu. Không hỏi lòng vòng."
            )

        if self._normalizer.has_drug_intent(normalized_query):
            return (
                "\n\n[TRẠNG THÁI TRA CỨU DƯỢC LÝ — TRẢ LỜI TRỰC TIẾP]: "
                "Bệnh nhân đang hỏi trực tiếp về THUỐC. KHÔNG được né tránh chỉ hỏi lại! "
                "Hãy liệt kê ngay các nhóm thuốc / hoạt chất không kê đơn phù hợp (ưu tiên bám sát [THÔNG TIN DƯỢC THƯ] đã cung cấp, "
                "phân nhóm rõ ràng nếu cần — ví dụ ho khan vs ho có đờm), nêu rõ công dụng chính, lưu ý an toàn / chống chỉ định quan trọng. "
                "Cuối câu chỉ hỏi thêm TỐI ĐA 1 câu ngắn tự nhiên (không dùng nhãn 'Hỏi thêm:') để lọc chính xác loại phù hợp nhất với bệnh nhân."
            )

        current_turn = sum(1 for m in self._history if m.role == "user") + 1
        no_accent = self._normalizer.strip_accents(normalized_query)
        words = no_accent.split()
        is_short_negative = len(words) <= 4 and any(
            w in {"khong", "ko", "k", "chua"} for w in words
        )

        if current_turn == 1:
            return (
                "\n\n[TRẠNG THÁI LƯỢT 1 — KHAI THÁC ĐẦU TIÊN]: "
                "Đồng cảm ngắn gọn 1 câu rồi đặt tự nhiên ĐÚNG 1 câu hỏi về tính chất hoặc thời gian triệu chứng. "
                "KHÔNG dùng nhãn 'Hỏi thêm:' và KHÔNG liệt kê 1, 2, 3."
            )
        if current_turn == 2 and not is_short_negative:
            return (
                "\n\n[TRẠNG THÁI LƯỢT 2 — KHAI THÁC BỔ SUNG]: "
                "Ghi nhận ngắn gọn 1 câu, TUYỆT ĐỐI KHÔNG lặp lại câu hỏi ở lượt 1. "
                "Đặt tự nhiên ĐÚNG 1 câu hỏi MỚI KHÁC HOÀN TOÀN (triệu chứng đi kèm, thói quen sinh hoạt, thuốc đang dùng hoặc yếu tố kích phát). "
                "KHÔNG dùng nhãn 'Hỏi thêm:'."
            )
        return (
            "\n\n[TRẠNG THÁI KẾT LUẬN — DỪNG HỎI BỆNH]: "
            "Bệnh nhân đã cung cấp đủ thông tin (hoặc xác nhận không có thêm triệu chứng khác). "
            "TUYỆT ĐỐI KHÔNG ĐƯỢC HỎI THÊM BẤT KỲ CÂU HỎI NÀO NỮA và KHÔNG lặp lại câu trả lời trước! "
            "Đi thẳng vào tư vấn hoàn chỉnh gồm 3 phần rõ ràng: "
            "(1) Nhận định các nguyên nhân khả dĩ nhất dựa trên thông tin đã có, "
            "(2) Hướng chăm sóc / điều chỉnh sinh hoạt hoặc thuốc hỗ trợ an toàn (bám sát Dược thư nếu có), "
            "(3) Dấu hiệu cảnh báo cần đi khám bác sĩ chuyên khoa."
        )

    def _build_messages(self, normalized_query: str, docs: list[KnowledgeDocument]) -> list[ChatMessage]:
        rag_block = ""
        if docs:
            limit = self._settings.rag.max_snippet_chars
            snippets = [doc.rag_text[:limit] for doc in docs]
            rag_block = "\n\n[THÔNG TIN DƯỢC THƯ QUỐC GIA & PHÁC ĐỒ THAM KHẢO]:\n" + "\n---\n".join(snippets)

        is_drug_mode = (
            (self._normalizer.has_drug_intent(normalized_query) or any("(DAV)" in d.title for d in docs))
            and not self._normalizer.is_red_flag_emergency(normalized_query)
        )
        if is_drug_mode:
            # Dùng riêng System Prompt Tra cứu Dược lý để không bị xung đột với luật "Lượt 1 chỉ hỏi 1 câu"
            sys_content = self._settings.prompts.system_drug_lookup + rag_block
        else:
            phase_directive = self._build_phase_directive(normalized_query, docs)
            sys_content = self._settings.prompts.system_clinical + rag_block + phase_directive

        messages = [ChatMessage(role="system", content=sys_content)]
        max_msgs = self._settings.app.max_history_turns
        messages.extend(self._history[-max_msgs:])
        messages.append(ChatMessage(role="user", content=normalized_query))
        return messages

    def _should_append_disclaimer(self, normalized_query: str, final_reply: str) -> bool:
        """
        Chỉ gắn Tuyên bố Miễn trừ Y khoa khi thực sự có tư vấn thuốc, cấp cứu,
        hoặc đã đưa ra kết luận/hướng xử trí — KHÔNG gắn khi bác sĩ mới đang hỏi khai thác triệu chứng.
        """
        if self._normalizer.has_drug_intent(normalized_query):
            return True
        if self._normalizer.is_red_flag_emergency(normalized_query):
            return True

        reply_no_accent = self._normalizer.strip_accents(final_reply)
        # Nếu câu trả lời ngắn và chủ yếu đang đặt câu hỏi khai thác tiếp -> chưa gắn cảnh báo
        if len(final_reply) < 380 and "?" in final_reply:
            return False

        treatment_markers = (
            "lieu dung", "lieu luong", "hoat chat", "thuoc", "duoc thu",
            "nguyen nhan kha di", "huong cham soc", "cach cham soc",
            "dau hieu nguy hiem", "canh bao", "mg", "goi/ngay", "vien"
        )
        return any(marker in reply_no_accent for marker in treatment_markers)

    def stream_turn(self, raw_user_input: str) -> tuple[list[KnowledgeDocument], Iterator[str]]:
        """
        Trả về (docs_trích_xuất, token_generator).
        History tự động cập nhật sau khi stream hoàn tất.
        """
        normalized = self._normalizer.normalize(raw_user_input)

        if self._normalizer.is_greeting_or_identity(normalized):
            def _greeting_gen() -> Iterator[str]:
                yield self._settings.prompts.greeting
            return [], _greeting_gen()

        docs = self._select_rag_docs(normalized)
        messages = self._build_messages(normalized, docs)

        def _response_gen() -> Iterator[str]:
            collected: list[str] = []
            for token in self._llm.stream_chat(messages):
                collected.append(token)
                yield token
            final_reply = "".join(collected).strip()
            if final_reply:
                if self._should_append_disclaimer(normalized, final_reply):
                    yield self._settings.prompts.medical_disclaimer
                self._history.append(ChatMessage(role="user", content=normalized))
                self._history.append(ChatMessage(role="assistant", content=final_reply))

        return docs, _response_gen()
