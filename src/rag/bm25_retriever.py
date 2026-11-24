import json
import math
import os
import re
from collections import Counter, defaultdict
from src.config.settings import NlpConfig, RagConfig
from src.domain.interfaces import IRagRetriever
from src.domain.models import DocType, KnowledgeDocument


_NON_BRAND_WORDS = frozenset({
    "đau", "họng", "bụng", "đầu", "lưng", "ngực", "răng", "mắt", "mũi", "tai",
    "sốt", "ho", "đờm", "mệt", "mỏi", "uể", "oải", "chóng", "mặt", "buồn", "nôn",
    "tiêu", "chảy", "táo", "bón", "ngứa", "dị", "ứng", "mề", "đay", "viêm", "nhiễm",
    "nước", "đá", "muối", "đường", "sữa", "cơm", "cháo", "trà", "rượu", "bia",
    "lạnh", "nóng", "khó", "thở", "nuốt", "vướng", "rát", "sưng", "đỏ", "hạch",
    "sáng", "trưa", "chiều", "tối", "đêm", "hôm", "qua", "nay", "ngày", "tuần", "tháng",
    "năm", "tuổi", "người", "lớn", "trẻ", "em", "bé", "bầu", "thai", "mẹ",
    "bị", "thấy", "cảm", "kèm", "theo", "từ", "lâu", "mấy", "nhiều", "ít", "nhẹ", "nặng",
    "béo", "gầy", "tăng", "giảm", "cân", "lỏng", "phân", "máu", "mủ", "trào", "ngược",
    "dạ", "dày", "gan", "thận", "tim", "phổi", "ruột", "xương", "khớp", "da", "tóc",
    "uống", "dùng", "thuốc", "viên", "gói", "liều", "cách", "trị", "chữa", "khám",
    "hỏi", "biết", "muốn", "cần", "nên", "phải", "được", "không", "chưa", "rồi", "đang",
    "xem", "giúp", "tư", "vấn", "như", "thế", "nào", "sao", "gì", "đâu", "lúc", "khi",
    "nhưng", "hoặc", "hay", "với", "cho", "của", "tại", "nhà", "bệnh", "viện", "bác", "sĩ",
    "extra", "plus", "forte", "kids", "baby", "max", "pro", "gold", "new", "tab", "cap",
})


class BM25MedicalRetriever(IRagRetriever):
    """Single Responsibility: Truy xuất BM25 (Dược thư + Vinmec) & Tra cứu nhanh 55.005 Biệt dược DAV."""

    def __init__(self, rag_config: RagConfig, nlp_config: NlpConfig):
        self._rag_cfg = rag_config
        self._nlp_cfg = nlp_config
        self._documents: list[KnowledgeDocument] = []
        self._doc_freqs: Counter[str] = Counter()
        self._doc_tokens: list[Counter[str]] = []
        self._title_tokens: list[list[str]] = []
        self._avg_dl: float = 0.0

        # Kho biệt dược DAV 55k (chỉ mục đảo siêu nhẹ + tra cứu O(1) theo Số GPLH)
        self._dav_records: list[tuple[KnowledgeDocument, str, str]] = []  # (doc, brand_lower, active_ingredient)
        self._dav_first_token_idx: dict[str, list[int]] = defaultdict(list)
        self._dav_gplh_idx: dict[str, int] = {}

        self._load_kb(rag_config.drugs_jsonl, "drug")
        self._load_kb(rag_config.diseases_jsonl, "disease")
        self._build_index()
        self._load_dav_brands(rag_config.dav_brands_jsonl)

    @property
    def document_count(self) -> int:
        return len(self._documents) + len(self._dav_records)

    def _tokenize(self, text: str) -> list[str]:
        words = [
            w for w in re.findall(r"\w+", text.lower())
            if len(w) >= 2 and w not in self._nlp_cfg.query_stopwords
        ]
        bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def _load_kb(self, path: str, doc_type: DocType) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                title = item.get("drug_name" if doc_type == "drug" else "disease_name", "")
                body = (
                    f"{title} {title} {item.get('drug_category', '')} "
                    f"{item.get('indications', '')} {item.get('symptoms', '')}"
                )
                self._documents.append(
                    KnowledgeDocument(
                        doc_type=doc_type,
                        title=title,
                        search_body=body,
                        rag_text=item.get("rag_text", ""),
                    )
                )

    def _build_index(self) -> None:
        total_len = 0
        for doc in self._documents:
            tokens = self._tokenize(doc.search_body)
            self._doc_tokens.append(Counter(tokens))
            self._title_tokens.append(self._tokenize(doc.title))
            total_len += len(tokens)
            for token in set(tokens):
                self._doc_freqs[token] += 1
        self._avg_dl = total_len / max(1, len(self._documents))

    def search(self, query: str, top_k_drugs: int = 2, top_k_diseases: int = 1) -> list[KnowledgeDocument]:
        q_tokens = self._tokenize(query)
        q_lower = query.lower()
        n_docs = len(self._documents)
        drug_scores: list[tuple[float, KnowledgeDocument]] = []
        disease_scores: list[tuple[float, KnowledgeDocument]] = []

        for idx, doc in enumerate(self._documents):
            title_low = doc.title.lower().strip()
            if title_low in self._nlp_cfg.common_false_drugs:
                continue

            tf_c = self._doc_tokens[idx]
            dl = sum(tf_c.values())
            score = 0.0

            for qt in q_tokens:
                if qt in tf_c:
                    df = self._doc_freqs[qt]
                    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    score += idf * (tf_c[qt] * 2.5) / (tf_c[qt] + 1.5 * (0.25 + 0.75 * dl / self._avg_dl))

                for tt in self._title_tokens[idx]:
                    if "_" in qt and qt == tt:
                        score += 20.0
                    elif qt.isascii() and tt.isascii() and len(qt) >= 6 and len(tt) >= 6:
                        if qt == tt or (abs(len(qt) - len(tt)) <= 2 and (tt.startswith(qt) or qt.startswith(tt))):
                            score += 25.0

            if len(title_low) >= 4 and title_low in q_lower:
                score += 35.0

            # Ưu tiên chuyên luận có đầy đủ liều dùng/chỉ định hơn các mục rỗng trong Dược thư
            if doc.doc_type == "drug" and doc.rag_text.count("Chưa có thông tin cụ thể") >= 5:
                score *= 0.45

            if score >= self._rag_cfg.min_score_threshold:
                (drug_scores if doc.doc_type == "drug" else disease_scores).append((score, doc))

        drug_scores.sort(key=lambda x: x[0], reverse=True)
        disease_scores.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in disease_scores[:top_k_diseases]] + [d for _, d in drug_scores[:top_k_drugs]]

    def _load_dav_brands(self, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                brand = item.get("brand_name", "").strip()
                if len(brand) < 3:
                    continue
                brand_low = brand.lower()
                ingredient = item.get("active_ingredient", "").strip()
                doc = KnowledgeDocument(
                    doc_type="drug",
                    title=f"{brand} (DAV)",
                    search_body=f"{brand} {ingredient}",
                    rag_text=item.get("rag_text", ""),
                )
                rec_idx = len(self._dav_records)
                self._dav_records.append((doc, brand_low, ingredient))

                gplh_codes = list(item.get("gplh_all", []))
                m_gplh = re.search(r"Số GPLH \(DAV\):\s*([^\n]+)", doc.rag_text)
                if m_gplh:
                    gplh_codes.extend(c.strip() for c in m_gplh.group(1).split(","))
                for code in gplh_codes:
                    norm_code = re.sub(r"[^a-z0-9]", "", str(code).lower())
                    if len(norm_code) >= 6:
                        self._dav_gplh_idx[norm_code] = rec_idx

                tokens = re.findall(r"[a-zA-Z0-9À-ỹ]+", brand_low)
                if tokens:
                    first_tok = tokens[0]
                    if len(first_tok) >= 4 and first_tok not in _NON_BRAND_WORDS and first_tok not in self._nlp_cfg.query_stopwords:
                        self._dav_first_token_idx[first_tok].append(rec_idx)

    def search_dav_brands(self, query: str, top_k: int = 2) -> list[tuple[KnowledgeDocument, str]]:
        if not self._dav_records:
            return []
        q_low = query.lower()

        # Ưu tiên 1: Tra cứu chính xác O(1) theo Số GPLH / Số đăng ký (vd: 955100201626, VD-12345-19)
        code_candidates = re.findall(r"\b[a-z]{0,4}[-]?\d{4,14}(?:[-]\d{2,4})?\b", q_low)
        exact_gplh_hits: list[tuple[KnowledgeDocument, str]] = []
        seen_gplh_idx: set[int] = set()
        for raw_code in code_candidates:
            norm_code = re.sub(r"[^a-z0-9]", "", raw_code)
            if len(norm_code) >= 6 and norm_code in self._dav_gplh_idx:
                idx = self._dav_gplh_idx[norm_code]
                if idx not in seen_gplh_idx:
                    seen_gplh_idx.add(idx)
                    doc, _, ingredient = self._dav_records[idx]
                    exact_gplh_hits.append((doc, ingredient))
        if exact_gplh_hits:
            return exact_gplh_hits[:top_k]

        q_tokens = re.findall(r"[a-zA-Z0-9À-ỹ]+", q_low)
        candidate_ids: set[int] = set()

        for tok in q_tokens:
            if len(tok) < 4 or tok in _NON_BRAND_WORDS or tok in self._nlp_cfg.query_stopwords:
                continue
            if tok in self._dav_first_token_idx:
                candidate_ids.update(self._dav_first_token_idx[tok])

        if not candidate_ids:
            return []

        scored: list[tuple[float, KnowledgeDocument, str]] = []
        q_tok_set = set(q_tokens)

        for idx in candidate_ids:
            doc, brand_low, ingredient = self._dav_records[idx]
            brand_toks = re.findall(r"[a-zA-Z0-9À-ỹ]+", brand_low)
            if not brand_toks:
                continue

            score = 0.0
            # Khớp chính xác toàn bộ tên biệt dược (có ranh giới từ)
            if re.search(rf"\b{re.escape(brand_low)}\b", q_low):
                score += 100.0 + len(brand_low)
            else:
                # Khớp từ đầu tiên đặc trưng (ASCII >= 5 ký tự) + số lượng từ trùng
                first_tok = brand_toks[0]
                if first_tok in q_tok_set and first_tok.isascii() and len(first_tok) >= 5:
                    overlap = sum(1 for t in brand_toks if t in q_tok_set)
                    score += 40.0 + overlap * 15.0 - abs(len(brand_toks) - overlap) * 4.0

            if score >= 35.0:
                scored.append((score, doc, ingredient))

        scored.sort(key=lambda x: x[0], reverse=True)
        seen_titles: set[str] = set()
        results: list[tuple[KnowledgeDocument, str]] = []
        for _, doc, ingredient in scored:
            if doc.title in seen_titles:
                continue
            seen_titles.add(doc.title)
            results.append((doc, ingredient))
            if len(results) >= top_k:
                break
        return results
