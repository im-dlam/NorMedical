import json
import math
import os
import re
from collections import Counter
from src.config.settings import NlpConfig, RagConfig
from src.domain.interfaces import IRagRetriever
from src.domain.models import DocType, KnowledgeDocument


class BM25MedicalRetriever(IRagRetriever):
    """Single Responsibility: Lập chỉ mục và truy xuất BM25 + Bigram trên Dược thư Quốc gia & Bệnh học Vinmec."""

    def __init__(self, rag_config: RagConfig, nlp_config: NlpConfig):
        self._rag_cfg = rag_config
        self._nlp_cfg = nlp_config
        self._documents: list[KnowledgeDocument] = []
        self._doc_freqs: Counter[str] = Counter()
        self._doc_tokens: list[Counter[str]] = []
        self._title_tokens: list[list[str]] = []
        self._avg_dl: float = 0.0

        self._load_kb(rag_config.drugs_jsonl, "drug")
        self._load_kb(rag_config.diseases_jsonl, "disease")
        self._build_index()

    @property
    def document_count(self) -> int:
        return len(self._documents)

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
