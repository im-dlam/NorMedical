import re
import unicodedata
from src.config.settings import NlpConfig
from src.domain.interfaces import ITextNormalizer


class VietnameseMedicalNormalizer(ITextNormalizer):
    """Single Responsibility: Chuẩn hóa lỗi chính tả tiếng Việt & nhận diện ý định đầu vào."""

    def __init__(self, nlp_config: NlpConfig):
        self._cfg = nlp_config
        self._compiled_typos = [
            (re.compile(pattern, flags=re.IGNORECASE), replacement)
            for pattern, replacement in nlp_config.typo_map.items()
        ]

    def normalize(self, text: str) -> str:
        cleaned = text.strip()
        for regex, replacement in self._compiled_typos:
            cleaned = regex.sub(replacement, cleaned)
        return cleaned

    def strip_accents(self, text: str) -> str:
        s = text.lower().replace("đ", "d")
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    def is_greeting_or_identity(self, normalized_text: str) -> bool:
        words = normalized_text.split()
        no_accent = self.strip_accents(normalized_text)
        if any(k in no_accent for k in self._cfg.identity_keywords):
            return True
        return len(words) <= 4 and any(w in self._cfg.greeting_words for w in no_accent.split())

    def has_drug_intent(self, normalized_text: str) -> bool:
        no_accent = self.strip_accents(normalized_text)
        if any(phrase in no_accent for phrase in self._cfg.no_drug_phrases):
            return False
        return any(k in no_accent for k in self._cfg.drug_intent_keywords)

    def is_short_confirmation(self, normalized_text: str) -> bool:
        words = normalized_text.split()
        no_accent = self.strip_accents(normalized_text)
        return len(words) <= 5 and any(w in self._cfg.confirm_words for w in no_accent.split())

    def is_red_flag_emergency(self, normalized_text: str) -> bool:
        no_accent = self.strip_accents(normalized_text)
        return any(flag in no_accent for flag in self._cfg.red_flag_keywords)

    def expand_drug_query(self, normalized_text: str) -> str:
        no_accent = self.strip_accents(normalized_text)
        tokens_set = set(re.findall(r"\w+", no_accent))
        extras: list[str] = []
        for key, expansion in self._cfg.symptom_drug_synonyms.items():
            if " " in key:
                if key in no_accent:
                    extras.append(expansion)
            else:
                if key in tokens_set:
                    extras.append(expansion)
        if not extras:
            return normalized_text
        return f"{normalized_text} {' '.join(extras)}"
