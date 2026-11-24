from abc import ABC, abstractmethod
from typing import Iterator
from src.domain.models import ChatMessage, KnowledgeDocument


class ITextNormalizer(ABC):
    """Interface Segregation: Chuẩn hóa văn bản & phân loại ý định cơ bản."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        pass

    @abstractmethod
    def strip_accents(self, text: str) -> str:
        pass

    @abstractmethod
    def is_greeting_or_identity(self, normalized_text: str) -> bool:
        pass

    @abstractmethod
    def has_drug_intent(self, normalized_text: str) -> bool:
        pass

    @abstractmethod
    def is_short_confirmation(self, normalized_text: str) -> bool:
        pass

    @abstractmethod
    def is_red_flag_emergency(self, normalized_text: str) -> bool:
        pass

    @abstractmethod
    def expand_drug_query(self, normalized_text: str) -> str:
        pass


class IRagRetriever(ABC):
    """Dependency Inversion: Bất kỳ engine RAG nào (BM25, FAISS, Hybrid) đều triển khai interface này."""

    @abstractmethod
    def search(self, query: str, top_k_drugs: int = 2, top_k_diseases: int = 1) -> list[KnowledgeDocument]:
        pass

    @property
    @abstractmethod
    def document_count(self) -> int:
        pass


class ILLMClient(ABC):
    """Dependency Inversion: Dễ dàng thay thế LM Studio bằng OpenAI, Ollama, vLLM mà không sửa service."""

    @abstractmethod
    def stream_chat(self, messages: list[ChatMessage]) -> Iterator[str]:
        pass
