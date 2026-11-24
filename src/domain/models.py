from dataclasses import dataclass, field
from typing import Literal

DocType = Literal["drug", "disease"]
RoleType = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class KnowledgeDocument:
    doc_type: DocType
    title: str
    search_body: str
    rag_text: str


@dataclass(frozen=True)
class ChatMessage:
    role: RoleType
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ConsultationTurn:
    user_query_raw: str
    user_query_normalized: str
    assistant_reply: str
    retrieved_docs: list[KnowledgeDocument] = field(default_factory=list)
    is_greeting: bool = False
