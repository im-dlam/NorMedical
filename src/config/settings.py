import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiKeys:
    lm_studio_api_key: str
    huggingface_token: str


@dataclass(frozen=True)
class AppConfig:
    name: str
    version: str
    max_history_turns: int


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    model: str
    timeout_seconds: int
    stream: bool


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    max_tokens: int


@dataclass(frozen=True)
class RagConfig:
    drugs_jsonl: str
    diseases_jsonl: str
    dav_brands_jsonl: str
    min_score_threshold: float
    top_k_drugs: int
    top_k_diseases: int
    max_snippet_chars: int
    min_words_for_auto_rag: int


@dataclass(frozen=True)
class PromptsConfig:
    greeting: str
    medical_disclaimer: str
    system_drug_lookup: str
    system_clinical: str


@dataclass(frozen=True)
class NlpConfig:
    typo_map: dict[str, str]
    common_false_drugs: frozenset[str]
    query_stopwords: frozenset[str]
    identity_keywords: tuple[str, ...]
    greeting_words: frozenset[str]
    drug_intent_keywords: tuple[str, ...]
    no_drug_phrases: tuple[str, ...]
    confirm_words: frozenset[str]
    red_flag_keywords: tuple[str, ...]
    symptom_drug_synonyms: dict[str, str]


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    api: ApiConfig
    decoding: DecodingConfig
    rag: RagConfig
    prompts: PromptsConfig
    nlp: NlpConfig
    keys: ApiKeys


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình bắt buộc: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_settings(root_dir: Path | None = None) -> Settings:
    base = root_dir or Path(__file__).resolve().parents[2]
    cfg = _load_json(base / "config.json")

    keys_path = base / "keys.json"
    if keys_path.exists():
        keys_raw = _load_json(keys_path)
    else:
        keys_raw = {
            "lm_studio_api_key": os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            "huggingface_token": os.getenv("HF_TOKEN", ""),
        }

    rag_raw = dict(cfg["rag"])
    rag_raw.setdefault("dav_brands_jsonl", ".\\src\\rag\\data\\dav_55k_brands.jsonl")
    for key in ("drugs_jsonl", "diseases_jsonl", "dav_brands_jsonl"):
        p = Path(rag_raw[key])
        if not p.is_absolute():
            rag_raw[key] = str((base / p).resolve())

    nlp_raw = cfg["nlp"]
    return Settings(
        app=AppConfig(**cfg["app"]),
        api=ApiConfig(**cfg["api"]),
        decoding=DecodingConfig(**cfg["decoding"]),
        rag=RagConfig(**rag_raw),
        prompts=PromptsConfig(**cfg["prompts"]),
        nlp=NlpConfig(
            typo_map=dict(nlp_raw["typo_map"]),
            common_false_drugs=frozenset(nlp_raw["common_false_drugs"]),
            query_stopwords=frozenset(nlp_raw["query_stopwords"]),
            identity_keywords=tuple(nlp_raw["identity_keywords"]),
            greeting_words=frozenset(nlp_raw["greeting_words"]),
            drug_intent_keywords=tuple(nlp_raw["drug_intent_keywords"]),
            no_drug_phrases=tuple(nlp_raw["no_drug_phrases"]),
            confirm_words=frozenset(nlp_raw["confirm_words"]),
            red_flag_keywords=tuple(nlp_raw.get("red_flag_keywords", [])),
            symptom_drug_synonyms=dict(nlp_raw.get("symptom_drug_synonyms", {})),
        ),
        keys=ApiKeys(
            lm_studio_api_key=keys_raw.get("lm_studio_api_key", "lm-studio"),
            huggingface_token=keys_raw.get("huggingface_token", ""),
        ),
    )
