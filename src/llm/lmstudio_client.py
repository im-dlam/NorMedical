import json
import urllib.error
import urllib.request
from typing import Iterator
from src.config.settings import ApiConfig, ApiKeys, DecodingConfig
from src.domain.interfaces import ILLMClient
from src.domain.models import ChatMessage


class LMStudioSSEClient(ILLMClient):
    """Single Responsibility: Giao tiếp OpenAI-compatible SSE Streaming với LM Studio Local Server."""

    def __init__(self, api_config: ApiConfig, decoding_config: DecodingConfig, api_keys: ApiKeys):
        self._api_cfg = api_config
        self._dec_cfg = decoding_config
        self._api_key = api_keys.lm_studio_api_key

    def stream_chat(self, messages: list[ChatMessage]) -> Iterator[str]:
        payload = json.dumps({
            "model": self._api_cfg.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self._dec_cfg.temperature,
            "top_p": self._dec_cfg.top_p,
            "top_k": self._dec_cfg.top_k,
            "repeat_penalty": self._dec_cfg.repeat_penalty,
            "presence_penalty": self._dec_cfg.presence_penalty,
            "frequency_penalty": self._dec_cfg.frequency_penalty,
            "max_tokens": self._dec_cfg.max_tokens,
            "stream": self._api_cfg.stream,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        req = urllib.request.Request(self._api_cfg.base_url, data=payload, headers=headers, method="POST")

        in_think = False
        try:
            with urllib.request.urlopen(req, timeout=self._api_cfg.timeout_seconds) as resp:
                if not self._api_cfg.stream:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if "</think>" in content:
                        content = content.split("</think>", 1)[-1]
                    yield content.strip()
                    return

                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    except json.JSONDecodeError:
                        continue

                    if not delta:
                        continue
                    if "<think>" in delta:
                        in_think = True
                        continue
                    if "</think>" in delta:
                        in_think = False
                        after = delta.split("</think>", 1)[-1]
                        if after:
                            yield after
                        continue
                    if not in_think:
                        yield delta

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Không kết nối được tới LM Studio ({self._api_cfg.base_url}). "
                f"Hãy mở LM Studio -> Developer (<-->) -> Start Server. Chi tiết: {exc}"
            ) from exc
