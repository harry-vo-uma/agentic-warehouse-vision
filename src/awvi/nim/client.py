"""Thin client over NVIDIA NIM (OpenAI-compatible) with a deterministic
offline fallback.

Design note: every call site works with *structured* results, never raw
strings. The JSON-repair path exists because VLMs under load will happily
wrap JSON in prose, and a demo that dies on a stray ``` is not a demo.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from typing import Any

from ..config import NIMSettings, get_settings
from .mock import MockNIMBackend

log = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class NIMError(RuntimeError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Handles: bare JSON, ```json fences, and JSON preceded/followed by prose.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if not m:
        raise NIMError(f"no JSON object found in model output: {text[:200]!r}")
    return json.loads(m.group(0))


class NIMClient:
    """Wraps chat completions, vision completions and embeddings.

    When no API key is present (or AWVI_FORCE_MOCK=1) the client transparently
    routes to `MockNIMBackend`, which is seeded and deterministic. That keeps
    the eval harness reproducible and lets the repo run on a laptop.
    """

    def __init__(self, settings: NIMSettings | None = None, mock: MockNIMBackend | None = None):
        self.settings = settings or get_settings().nim
        self.mock = mock or MockNIMBackend()
        self._http = None
        self.call_count = 0
        self.total_latency_ms = 0.0

    # ---------------------------------------------------------------- transport
    @property
    def live(self) -> bool:
        return self.settings.enabled

    def _client(self):
        if self._http is None:
            import httpx  # imported lazily so mock-only runs need no network stack

            self._http = httpx.Client(
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_s,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Accept": "application/json",
                },
            )
        return self._http

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                r = self._client().post(path, json=payload)
                if r.status_code == 429 or r.status_code >= 500:
                    raise NIMError(f"{r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                sleep = min(2**attempt * 0.5, 8.0)
                log.warning("NIM call failed (attempt %d): %s; retrying in %.1fs", attempt + 1, exc, sleep)
                time.sleep(sleep)
        raise NIMError(f"NIM request to {path} failed after retries") from last

    # ------------------------------------------------------------------- public
    def chat(self, prompt: str, *, system: str = "", temperature: float = 0.2, max_tokens: int = 900) -> str:
        t0 = time.perf_counter()
        try:
            if not self.live:
                return self.mock.chat(prompt, system=system)
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
            data = self._post(
                "/chat/completions",
                {
                    "model": self.settings.llm_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            return data["choices"][0]["message"]["content"]
        finally:
            self.call_count += 1
            self.total_latency_ms += (time.perf_counter() - t0) * 1000

    def vlm(
        self,
        prompt: str,
        images_b64: Sequence[str],
        *,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 900,
        mock_hint: dict[str, Any] | None = None,
    ) -> str:
        """Multi-image reasoning call.

        `mock_hint` carries the ground-truth-free scene features the mock
        backend uses to synthesise a plausible reading. It is ignored entirely
        when a live endpoint is configured.
        """
        t0 = time.perf_counter()
        try:
            if not self.live:
                return self.mock.vlm(prompt, len(images_b64), hint=mock_hint or {})
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for b64 in images_b64:
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": content}
            ]
            data = self._post(
                "/chat/completions",
                {
                    "model": self.settings.vlm_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            return data["choices"][0]["message"]["content"]
        finally:
            self.call_count += 1
            self.total_latency_ms += (time.perf_counter() - t0) * 1000

    def embed(self, texts: Sequence[str], *, input_type: str = "passage") -> list[list[float]]:
        if not self.live:
            return [self.mock.embed(t) for t in texts]
        data = self._post(
            "/embeddings",
            {
                "model": self.settings.embed_model,
                "input": list(texts),
                "input_type": input_type,
                "encoding_format": "float",
            },
        )
        return [d["embedding"] for d in data["data"]]

    def json_call(self, prompt: str, **kw) -> dict[str, Any]:
        """chat() + tolerant JSON extraction, with one repair round-trip."""
        raw = self.chat(prompt, **kw)
        try:
            return extract_json(raw)
        except (NIMError, json.JSONDecodeError):
            repaired = self.chat(
                "Return ONLY valid minified JSON, no prose, no code fences. "
                f"Fix this so it parses:\n{raw[:2000]}",
                temperature=0.0,
            )
            return extract_json(repaired)

    def stats(self) -> dict[str, Any]:
        return {
            "mode": "live" if self.live else "mock",
            "calls": self.call_count,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(self.total_latency_ms / self.call_count, 2) if self.call_count else 0.0,
        }


_CLIENT: NIMClient | None = None


def get_client() -> NIMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = NIMClient()
    return _CLIENT
