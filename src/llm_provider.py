"""Provider-agnostic chat client for any OpenAI-compatible endpoint, configured only through environment.

Responses are cached on disk by (model, messages, temperature, sample index) so that re-runs do not repeat
paid calls. A hard budget stops a run. Keys are never logged. FakeProvider supports offline tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)  # the project's key must win over any older shell export


def configured_models() -> list[str]:
    return [m.strip() for m in os.environ.get("LLM_MODELS", "").split(",") if m.strip()]


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    repairs: int = 0
    parse_failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ChatProvider:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, budget_usd: Optional[float] = None,
                 cache_dir: Optional[Path] = None, usage: Optional[Usage] = None, timeout: float = 90.0):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self.budget = budget_usd if budget_usd is not None else float(os.environ.get("LLM_BUDGET_USD", "10"))
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "approval-laundering" / "llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = usage or Usage()
        self.timeout = timeout
        import certifi
        self._ssl = ssl.create_default_context(cafile=certifi.where())

    def _raw(self, model: str, messages: list[dict], temperature: float) -> dict:
        if not self.api_key:
            raise ProviderError("LLM_API_KEY is not set")
        body = json.dumps({"model": model, "messages": messages, "temperature": temperature, "max_tokens": 700,
                           "response_format": {"type": "json_object"}, "usage": {"include": True}}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "X-Title": "approval-laundering-sandbox"})
        last = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl) as r:
                    data = json.load(r)
                if "choices" in data and data["choices"]:
                    return data
                last = ProviderError(f"no choices: {str(data)[:200]}")
            except urllib.error.HTTPError as e:
                detail = e.read()[:200].decode(errors="replace")
                if e.code in (401, 402, 403):
                    raise ProviderError(f"HTTP {e.code}: {detail}") from e
                last = ProviderError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as e:
                last = ProviderError(str(e))
            time.sleep(min(30, 2 ** attempt))
        raise last

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0, sample: int = 0) -> str:
        key = hashlib.sha256(json.dumps([model, messages, temperature, sample], sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with self.usage.lock:
                self.usage.cached += 1
            return json.loads(path.read_text())["content"]
        with self.usage.lock:
            if self.usage.cost_usd >= self.budget:
                raise BudgetExceeded(f"spent ${self.usage.cost_usd:.2f} of ${self.budget:.2f}")
        data = self._raw(model, messages, temperature)
        content = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage") or {}
        with self.usage.lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += u.get("prompt_tokens", 0) or 0
            self.usage.completion_tokens += u.get("completion_tokens", 0) or 0
            self.usage.cost_usd += float(u.get("cost", 0) or 0)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"model": data.get("model", model), "content": content}))
        tmp.replace(path)
        return content


class FakeProvider:
    """Offline stand-in: a function from (model, messages) to a reply string. Output is never reported as LLM data."""

    def __init__(self, fn: Callable[[str, list[dict]], str]):
        self.fn, self.usage = fn, Usage()

    def chat(self, model, messages, temperature=0.0, sample=0) -> str:
        self.usage.calls += 1
        return self.fn(model, messages)


def extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(t[start:end + 1])


def structured(provider, model: str, messages: list[dict], schema: Type[T], temperature: float, sample: int) -> Optional[T]:
    """One call plus one repair attempt, validated strictly. None means the model never produced valid output."""
    reply = provider.chat(model, messages, temperature, sample)
    try:
        return schema.model_validate(extract_json(reply))
    except (ValueError, ValidationError) as first:
        with provider.usage.lock:
            provider.usage.repairs += 1
        fix = messages + [{"role": "assistant", "content": reply[:1500]},
                          {"role": "user", "content": f"That was not valid against the schema ({str(first)[:300]}). Reply with only the corrected JSON object."}]
        try:
            return schema.model_validate(extract_json(provider.chat(model, fix, temperature, sample)))
        except (ValueError, ValidationError):
            with provider.usage.lock:
                provider.usage.parse_failures += 1
            return None
