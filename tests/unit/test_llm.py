"""Tests for LLM clients with mocked SDKs (no network calls).

Each test injects a fake `client` whose `messages.create` /
`chat.completions.create` returns a canned response object. We verify the
client correctly:
  - sends documents + system prompt + cache control as expected
  - parses the JSON decision payload
  - extracts citations (Anthropic) or skips citation (OpenAI-compatible)
  - reports usage tokens accurately
"""
from __future__ import annotations

import json
import types

import pytest

from compliance_extractor.llm.anthropic_client import AnthropicClient
from compliance_extractor.llm.base import ExtractionRequest, SourceDocument
from compliance_extractor.llm.openai_compatible import OpenAICompatibleClient
from compliance_extractor.llm.parsing import extract_json_object, parse_extraction_payload
from compliance_extractor.llm.registry import build_client, list_providers


def _mk_request(doc_text: str = "Multi-factor authentication is required for administrators.") -> ExtractionRequest:
    return ExtractionRequest(
        control_id="A.5.15",
        control_title="Access Control",
        control_description="The organization restricts access via authentication controls.",
        documents=[SourceDocument(doc_id="policy.txt", text=doc_text, title="Policy")],
        system_prompt="ignored — client uses prompts.SYSTEM_PROMPT",
    )


@pytest.mark.fast
def test_extract_json_object_handles_markdown_fence() -> None:
    text = '```json\n{"decision": "complies", "evidence_quote": "x", "evidence_quote_source": "p", "rationale": "r"}\n```'
    obj = extract_json_object(text)
    assert obj["decision"] == "complies"


@pytest.mark.fast
def test_extract_json_object_recovers_from_preamble() -> None:
    text = 'Sure, here is my answer: {"decision": "no-evidence", "evidence_quote": "", "evidence_quote_source": "", "rationale": "n/a"}'
    obj = extract_json_object(text)
    assert obj["decision"] == "no-evidence"


@pytest.mark.fast
def test_parse_extraction_payload_validates_decision() -> None:
    bad = '{"decision": "yolo", "evidence_quote": "", "evidence_quote_source": "", "rationale": ""}'
    with pytest.raises(ValueError, match="Invalid decision"):
        parse_extraction_payload(bad)


def _fake_anthropic_response(text: str, citation: dict | None = None, usage: dict | None = None):
    text_block = types.SimpleNamespace(
        type="text",
        text=text,
        citations=[types.SimpleNamespace(**citation)] if citation else [],
    )
    usage_obj = types.SimpleNamespace(**(usage or {"input_tokens": 100, "output_tokens": 30}))
    return types.SimpleNamespace(content=[text_block], usage=usage_obj, stop_reason="end_turn")


@pytest.mark.fast
def test_anthropic_client_parses_response_and_citation(tmp_path, monkeypatch) -> None:
    payload = {
        "decision": "complies",
        "evidence_quote": "Multi-factor authentication is required for administrators.",
        "evidence_quote_source": "policy.txt",
        "rationale": "MFA is explicitly required.",
    }
    citation = {
        "type": "char_location",
        "document_index": 0,
        "start_char_index": 0,
        "end_char_index": 58,
        "cited_text": "Multi-factor authentication is required for administrators.",
    }
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_anthropic_response(json.dumps(payload), citation=citation)

    fake_client = types.SimpleNamespace(messages=FakeMessages())
    monkeypatch.setenv("HOME", str(tmp_path))
    client = AnthropicClient(model="claude-opus-4-7", client=fake_client, use_cache=False)
    resp = client.extract(_mk_request())

    assert resp.decision == "complies"
    assert resp.evidence_quote.startswith("Multi-factor")
    assert resp.evidence_quote_source == "policy.txt"
    assert resp.citation is not None
    assert resp.citation.char_start == 0 and resp.citation.char_end == 58
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 30
    assert resp.provider == "anthropic"

    docs = captured["messages"][0]["content"]
    document_blocks = [b for b in docs if b["type"] == "document"]
    assert len(document_blocks) == 1
    assert document_blocks[0]["citations"] == {"enabled": True}
    assert document_blocks[0].get("cache_control") == {"type": "ephemeral"}


@pytest.mark.fast
def test_openai_compatible_client_parses_response(tmp_path, monkeypatch) -> None:
    payload = {
        "decision": "maybe-complies",
        "evidence_quote": "Backups are taken nightly.",
        "evidence_quote_source": "policy.txt",
        "rationale": "Mentions backups but not encryption.",
    }
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content=json.dumps(payload)),
                finish_reason="stop",
            )
            usage = types.SimpleNamespace(prompt_tokens=200, completion_tokens=42)
            return types.SimpleNamespace(choices=[choice], usage=usage)

    fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setenv("HOME", str(tmp_path))
    client = OpenAICompatibleClient(provider="openai", model="gpt-4o-mini", client=fake_client, use_cache=False)
    resp = client.extract(_mk_request())

    assert resp.decision == "maybe-complies"
    assert resp.evidence_quote == "Backups are taken nightly."
    assert resp.citation is None
    assert resp.usage.input_tokens == 200 and resp.usage.output_tokens == 42
    assert resp.provider == "openai"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == "gpt-4o-mini"


@pytest.mark.fast
def test_openai_compatible_client_ollama_skips_json_mode(monkeypatch, tmp_path) -> None:
    payload = {"decision": "no-evidence", "evidence_quote": "", "evidence_quote_source": "", "rationale": "nothing"}
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content=json.dumps(payload)),
                finish_reason="stop",
            )
            usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
            return types.SimpleNamespace(choices=[choice], usage=usage)

    fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setenv("HOME", str(tmp_path))
    client = OpenAICompatibleClient(provider="ollama", client=fake_client, use_cache=False)
    resp = client.extract(_mk_request())

    assert resp.decision == "no-evidence"
    assert "response_format" not in captured
    assert client.base_url == "http://localhost:11434/v1"


@pytest.mark.fast
def test_registry_dispatch() -> None:
    assert set(list_providers()) == {"anthropic", "openai", "deepseek", "mistral", "ollama"}
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_client("unknown")
