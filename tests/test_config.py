from __future__ import annotations

import pytest

from llmqa.config import openai_api_key_configured, require_openai_api_key


def test_openai_key_configuration_checks_presence_without_returning_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-value")

    assert openai_api_key_configured() is True
    assert require_openai_api_key() is None


def test_openai_key_configuration_rejects_missing_or_blank_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert openai_api_key_configured() is False
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        require_openai_api_key()

    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert openai_api_key_configured() is False
