"""Runtime configuration checks that never expose secret values."""

from __future__ import annotations

import os


def openai_api_key_configured() -> bool:
    """Return whether the OpenAI SDK can authenticate from the process environment."""

    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def require_openai_api_key() -> None:
    """Fail clearly before a command starts billable OpenAI work."""

    if not openai_api_key_configured():
        raise RuntimeError(
            "OPENAI_API_KEY is not set in this process. Export it before running "
            "OpenAI-backed embedding or generation commands."
        )
