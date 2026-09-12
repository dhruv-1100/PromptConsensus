"""Make the backend package importable no matter where pytest is invoked from."""
import os
import sys

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


import pytest


@pytest.fixture(autouse=True)
def no_network_calls(monkeypatch):
    """
    Fail loudly instead of calling OpenRouter.

    backend/.env holds a real key, so a test that accidentally reaches a live
    agent (for example by falling back to DEFAULT_REWRITER_SPECS) would spend
    credits and write to the repo's data files. Tests must stub their agents.
    """
    def blocked(*args, **kwargs):
        raise AssertionError(
            "A test tried to call OpenRouter. Stub the agent functions instead "
            "of invoking the live pipeline."
        )

    import live_mode_utils

    monkeypatch.setattr(live_mode_utils, "invoke_openrouter_model", blocked)
    for module in ("agents.intent_extractor", "agents.rewriter_a", "agents.rewriter_b",
                   "agents.rewriter_c", "agents.council", "pipeline.graph"):
        monkeypatch.setattr(f"{module}.invoke_openrouter_model", blocked, raising=False)
