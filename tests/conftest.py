import pytest


@pytest.fixture(autouse=True)
def _claude_api_on_for_tests(monkeypatch):
    """Claude API usage is off in production (USE_CLAUDE_API); the tests
    exercise the handicapper and news code with fake clients, so turn the
    gates on for them. test_api_off_by_default checks the real default."""
    import cfb_handicap
    import cfb_news
    monkeypatch.setattr(cfb_news, "API_ON", True)
    monkeypatch.setattr(cfb_handicap, "HANDICAP", True)
