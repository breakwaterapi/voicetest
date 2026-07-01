import voicetest
from voicetest.config import Settings


def test_import():
    assert voicetest is not None


def test_settings_no_env(monkeypatch):
    # Clear any env vars that might be set in the environment
    env_vars = [
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
        "VERIFIED_DEST_NUMBERS", "PUBLIC_WS_URL", "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "LLM_CALLER_MODEL", "LLM_JUDGE_MODEL", "VOICETEST_DB_URL",
        "VOICETEST_BLOB_DIR",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    s = Settings(_env_file=None)
    assert s.voicetest_db_url == "sqlite:///voicetest.db"
    assert s.voicetest_blob_dir == "./recordings"
    assert s.twilio_account_sid is None
