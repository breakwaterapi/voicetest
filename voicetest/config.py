from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telephony
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_from_number: str | None = None
    verified_dest_numbers: str | None = None  # comma-separated allowlist
    public_ws_url: str | None = None

    # STT / TTS
    deepgram_api_key: str | None = None
    cartesia_api_key: str | None = None

    # LLMs
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_caller_model: str | None = None
    llm_judge_model: str | None = None

    # App
    voicetest_db_url: str = "sqlite:///voicetest.db"
    voicetest_blob_dir: str = "./recordings"

    @property
    def verified_dest_list(self) -> list[str]:
        if not self.verified_dest_numbers:
            return []
        return [
            n.strip().replace("-", "").replace(" ", "")
            for n in self.verified_dest_numbers.split(",")
            if n.strip()
        ]
