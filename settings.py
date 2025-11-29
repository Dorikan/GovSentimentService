from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    """Настройки приложения"""

    LLM_NAME: str = "qwen/qwen3-235b-a22b:free"
    OPENROUTER_API_KEY: str
    BASE_URL: str = "https://openrouter.ai/api/v1"

    BATCH_SIZE: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
