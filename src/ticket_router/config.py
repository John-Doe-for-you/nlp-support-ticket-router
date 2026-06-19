"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = "sqlite:///./tickets.db"

    category_model_path: str = "artifacts/category_model.joblib"
    tfidf_vectorizer_path: str = "artifacts/tfidf_vectorizer.joblib"
    label_encoder_path: str = "artifacts/label_encoder.joblib"


settings = Settings()
