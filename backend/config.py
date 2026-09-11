from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=".env", extra="ignore", env_ignore_empty=True)
    model_dir: Path = Path("models")
    history_db: Path = Path("data/prediction_history.db")
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_bytes: int = Field(5 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    max_batch_rows: int = Field(10000, ge=1, le=50000)
    risk_medium: float = Field(.3, gt=0, lt=1)
    risk_high: float = Field(.7, gt=0, lt=1)
    default_threshold: float | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def ordered_risk_bands(self):
        if self.risk_medium >= self.risk_high:
            raise ValueError("Risk medium boundary must be below high boundary.")
        return self
