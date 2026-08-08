"""Environment settings and interest-profile loading."""

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from modelscout.agents.schemas import InterestProfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = PROJECT_ROOT / "config" / "interest_profiles"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    anthropic_api_key: str = ""
    database_url: str = "postgresql://modelscout:modelscout_dev_only@localhost:5433/modelscout"
    anthropic_extractor_model: str = "claude-sonnet-5"

    # Opt-in cost estimation for observability.py -- unset by default rather
    # than hardcoding pricing figures that go stale. Set both to enable.
    price_per_mtok_input: float | None = None
    price_per_mtok_output: float | None = None


def load_interest_profile(name_or_path: str) -> InterestProfile:
    """Load an interest profile by name (looked up in config/interest_profiles/)
    or by an explicit path."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        path = candidate
    else:
        path = PROFILES_DIR / f"{name_or_path}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Interest profile not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return InterestProfile.model_validate(raw)


settings = Settings()
