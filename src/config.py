"""Application configuration for the BPMN-Text Conformance Checker.

Loads settings from environment variables (typically via a ``.env`` file) using
``python-dotenv``.  All other modules should import constants from here instead
of calling ``os.getenv`` directly.

Usage::

    from src.config import settings

    llm = ChatGoogleGenerativeAI(model=settings.gemini_model, ...)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Bootstrap logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Load .env from the project root ───────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)
    logger.debug("Loaded environment variables from %s", _ENV_FILE)
else:
    logger.warning(
        ".env file not found at %s — make sure GOOGLE_API_KEY is set in the environment.",
        _ENV_FILE,
    )


# ── Settings ──────────────────────────────────────────────────────────────────


class Settings:
    """Central configuration object.

    All values are read from environment variables at import time.
    Override any value by setting the corresponding variable before starting
    the process or by editing ``.env``.

    Attributes:
        google_api_key: Google Generative AI API key (required).
        gemini_model: Gemini model identifier used by all LLM agents.
        gemini_temperature: Sampling temperature passed to Gemini.
        llm_max_retries: Maximum number of retries on unparseable LLM output.
        data_dir: Absolute path to the ``data/`` directory.
        log_level: Python logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """

    # ── LLM ───────────────────────────────────────────────────────────────────
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))

    # ── Retry policy for LLM calls ────────────────────────────────────────────
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # ── File-system paths ─────────────────────────────────────────────────────
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        """Validate that required settings are present.

        Raises:
            EnvironmentError: If ``GOOGLE_API_KEY`` is not set.
        """
        if not self.google_api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Add it to your .env file or export it in your shell."
            )

    def __repr__(self) -> str:
        """Return a safe string representation (API key is masked)."""
        masked_key = (
            f"{self.google_api_key[:6]}…" if len(self.google_api_key) > 6 else "***"
        )
        return (
            f"Settings("
            f"gemini_model={self.gemini_model!r}, "
            f"gemini_temperature={self.gemini_temperature}, "
            f"llm_max_retries={self.llm_max_retries}, "
            f"google_api_key={masked_key!r}, "
            f"data_dir={self.data_dir}"
            f")"
        )


# ── Module-level singleton ─────────────────────────────────────────────────────
settings = Settings()

# Apply dynamic log level from config
logging.getLogger().setLevel(settings.log_level.upper())
