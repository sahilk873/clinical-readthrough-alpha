import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration for clinical-alpha."""

    # API keys
    POLYGON_API_KEY: Optional[str] = os.getenv("POLYGON_API_KEY", "")

    # Data directories
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    RAW_DIR: Path = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
    PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    SAMPLE_DIR: Path = Path(os.getenv("SAMPLE_DATA_DIR", "data/samples"))
    REPORT_DIR: Path = Path(os.getenv("REPORT_DIR", "reports"))
    TABLE_DIR: Path = Path(os.getenv("TABLE_DIR", "reports/tables"))
    FIGURE_DIR: Path = Path(os.getenv("FIGURE_DIR", "reports/figures"))

    # Fetch limits
    MAX_TRIAL_FETCH: int = int(os.getenv("MAX_TRIAL_FETCH", "5000"))
    MAX_FDA_FETCH: int = int(os.getenv("MAX_FDA_FETCH", "1000"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Event processing
    EVENT_LOOKBACK_DAYS: int = int(os.getenv("EVENT_LOOKBACK_DAYS", "90"))

    # Default estimation window for abnormal returns (trading days)
    ESTIMATION_WINDOW: int = 252
    EVENT_WINDOW_BEFORE: int = 10
    EVENT_WINDOW_AFTER: int = 20

    # Backtest defaults
    BACKTEST_TOP_K: int = 5
    BACKTEST_HOLDING_PERIODS: list[int] | None = None
    BACKTEST_TRANSACTION_COST: float = 0.0010

    # Robustness defaults
    DEFAULT_EVENT_TYPES: list[str] | None = None
    DEFAULT_CONFIDENCE_THRESHOLDS: list[float] | None = None
    DEFAULT_OVERLAP_DAYS: int = 30

    def __init__(self):
        self._ensure_dirs()
        if self.BACKTEST_HOLDING_PERIODS is None:
            self.BACKTEST_HOLDING_PERIODS = [5, 10, 21, 63]
        if self.DEFAULT_EVENT_TYPES is None:
            self.DEFAULT_EVENT_TYPES = ["fda_approval", "fda_rejection", "trial_result"]
        if self.DEFAULT_CONFIDENCE_THRESHOLDS is None:
            self.DEFAULT_CONFIDENCE_THRESHOLDS = [0.7, 0.8, 0.9]

    def _ensure_dirs(self):
        for d in [
            self.DATA_DIR,
            self.RAW_DIR,
            self.PROCESSED_DIR,
            self.SAMPLE_DIR,
            self.REPORT_DIR,
            self.TABLE_DIR,
            self.FIGURE_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)
