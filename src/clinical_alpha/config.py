"""Central configuration with pydantic validation."""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from clinical_alpha.exceptions import ConfigError


class Settings(BaseSettings):
    """Hierarchical configuration for clinical-alpha.

    All values can be overridden via environment variables or .env file.
    """

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── API Keys ──────────────────────────────────────────────────
    polygon_api_key: str = Field(default="", description="Polygon.io API key (optional)")

    # ── Data Directories ──────────────────────────────────────────
    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    sample_dir: Path = Field(default=Path("data/samples"))
    report_dir: Path = Field(default=Path("reports"))
    table_dir: Path = Field(default=Path("reports/tables"))
    figure_dir: Path = Field(default=Path("reports/figures"))

    # ── Fetch Limits ──────────────────────────────────────────────
    max_trial_fetch: int = Field(default=5000, ge=1, le=100_000)
    max_fda_fetch: int = Field(default=1000, ge=1, le=50_000)

    # ── Logging ────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = Field(default=None)

    # ── Event Processing ──────────────────────────────────────────
    event_lookback_days: int = Field(default=90, ge=0, le=730)

    # ── Abnormal Return Estimation ────────────────────────────────
    estimation_window: int = Field(default=252, ge=20, le=756)
    event_window_before: int = Field(default=10, ge=0, le=120)
    event_window_after: int = Field(default=20, ge=1, le=252)
    event_windows: list[tuple[int, int]] = Field(
        default=[(-1, 1), (-3, 3), (-5, 10), (-10, 20), (-20, 40)],
        description="Multiple event windows to test",
    )

    # ── Factor Model ──────────────────────────────────────────────
    factor_models: list[str] = Field(
        default=["spy_adjusted", "xlv_adjusted", "xbi_adjusted", "market_model", "ff3"],
    )

    # ── Backtest ──────────────────────────────────────────────────
    backtest_top_k: int = Field(default=5, ge=1, le=50)
    backtest_holding_periods: list[int] = Field(default=[5, 10, 21, 63])
    backtest_transaction_cost_model: str = Field(
        default="linear", description="linear | quadratic | volume_based"
    )
    backtest_base_tc_bps: float = Field(default=10.0, ge=0, le=500)
    backtest_slippage_bps: float = Field(default=5.0, ge=0, le=200)
    backtest_weighting: str = Field(
        default="equal",
        description="equal | value | volatility_target | risk_parity | min_variance",
    )
    backtest_vol_target: float = Field(default=0.15, ge=0.01, le=0.50)
    backtest_max_leverage: float = Field(default=1.0, ge=1.0, le=5.0)
    backtest_max_turnover: float = Field(default=0.50, ge=0.0, le=1.0)
    backtest_sector_neutral: bool = Field(default=False)
    backtest_long_short: bool = Field(default=False)
    backtest_short_rebate_rate: float = Field(default=0.003, ge=0.0, le=0.10)
    backtest_short_fee_bps: float = Field(default=35.0, ge=0.0, le=500.0)

    # ── Robustness ────────────────────────────────────────────────
    default_event_types: list[str] = Field(
        default=["fda_approval", "trial_result"],
    )
    default_confidence_thresholds: list[float] = Field(
        default=[0.6, 0.7, 0.8, 0.9, 0.95],
    )
    default_overlap_days_options: list[int] = Field(default=[0, 30, 60, 90])

    # ── Statistical Tests ─────────────────────────────────────────
    bootstrap_n_iterations: int = Field(default=10_000, ge=100, le=1_000_000)
    permutation_n_iterations: int = Field(default=10_000, ge=100, le=1_000_000)
    alpha_level: float = Field(default=0.05, ge=0.001, le=0.20)

    # ── Risk Model ────────────────────────────────────────────────
    risk_lookback: int = Field(default=252, ge=20, le=756)
    risk_shrinkage_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_n_factors: int = Field(default=5, ge=1, le=20)

    # ── Data Quality ──────────────────────────────────────────────
    min_observations_pct: float = Field(default=0.80, ge=0.0, le=1.0)
    max_return_zscore: float = Field(default=6.0, ge=1.0, le=20.0)
    min_price: float = Field(default=1.0, ge=0.01, le=100.0)
    max_missing_consecutive_days: int = Field(default=10, ge=0, le=252)

    signal_min_obs: int = Field(default=10, ge=5, le=1000)
    signal_n_quantiles: int = Field(default=5, ge=2, le=20)
    signal_max_lag: int = Field(default=63, ge=1, le=252)
    signal_method: str = Field(default="pearson")

    n_bootstrap_fdr: int = Field(default=1000, ge=100, le=100_000)
    structural_break_min_segment: int = Field(default=20, ge=10, le=252)
    bayes_prior_scale: float = Field(default=0.5, ge=0.01, le=5.0)

    black_litterman_tau: float = Field(default=0.05, ge=0.001, le=1.0)
    black_litterman_risk_aversion: float = Field(default=2.5, ge=0.1, le=20.0)

    @field_validator("backtest_holding_periods")
    @classmethod
    def check_holding_periods(cls, v: list[int]) -> list[int]:
        for hp in v:
            if hp <= 0:
                raise ConfigError(f"Holding period must be positive: {hp}")
        return v

    @field_validator("event_windows")
    @classmethod
    def check_event_windows(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        for start, end in v:
            if start >= end:
                raise ConfigError(f"Event window start must be before end: ({start}, {end})")
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.sample_dir,
            self.report_dir,
            self.table_dir,
            self.figure_dir,
            self.report_dir / "tables",
            self.report_dir / "figures",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def event_window_default(self) -> tuple[int, int]:
        return (self.event_window_before, self.event_window_after)


settings = Settings()
