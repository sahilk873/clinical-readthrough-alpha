"""Custom exception hierarchy for clinical-alpha."""


class ClinicalAlphaError(Exception):
    """Base exception for all clinical-alpha errors."""


class ConfigError(ClinicalAlphaError):
    """Invalid or missing configuration."""


class DataError(ClinicalAlphaError):
    """Data fetching, validation, or processing error."""


class DataQualityError(DataError):
    """Data quality check failure."""


class PipelineError(ClinicalAlphaError):
    """Pipeline execution error."""


class GraphError(ClinicalAlphaError):
    """Graph construction or query error."""


class EventError(ClinicalAlphaError):
    """Event extraction or processing error."""


class BacktestError(ClinicalAlphaError):
    """Backtest simulation error."""


class RiskError(ClinicalAlphaError):
    """Risk model computation error."""


class StatisticalTestError(ClinicalAlphaError):
    """Statistical test computation error."""


class PriceError(DataError):
    """Price data fetching or processing error."""


class SignalError(ClinicalAlphaError):
    """Signal computation error."""


class FactorModelError(ClinicalAlphaError):
    """Factor model estimation error."""


class AttributionError(ClinicalAlphaError):
    """Performance attribution error."""
