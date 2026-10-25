"""External model/provider integration entry points.

Provider code must not contain repository paths or persist credentials.
"""

from .client import call_vlm_json, rate_limited_error, rate_limit_sleep_seconds, resolve_vlm_model, resolve_vlm_retries, resolve_vlm_timeout, vlm_credentials_available
from .aamas import AamasProceedingsProvider
from .literature import (
    ArxivProvider,
    CrossrefProvider,
    DblpProvider,
    EuropePmcProvider,
    LiteratureHttpClient,
    ProviderPage,
    ProviderRequestError,
    SemanticScholarProvider,
    UnpaywallProvider,
    reset_literature_circuit_state,
)
from .arxiv_metadata import ArxivMetadataIndex, batch_match_exact
__all__ = [
    "AamasProceedingsProvider",
    "ArxivMetadataIndex",
    "ArxivProvider",
    "CrossrefProvider",
    "DblpProvider",
    "EuropePmcProvider",
    "LiteratureHttpClient",
    "ProviderPage",
    "ProviderRequestError",
    "SemanticScholarProvider",
    "UnpaywallProvider",
    "batch_match_exact",
    "reset_literature_circuit_state",
    "call_vlm_json",
    "rate_limited_error",
    "rate_limit_sleep_seconds",
    "resolve_vlm_model",
    "resolve_vlm_retries",
    "resolve_vlm_timeout",
    "vlm_credentials_available",
]
