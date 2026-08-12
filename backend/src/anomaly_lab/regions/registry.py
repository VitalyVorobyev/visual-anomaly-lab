"""Lazy registry for dataset-agnostic region extractors (ADR-0033)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from anomaly_lab.regions.base import RegionExtractor, RegionExtractorDescription


class UnknownRegionExtractorError(Exception):
    pass


def _identity() -> type[RegionExtractor]:
    from anomaly_lab.regions.identity import IdentityExtractor

    return IdentityExtractor


def _foreground_threshold() -> type[RegionExtractor]:
    from anomaly_lab.regions.threshold import ForegroundThresholdExtractor

    return ForegroundThresholdExtractor


def _mobile_sam() -> type[RegionExtractor]:
    from anomaly_lab.regions.mobile_sam import MobileSamRegionExtractor

    return MobileSamRegionExtractor


LOADERS: dict[str, Callable[[], type[RegionExtractor]]] = {
    "identity": _identity,
    "foreground_threshold": _foreground_threshold,
    "mobile_sam": _mobile_sam,
}


def registered_keys() -> tuple[str, ...]:
    return tuple(LOADERS)


def get_extractor_class(key: str) -> type[RegionExtractor]:
    loader = LOADERS.get(key)
    if loader is None:
        known = ", ".join(sorted(LOADERS))
        raise UnknownRegionExtractorError(
            f"no region extractor is registered under {key!r}; known extractors are {known}"
        )
    return loader()


def validate_config(key: str, config: dict[str, Any]) -> BaseModel:
    return get_extractor_class(key).config_model().model_validate(config)


def build(
    key: str, config: dict[str, Any], *, assets: dict[str, Path] | None = None
) -> RegionExtractor:
    extractor_type = get_extractor_class(key)
    return extractor_type(extractor_type.config_model().model_validate(config), assets=assets)


def describe(key: str) -> RegionExtractorDescription:
    extractor_type = get_extractor_class(key)
    return RegionExtractorDescription(
        key=key,
        title=extractor_type.title,
        summary=extractor_type.summary,
        availability=extractor_type.availability(),
        required_assets=list(extractor_type.required_assets),
        config_schema=extractor_type.config_model().model_json_schema(),
    )


def describe_all() -> list[RegionExtractorDescription]:
    return [describe(key) for key in LOADERS]
