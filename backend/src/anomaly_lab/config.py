"""Application settings.

This module is the **single** place permitted to derive a filesystem path from its own
location. Every other module receives paths from a `Settings` instance — no module builds
a path from `__file__` or the current working directory (see `docs/architecture/repository.md`).

All settings are read from the environment with the `ANOMALY_LAB_` prefix, so the same
package runs identically under `uv run`, under pytest, and as a Tauri sidecar.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/anomaly_lab/config.py -> backend/src/anomaly_lab -> backend/src -> backend -> repo
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_data_dir() -> Path:
    """Repo-local `data/` (ADR-0002, ADR-0004).

    Only evaluated when `ANOMALY_LAB_DATA_DIR` is unset. A packaged build (M7) does not
    live in a source checkout, so it must set that variable explicitly rather than rely
    on this default — hence the marker check instead of a silent wrong answer.
    """
    if not (_REPO_ROOT / "docs").is_dir():
        msg = (
            f"Cannot infer the repository root from {__file__} (looked at {_REPO_ROOT}). "
            "Set ANOMALY_LAB_DATA_DIR to choose the data directory explicitly."
        )
        raise RuntimeError(msg)
    return _REPO_ROOT / "data"


class Settings(BaseSettings):
    """Runtime configuration. Immutable for the lifetime of a process."""

    model_config = SettingsConfigDict(env_prefix="ANOMALY_LAB_", frozen=True)

    data_dir: Path = Field(default_factory=_default_data_dir)
    """Root of all app-managed state. Gitignored; safe to delete to reset."""

    reference_datasets_dir: Path = _REPO_ROOT / "datasets"
    """Optional local public benchmark packs. Their files remain read-only."""

    host: str = "127.0.0.1"
    """Loopback only (§11). Never bind 0.0.0.0 — there is no authentication."""

    port: int = 8000
    """0 asks the OS for an ephemeral port; the chosen port is announced on stdout."""

    parent_pid: int | None = None
    """When set, the sidecar self-terminates once this process is no longer its parent."""

    dev_cors: bool = True
    """Permit browser and Tauri WebView origins. Development affordance only."""

    @field_validator("data_dir", "reference_datasets_dir")
    @classmethod
    def _absolute_data_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.sqlite3"

    @property
    def manifests_dir(self) -> Path:
        """Committed import manifests — the reproducibility record (ADR-0006)."""
        return self.data_dir / "manifests"

    @property
    def thumbnails_dir(self) -> Path:
        """Rendered image tiers, keyed by image id (§9)."""
        return self.data_dir / "thumbnails"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def annotations_dir(self) -> Path:
        """Materialised, app-owned annotation revisions in source-image coordinates."""
        return self.data_dir / "annotations"

    def annotation_image_dir(self, image_id: int) -> Path:
        return self.annotations_dir / f"image-{image_id}"

    @property
    def jobs_log_dir(self) -> Path:
        """Logs for jobs bound to no experiment — import, verify, prewarm (§6)."""
        return self.data_dir / "jobs" / "logs"

    @property
    def model_cache_dir(self) -> Path:
        """Downloaded assets shared across experiments — pretrained weights, penalty sets.

        Deliberately outside `artifacts/`: a 1.5 GB ImageNette copy per experiment would
        be absurd, and these files belong to the method rather than to any one run.
        Safe to delete; the next training run re-downloads what it needs.
        """
        return self.data_dir / "model-cache"

    @property
    def model_assets_dir(self) -> Path:
        """Licensed, integrity-checked assets used by interactive model tools."""
        return self.model_cache_dir / "assets"

    @property
    def region_profiles_dir(self) -> Path:
        """App-owned prepared pixels and transforms, keyed by immutable profile revision."""
        return self.data_dir / "region-profiles"

    def region_profile_dir(self, profile_id: int) -> Path:
        return self.region_profiles_dir / f"profile-{profile_id}"

    def experiment_dir(self, experiment_id: int) -> Path:
        return self.artifacts_dir / f"exp-{experiment_id}"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Call `get_settings.cache_clear()` in tests."""
    return Settings()
