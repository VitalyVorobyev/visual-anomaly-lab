"""visual-anomaly-lab backend package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("anomaly-lab")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
