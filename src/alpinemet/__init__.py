"""alpinemet: multi-scale weather and climate data evaluation for Alpine renewable energy.

Companion code for Schicker et al. (2026), "Beyond Resolution", iScience.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("alpinemet")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
