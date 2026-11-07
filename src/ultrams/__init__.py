"""UltraMS spectrum encoder."""

from importlib.metadata import PackageNotFoundError, version

from .inference import UltraMS, SpectrumEmbedding
from .finetune import UltraMSPredictor
from .spectra import read_mgf, read_mzml, read_spectra

try:
    __version__ = version("ultrams")
except PackageNotFoundError:  # source checkout without installation metadata
    __version__ = "0.3.1"

__all__ = [
    "UltraMS",
    "SpectrumEmbedding",
    "UltraMSPredictor",
    "read_mgf",
    "read_mzml",
    "read_spectra",
    "__version__",
]
