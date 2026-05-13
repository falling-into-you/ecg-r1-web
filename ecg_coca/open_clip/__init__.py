"""Minimal OpenCLIP package init for ECG-R1 runtime imports."""

from .factory import create_model_and_transforms, get_model_config, load_checkpoint
from .model import get_cast_dtype, get_input_dtype
