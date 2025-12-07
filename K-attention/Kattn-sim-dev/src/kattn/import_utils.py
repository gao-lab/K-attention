import importlib.util
import importlib.metadata
from logging import getLogger
from packaging import version

logger = getLogger(__name__)


"""
Modified from transformers: utils/import_utils.py
"""
def _is_package_available(pkg_name: str) -> bool:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    return package_exists


def is_torch_available():
    return _is_package_available("torch")

def is_flash_attn_2_available():
    if not is_torch_available():
        return False

    # TODO, parse version
    if not _is_package_available("flash_attn"):
        return False

    # Let's add an extra check to see if cuda is available
    import torch

    if not torch.cuda.is_available():
        return False

    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    else:
        return False
