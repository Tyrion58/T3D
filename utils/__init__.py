"""Shared utilities for configuration, paths, and compatibility helpers.

Compatibility patches are intentionally not applied on package import. Import
``utils.hf_compat`` explicitly from entry points that load SDAR remote code.
"""

from .paths import REPO_ROOT
