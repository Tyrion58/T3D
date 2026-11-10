import sys
import types
import importlib
import importlib.abc
import importlib.machinery

# ---------------------------------------------------------------------------
# Workaround for broken modeling_sdar.py in upstream HuggingFace repos
# JetLM/SDAR-{1.7B,4B}-Chat (commit from 2026-02-13).
#
# The updated modeling_sdar.py:
#   - imports fused_linear_diffusion_cross_entropy.py which was never uploaded
#   - uses TransformersKwargs / auto_docstring / can_return_tuple that require
#     transformers >= 4.53, but the project pins 4.52.4
#
# Fix: pin these repos to the last compatible revision so the old (working)
# modeling_sdar.py is downloaded instead.  The pin is applied automatically
# to every AutoModel / AutoTokenizer.from_pretrained call.
# ---------------------------------------------------------------------------

# repo_id -> last revision *before* the breaking Feb-13 update
_PINNED_REVISIONS = {
    "JetLM/SDAR-4B-Chat": "d8a00c3e1faa",
    "JetLM/SDAR-1.7B-Chat": "e2fcc4b23a0f",
}


def _patch_from_pretrained():
    """Inject ``revision=`` into from_pretrained for repos with broken HEAD."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return

    for cls in (AutoModelForCausalLM, AutoTokenizer):
        _orig = cls.from_pretrained.__func__  # unwrap classmethod

        def _make_wrapper(orig_fn):
            def wrapper(klass, pretrained_model_name_or_path, *args, **kwargs):
                name = str(pretrained_model_name_or_path)
                if name in _PINNED_REVISIONS and "revision" not in kwargs:
                    kwargs["revision"] = _PINNED_REVISIONS[name]
                return orig_fn(klass, pretrained_model_name_or_path, *args, **kwargs)
            return classmethod(wrapper)

        cls.from_pretrained = _make_wrapper(_orig)


_patch_from_pretrained()


# ---------------------------------------------------------------------------
# Patch LossKwargs into transformers.utils if missing.
# The cached modeling_sdar.py imports LossKwargs which doesn't exist in
# some transformers versions.  Injecting a stub prevents ImportError.
# ---------------------------------------------------------------------------
def _patch_loss_kwargs():
    import transformers.utils as _tu
    if not hasattr(_tu, "LossKwargs"):
        from typing import TypedDict
        class LossKwargs(TypedDict, total=False):
            pass
        _tu.LossKwargs = LossKwargs

_patch_loss_kwargs()
