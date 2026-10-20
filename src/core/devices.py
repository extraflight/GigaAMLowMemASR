"""PyTorch device selection helpers."""

from __future__ import annotations


def best_torch_device(torch_module=None) -> str:
    """Return the best available PyTorch device string."""
    if torch_module is None:
        try:
            import torch  # noqa: F811
        except ImportError:
            return "cpu"
    else:
        torch = torch_module

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_accelerator_cache() -> None:
    """Clear accelerator caches to free memory."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
