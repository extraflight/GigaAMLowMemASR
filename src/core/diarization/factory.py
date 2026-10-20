"""Lazy diarization backend selection — ONNX only."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def create_diarization_backend(
    backend: str,
    *,
    hf_token: str | None = None,
    device: str = "auto",
    provider: str = "auto",
    model_dir: str | None = None,
    legacy_factory: Callable[..., Any] | None = None,
    **pyannote_options: Any,
) -> Any:
    """Создать backend диаризации; тяжёлые зависимости импортируются лениво.

    Поддерживается только ONNX-пайплайн (pyannote-segmentation-3.0 + WeSpeaker).
    Не требует PyTorch и HF-токена.
    """
    if backend != "onnx":
        if legacy_factory is not None:
            return legacy_factory(backend=backend, hf_token=hf_token, device=device)
        raise ValueError(f"Диаризация '{backend}' недоступна: поддерживается только 'onnx'")

    from .onnx_backend import OnnxDiarizationBackend  # noqa: PLC0415

    kwargs: dict[str, Any] = {"provider": provider}
    if model_dir is not None:
        kwargs["model_dir"] = model_dir
    return OnnxDiarizationBackend(**kwargs)
