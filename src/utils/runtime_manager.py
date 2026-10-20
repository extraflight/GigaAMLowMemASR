"""Менеджер кэша моделей HuggingFace (ONNX-only).

Не импортирует torch и может безопасно вызываться на любом этапе старта.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .atomic_json import save_json_atomic

# ── Пути кэша ────────────────────────────────────────────────────────────────


def base_dir() -> Path:
    """Корневая папка кэша приложения."""
    override = os.environ.get("GIGAAM_RUNTIME_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path("C:/GigaAMGUICash")
    if sys.platform == "darwin":
        return Path(os.path.expanduser("~/Library/Caches/GigaAMGUICash"))
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(xdg) / "GigaAMGUICash"


def hf_cache_dir() -> Path:
    """Папка кэша моделей HuggingFace."""
    override = os.environ.get("HF_HOME")
    if override:
        return Path(os.path.expanduser(override))
    return base_dir() / "hf"


def bundled_hf_cache_dir(frozen: bool | None = None) -> Path | None:
    """Кэш моделей, положенный рядом со сборкой в офлайн-варианте."""
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    if not frozen:
        return None

    executable = Path(sys.executable).resolve()
    candidates = [executable.parent]
    if sys.platform == "darwin":
        parents = executable.parents
        if len(parents) > 3 and parents[2].name.endswith(".app"):
            candidates.extend((parents[3], parents[1] / "Resources"))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass))

    for candidate in candidates:
        hub = candidate / "models" / "hf" / "hub"
        if hub.is_dir() and any(hub.iterdir()):
            return hub.parent
    return None
