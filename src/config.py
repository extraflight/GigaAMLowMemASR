"""
Конфигурация CLI-транскрибатора GigaAM v3 (ONNX-only)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .core.runtime_options import (
    parse_bool,
    validate_backend_name,
    validate_onnx_provider,
)
from .data_paths import bootstrap_data_dir

# Режим Docker — без записи файлов на диск
DOCKER_MODE = os.getenv("DOCKER_MODE", "1").lower() in ("1", "true", "yes")

APP_CONFIG_DIR_NAME = "GigaAMTranscriber"


def _validate_onnx_quantization(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized != "int8":
        raise ValueError(f"Unsupported ONNX quantization: {normalized}")
    return normalized


def user_config_dir() -> Path:
    """Persistent per-user config directory."""
    override = os.environ.get("GIGAAM_CONFIG_DIR")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_CONFIG_DIR_NAME
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_CONFIG_DIR_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / APP_CONFIG_DIR_NAME


def user_env_path() -> Path:
    return user_config_dir() / ".env"


def project_env_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".env"


# HuggingFace кэш
BUNDLED_MODELS_DIR = None


def _setup_huggingface_cache():
    global BUNDLED_MODELS_DIR
    try:
        from .utils.runtime_manager import bundled_hf_cache_dir, hf_cache_dir

        BUNDLED_MODELS_DIR = bundled_hf_cache_dir()
        hf_dir = hf_cache_dir()
    except Exception:
        BUNDLED_MODELS_DIR = None
        hf_dir = Path.home() / ".cache" / "huggingface"

    if 'HF_HOME' not in os.environ:
        try:
            hf_dir.mkdir(parents=True, exist_ok=True)
            os.environ['HF_HOME'] = str(hf_dir)
        except OSError as e:
            print(f"Не удалось создать директорию кэша HuggingFace: {e}")


# Загрузка переменных из .env файла
def load_env():
    if DOCKER_MODE:
        # В Docker .env не читаем — всё через ENV переменные
        return
    for env_path in (user_env_path(), project_env_path()):
        if env_path.exists():
            load_dotenv(env_path, override=False)


load_env()
if not DOCKER_MODE:
    bootstrap_data_dir()
_setup_huggingface_cache()

# HuggingFace токен (для ONNX-диаризации не нужен)
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN and HF_TOKEN.startswith("hf_"):
    os.environ["HF_TOKEN"] = HF_TOKEN

# Настройки модели
MODEL_NAME = os.getenv("MODEL_NAME", "ai-sage/GigaAM-v3")
MODEL_REVISION = os.getenv("MODEL_REVISION", "e2e_rnnt")

# ASR backend — ONNX (PyTorch/MLX удалены)
ASR_BACKEND = validate_backend_name(os.getenv("ASR_BACKEND", "onnx"))
ASR_MODEL = os.getenv("ASR_MODEL", MODEL_REVISION)
ASR_SEGMENTATION_MODE = os.getenv("ASR_SEGMENTATION_MODE", "vad").strip().lower()
if ASR_SEGMENTATION_MODE not in {"vad", "overlap_chunks", "fixed_chunks"}:
    ASR_SEGMENTATION_MODE = "vad"

# ONNX Runtime
ONNX_PROVIDER = validate_onnx_provider(os.getenv("ONNX_PROVIDER"))
ONNX_QUANTIZATION = _validate_onnx_quantization(os.getenv("ONNX_QUANTIZATION"))
ONNX_MODEL_DIR = os.getenv("ONNX_MODEL_DIR", "").strip() or None
ONNX_VAD_MODEL = os.getenv("ONNX_VAD_MODEL", "silero").strip() or "silero"

# Аудио
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
AUDIO_PREPROCESSING_MODE = os.getenv("AUDIO_PREPROCESSING_MODE", "auto").strip().lower()
if AUDIO_PREPROCESSING_MODE not in {"off", "auto", "light", "denoise"}:
    AUDIO_PREPROCESSING_MODE = "auto"

# Диаризация — только ONNX (без токена)
DIARIZATION_BACKEND = "onnx"

# Статистика — в Docker не сохраняем
if DOCKER_MODE:
    STATS_FILE = None
else:
    STATS_FILE = os.path.expanduser(os.getenv("STATS_FILE", "processing_stats.json"))
    if not os.path.isabs(STATS_FILE):
        STATS_FILE = str(user_config_dir() / STATS_FILE)

# Поддерживаемые форматы файлов (входные)
SUPPORTED_FORMATS = (
    'Media files',
    '*.mp3 *.wav *.m4a *.aac *.mp4 *.avi *.mov *.mkv *.webm *.flac *.ogg *.wma *.qta *.3gp'
)

MEDIA_EXTENSIONS = (
    '.mp3', '.wav', '.m4a', '.aac', '.mp4', '.avi', '.mov', '.mkv', '.webm',
    '.flac', '.ogg', '.wma', '.qta', '.3gp'
)

# Форматы выходных файлов
OUTPUT_FORMATS = {
    'txt':                  'Текст (.txt)',
    'txt_timecodes':        'Таймкоды (_timecodes.txt)',
    'txt_diarize':          'Диаризация (_diarize.txt)',
    'txt_diarize_timecodes': 'Диар.+тайм. (_diarize_timecodes.txt)',
    'md':                   'Markdown (.md)',
    'srt':                  'SRT (.srt)',
    'vtt':                  'VTT (.vtt)',
}

# Telegram-бот
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
