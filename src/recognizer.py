"""ONNX-распознавание речи с диаризацией для Telegram-бота.

Длинные записи режутся ffmpeg на части (по умолчанию 15 минут, 16 кГц моно WAV)
и обрабатываются последовательно, поэтому пик памяти определяется длиной
части, а не всей записи.

В Docker-режиме:
- временные файлы удаляются сразу после обработки (в т.ч. при ошибке)
- результаты не сохраняются на диск — только возвращаются в память
"""

from __future__ import annotations

import gc
import glob
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time as _time
from collections.abc import Callable
from pathlib import Path

from src.core.model_loader import ModelLoader
from src.core.progress import ProgressEvent
from src.services.transcription_service import build_processor

logger = logging.getLogger(__name__)

# Длина одной части в секундах. 0 — не резать (старое поведение).
CHUNK_SEC = int(os.getenv("CHUNK_SECONDS", "900"))
FFMPEG_TIMEOUT_SEC = 60 * 60
# Логировать RSS раз в ~20 с по ходу обработки (уровень WARNING, виден при LOG_LEVEL=WARNING).
LOG_RSS = os.getenv("LOG_RSS", "1") == "1"


def _rss_mb() -> float:
    """Текущее потребление памяти процессом (МБ)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _rss_peak_mb() -> float:
    """Пиковое потребление памяти процессом за всё время жизни (МБ)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _trim_memory() -> None:
    """Вернуть освобождённую кучу glibc операционной системе."""
    try:
        import ctypes  # noqa: PLC0415

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _split_audio(src: str, out_dir: str) -> list[str]:
    """Одним проходом ffmpeg: 16 кГц моно WAV, нарезка на части по CHUNK_SEC.

    Возвращает список путей по порядку или [] при неудаче (тогда обрабатываем
    исходный файл целиком, как раньше).
    """
    if CHUNK_SEC <= 0:
        return []
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg не найден, нарезка на части отключена")
        return []

    pattern = os.path.join(out_dir, "chunk_%03d.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-y",
                "-i", src,
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                "-f", "segment", "-segment_time", str(CHUNK_SEC),
                "-reset_timestamps", "1",
                pattern,
            ],
            check=True,
            timeout=FFMPEG_TIMEOUT_SEC,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Нарезка через ffmpeg не удалась: %s", exc)
        return []

    return sorted(glob.glob(os.path.join(out_dir, "chunk_*.wav")))


class Recognizer:
    """Одноразовый загрузчик модели + транскрибация аудио-байтов."""

    def __init__(
        self,
        *,
        enable_diarization: bool = True,
        num_speakers: int | None = None,
    ):
        self._enable_diarization = enable_diarization
        self._num_speakers = num_speakers
        self._model_loader: ModelLoader | None = None
        self._loaded = False
        # Защита от параллельных вызовов (двойная загрузка модели, двойной пик памяти)
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._model_loader is None:
            self._model_loader = ModelLoader()

        logger.info("Загрузка модели GigaAM-v3...")

        def _log(msg: str) -> None:
            logger.info(f"[MODEL] {msg}")

        try:
            ok = self._model_loader.load_model(logger=_log)
            if not ok:
                diag = getattr(self._model_loader, "diagnostics", lambda: {})()
                error_info = diag if isinstance(diag, dict) else {}
                factory_error = error_info.get("error", "Неизвестная ошибка")
                backend = error_info.get("active_backend", "неизвестный")

                error_msg = (
                    f"❌ Не удалось загрузить модель распознавания.\n\n"
                    f"Backend: {backend}\n"
                    f"Ошибка: {factory_error}\n\n"
                    f"Возможные причины:\n"
                    f"  • Модель ещё не загружена (первый запуск, нужна интернет-связь)\n"
                    f"  • Нехватка памяти (минимум 2 ГБ свободно)\n"
                    f"  • Файлы модели повреждены\n\n"
                    f"Попробуйте перезапустить бота через 1-2 минуты."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            self._loaded = True
            logger.info("✅ Модель GigaAM-v3 загружена успешно")

        except RuntimeError:
            raise
        except Exception as exc:
            error_msg = (
                f"❌ Критическая ошибка при загрузке модели: {type(exc).__name__}: {exc}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from exc

    def transcribe(
        self,
        audio_input: bytes | str,
        *,
        diarize: bool | None = None,
        speaker_names: dict[str, str] | None = None,
        num_speakers: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[str, dict]:
        """Распознать речь из байтов или файла, вернуть текст и статистику.

        Возвращает: (текст, {char_count, total_time, prep_time,
                             transcribe_time, chunks})
        """
        with self._lock:
            self._ensure_loaded()

            diarize = diarize if diarize is not None else self._enable_diarization

            if isinstance(audio_input, str) and os.path.exists(audio_input):
                return self._process_file(
                    audio_input,
                    diarize=diarize,
                    speaker_names=speaker_names,
                    num_speakers=num_speakers,
                    progress_callback=progress_callback,
                )

            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(fd, "wb") as f:
                    f.write(audio_input)  # type: ignore[arg-type]

                return self._process_file(
                    tmp_path,
                    diarize=diarize,
                    speaker_names=speaker_names,
                    num_speakers=num_speakers,
                    progress_callback=progress_callback,
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def _process_file(
        self,
        audio_path: str,
        *,
        diarize: bool,
        speaker_names: dict[str, str] | None = None,
        num_speakers: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[str, dict]:
        """Нарезать на части, обработать по очереди, склеить текст."""
        start_time = _time.time()

        def notify(msg: str) -> None:
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        chunk_dir = tempfile.mkdtemp(prefix="chunks_")
        processor = None
        try:
            t0 = _time.time()
            if CHUNK_SEC > 0:
                notify("Подготовка звука (ffmpeg)...")
            chunks = _split_audio(audio_path, chunk_dir)
            split_time = _time.time() - t0

            owned = bool(chunks)  # части — наши временные файлы
            if not chunks:
                chunks = [audio_path]

            n = len(chunks)
            # Один процессор на весь файл: ONNX-модели диаризации грузятся один раз,
            # а не для каждой части (state процессора между файлами не хранится).
            processor = build_processor(
                model_loader=self._model_loader,
                stats_manager=None,
                logger=lambda msg: logger.info(msg),
                progress_callback=None,
                diarization_backend="onnx",
            )
            texts: list[str] = []
            diar_errors: list[str] = []
            prep_time = split_time
            transcribe_time = 0.0

            for i, path in enumerate(chunks):
                cb = progress_callback
                if n > 1 and progress_callback is not None:
                    prefix = f"Часть {i + 1}/{n}. "

                    def cb(msg: str, _p: str = prefix) -> None:
                        progress_callback(_p + msg)  # type: ignore[misc]

                label = f"Часть {i + 1}/{n}" if n > 1 else "Файл"
                logger.warning("%s: старт, RSS %.0f МБ", label, _rss_mb())

                text, res = self._run_one(
                    processor,
                    path,
                    diarize=diarize,
                    speaker_names=speaker_names,
                    num_speakers=num_speakers,
                    progress_callback=cb,
                )

                d = res.get("diarization") if isinstance(res, dict) else None
                if isinstance(d, dict) and d.get("error"):
                    diar_errors.append(str(d["error"]))

                if owned:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                gc.collect()
                logger.warning(
                    "%s: готово, RSS %.0f МБ, пик %.0f МБ", label, _rss_mb(), _rss_peak_mb()
                )

                if n > 1:
                    text = f"[{_fmt_ts(i * CHUNK_SEC)}]\n{text}"
                texts.append(text)
                prep_time += res.get("preprocessing_time", 0.0)
                transcribe_time += res.get("transcription_time", 0.0)

            full_text = "\n\n".join(texts)
            stats = {
                "char_count": len(full_text),
                "total_time": _time.time() - start_time,
                "prep_time": prep_time,
                "transcribe_time": transcribe_time,
                "chunks": n,
                "diarization_error": "; ".join(dict.fromkeys(diar_errors)) or None,
            }
            return full_text, stats

        finally:
            self._release_processor(processor)
            processor = None
            shutil.rmtree(chunk_dir, ignore_errors=True)
            gc.collect()
            _trim_memory()
            if LOG_RSS:
                logger.warning(
                    "После обработки и выгрузки диаризации: RSS %.0f МБ (пик %.0f МБ)",
                    _rss_mb(),
                    _rss_peak_mb(),
                )

    @staticmethod
    def _release_processor(processor) -> None:
        """Выгрузить ONNX-модели диаризации: между запросами они не нужны."""
        manager = getattr(processor, "_diarization_manager", None)
        if manager is not None:
            try:
                manager.unload()
            except Exception:
                logger.debug("Не удалось выгрузить диаризацию", exc_info=True)
            processor._diarization_manager = None

    def _run_one(
        self,
        processor,
        path: str,
        *,
        diarize: bool,
        speaker_names: dict[str, str] | None,
        num_speakers: int | None,
        progress_callback: Callable[[str], None] | None,
    ) -> tuple[str, dict]:
        """Обработать один файл (часть) и вернуть (текст, результат процессора)."""
        output_dir = str(Path(path).parent)
        stem = Path(path).stem
        diarize_path = Path(output_dir) / f"{stem}_diarize.txt"
        txt_path = Path(output_dir) / f"{stem}.txt"

        seen_stages: set[str] = set()
        last_rss_log = 0.0

        def _progress(event: ProgressEvent) -> None:
            nonlocal last_rss_log
            stage = event.stage
            pct = event.stage_progress
            proc_sec = event.processed_seconds
            total_sec = event.total_seconds

            if stage:
                seen_stages.add(stage)
            stage_num = len(seen_stages)

            if pct is not None and total_sec and total_sec > 0:
                msg = f"Стадия {stage_num}: {stage} — {pct*100:.0f}% ({proc_sec:.0f}/{total_sec:.0f} с)"
            elif pct is not None and pct > 0:
                msg = f"Стадия {stage_num}: {stage} — {pct*100:.0f}%"
            elif proc_sec and total_sec and total_sec > 0:
                pct_actual = proc_sec / total_sec * 100
                msg = f"Стадия {stage_num}: {stage} — {pct_actual:.0f}% ({proc_sec:.1f}/{total_sec:.1f} с)"
            elif proc_sec:
                msg = f"Стадия {stage_num}: {stage} — {proc_sec:.1f} с"
            else:
                return

            logger.info(msg)
            if LOG_RSS:
                now = _time.time()
                if now - last_rss_log >= 20:
                    last_rss_log = now
                    logger.warning("%s | RSS %.0f МБ", msg, _rss_mb())

            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        try:
            processor.progress_callback = _progress

            result = processor.process_file(
                filepath=path,
                output_dir=output_dir,
                file_index=0,
                total_files=1,
                enable_diarization=diarize,
                diarization_backend="onnx",
                num_speakers=(
                    num_speakers if num_speakers is not None else self._num_speakers
                ),
                speaker_names=speaker_names,
                output_formats=["txt"],
            )

            if not result["success"]:
                raise RuntimeError(
                    f"Ошибка обработки: {result.get('error', 'неизвестная')}"
                )

            if diarize and diarize_path.exists():
                text = diarize_path.read_text(encoding="utf-8")
            elif txt_path.exists():
                text = txt_path.read_text(encoding="utf-8")
            else:
                text = result.get("text", "Распознавание не вернуло текст")

            return text, result

        finally:
            for p in (diarize_path, txt_path):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            processor.progress_callback = None
            gc.collect()

    def unload(self) -> None:
        """Освободить модель."""
        with self._lock:
            if self._model_loader is not None:
                self._model_loader.unload()
                self._loaded = False
                gc.collect()


def build_recognizer(
    *,
    enable_diarization: bool = True,
    num_speakers: int | None = None,
) -> Recognizer:
    """Фабрика Recognizer с кэшированием модели."""
    return Recognizer(
        enable_diarization=enable_diarization,
        num_speakers=num_speakers,
    )
