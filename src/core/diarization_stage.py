"""Стадия диаризации ONNX (pyannote-segmentation-3.0 + WeSpeaker).

Не требует PyTorch и HF-токена. Сбой диаризации никогда не теряет транскрипт
и не подменяется фиктивной разметкой «Спикер №1».
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .processing_support import DiarizationOutcome, DiarizationSetupError

if TYPE_CHECKING:
    from .diarization.base import DiarizationBackend

_module_logger = logging.getLogger("src.core.processor")


class DiarizationStageMixin:
    """Требует от класса: logger, model_loader, _diarization_manager,
    _diarization_provider, _diarization_factory_error, _active_diarization_backend,
    _update_progress и _stage_reporter."""

    @property
    def diarization_manager(self) -> DiarizationBackend | None:
        """Ленивая загрузка ONNX backend с актуальным provider-ом."""
        from ..config import ONNX_MODEL_DIR, ONNX_PROVIDER
        from .diarization.factory import create_diarization_backend

        provider = getattr(self.model_loader, "requested_provider", None) or ONNX_PROVIDER

        if (
            self._diarization_manager is not None
            and self._diarization_provider != provider
        ):
            try:
                self._diarization_manager.unload()
            except Exception:
                pass
            self._diarization_manager = None

        if self._diarization_manager is None:
            try:
                self._diarization_manager = create_diarization_backend(
                    "onnx",
                    provider=provider,
                    model_dir=ONNX_MODEL_DIR,
                )
                self._diarization_provider = provider
                self._diarization_factory_error = None
            except Exception as e:
                self._diarization_factory_error = e
                self.logger(f"Не удалось подготовить определение говорящих: {e}")
        return self._diarization_manager

    def _run_diarization(
        self,
        audio_path: str,
        utterances: list,
        *,
        requested: bool,
        num_speakers: int | None,
        speaker_names: dict[str, str] | None = None,
    ) -> DiarizationOutcome:
        """Диаризация с понятной причиной отказа; транскрипт не теряется никогда."""
        if not requested:
            return DiarizationOutcome(utterances)

        if not utterances:
            return DiarizationOutcome(utterances, attempted=True)

        self.logger(f"Определяем говорящих (ONNX)…")
        try:
            manager = self.diarization_manager
            if manager is None:
                raise RuntimeError("ONNX-диаризация недоступна")

            device = getattr(manager, "device", None)
            provider = getattr(manager, "provider", None)
            runtime_details = []
            if device:
                runtime_details.append(f"устройство {device}")
            if provider:
                runtime_details.append(f"провайдер {provider}")
            if runtime_details:
                self.logger(
                    "Определение говорящих выполняется на: "
                    + ", ".join(runtime_details)
                )

            self._update_progress("diarization", None)
            mapped = self._apply_diarization(
                audio_path,
                utterances,
                manager=manager,
                num_speakers=num_speakers,
                progress_callback=self._stage_reporter("diarization"),
                speaker_names=speaker_names,
            )
            self._update_progress("diarization", 1.0)
            self.logger(
                f"Говорящих найдено: {len(set(u.get('speaker', 'Неизвестный спикер') for u in mapped))}"
            )
            return DiarizationOutcome(mapped, applied=True, attempted=True)
        except Exception as e:
            self.logger(
                f"Не удалось определить говорящих, текст сохранён без разметки по говорящим: {e}"
            )
            return DiarizationOutcome(utterances, error=str(e), attempted=True)

    def _apply_diarization(
        self,
        audio_path: str,
        utterances: list,
        num_speakers: int | None = None,
        progress_callback=None,
        manager=None,
        speaker_names: dict[str, str] | None = None,
    ) -> list:
        """Применяет ONNX-диаризацию к сегментам транскрипции."""
        if manager is None:
            manager = self.diarization_manager
        if not manager:
            cause = self._diarization_factory_error
            if cause is not None:
                raise DiarizationSetupError(
                    f"Не удалось подготовить определение говорящих: {type(cause).__name__}: {cause}"
                ) from cause
            raise RuntimeError("Менеджер диаризации недоступен.")

        try:
            kwargs = {}
            if num_speakers is not None:
                kwargs["num_speakers"] = num_speakers
            if speaker_names is not None:
                kwargs["speaker_names"] = speaker_names

            speaker_segments = manager.diarize(
                audio_path,
                **kwargs,
                progress_callback=progress_callback,
            )
            if not speaker_segments:
                raise RuntimeError("Диаризатор не вернул ни одного speaker-сегмента.")

            return manager.map_speakers_to_transcription(
                utterances,
                speaker_segments
            )

        except Exception as e:
            _module_logger.error("Diarization failed for %s", audio_path, exc_info=True)
            raise
