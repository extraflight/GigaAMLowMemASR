"""Complete ONNX diarization pipeline: segmentation → embeddings → clustering."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np

from ...utils.model_cache import onnx_model_is_local
from ..model_preparation import PreparationCancelled, PreparationState
from .clustering import cluster_embeddings
from .mapping import SpeakerMappingMixin
from .onnx_embeddings import ONNX_EMBEDDING_REPO, OnnxSpeakerEmbeddings
from .onnx_segmentation import ONNX_SEGMENTATION_REPO, OnnxSegmentation
from .reconstruction import reconstruct_speaker_segments


class OnnxDiarizationBackend(SpeakerMappingMixin):
    backend = "onnx"

    def __init__(
        self,
        *,
        provider: str = "auto",
        model_dir: str | None = None,
        segmenter: OnnxSegmentation | None = None,
        embedding_extractor: OnnxSpeakerEmbeddings | None = None,
        cluster_fn: Callable = cluster_embeddings,
        reconstruct_fn: Callable = reconstruct_speaker_segments,
    ) -> None:
        self.provider = provider
        self.model_dir = model_dir
        self.hf_token = None
        self._segmenter = segmenter or OnnxSegmentation(
            provider=provider,
            model_dir=model_dir,
        )
        self._embedding_extractor = embedding_extractor or OnnxSpeakerEmbeddings(
            provider=provider,
            model_dir=model_dir,
        )
        self._cluster_fn = cluster_fn
        self._reconstruct_fn = reconstruct_fn

    SAMPLE_RATE = 16_000

    @staticmethod
    def _progress(callback, value: float) -> None:
        if callback is not None:
            callback(value, None, None)

    @classmethod
    def _to_target_rate(cls, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Привести дорожку к 16 кГц.

        AUDIO_SAMPLE_RATE настраивается через env, а pyannote и Sortformer
        ресемплируют сами — падать на легитимной настройке нельзя.
        """
        if sample_rate == cls.SAMPLE_RATE:
            return waveform
        import soxr  # noqa: PLC0415

        resampled = soxr.resample(waveform, sample_rate, cls.SAMPLE_RATE)
        return np.asarray(resampled, dtype=np.float32)

    def _extract_accepts_progress(self) -> bool:
        try:
            return "progress" in inspect.signature(self._embedding_extractor.extract).parameters
        except (TypeError, ValueError):
            return False

    def diarize(
        self,
        audio_path: str,
        num_speakers: int | None = None,
        progress_callback=None,
        speaker_names: dict[str, str] | None = None,
    ):
        import soundfile as sf  # noqa: PLC0415

        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        # Канонический WAV — моно: берём столбец без копии. Раньше mean(axis=1)
        # делал вторую полную копию дорожки (~230 МБ на час), а `samples` жил
        # до конца функции.
        if samples.shape[1] == 1:
            waveform = np.ascontiguousarray(samples[:, 0])
        else:
            waveform = samples.mean(axis=1).astype(np.float32, copy=False)
        del samples
        waveform = self._to_target_rate(waveform, sample_rate)
        segmentation = self._segmenter.infer(waveform)
        self._progress(progress_callback, 0.35)

        extract_kwargs = {}
        if progress_callback is not None and self._extract_accepts_progress():
            extract_kwargs["progress"] = lambda fraction: self._progress(
                progress_callback, 0.35 + 0.35 * fraction
            )
        embeddings = self._embedding_extractor.extract(
            waveform, segmentation, **extract_kwargs
        )
        # Дорожка дальше не нужна: освобождаем до кластеризации.
        del waveform
        if not bool(np.any(embeddings.valid)):
            raise ValueError("ONNX diarization не получила валидные speaker embeddings")
        self._progress(progress_callback, 0.7)
        assignments = self._cluster_fn(embeddings, num_speakers=num_speakers)
        segments = self._reconstruct_fn(segmentation, assignments)
        if not segments:
            raise ValueError("ONNX diarization не восстановила speaker-сегменты")
        segments = self._rename_speakers(segments, speaker_names=speaker_names)
        self._progress(progress_callback, 1.0)
        return segments

    def prepare(self, report=None, cancel_check=None):
        """Загрузить обе ONNX-модели до обработки первого файла."""
        emit = report or (lambda _state, **_kwargs: None)
        cancelled = cancel_check or (lambda: False)
        if cancelled():
            raise PreparationCancelled("Подготовка ONNX-диаризации отменена")
        if not onnx_model_is_local(ONNX_SEGMENTATION_REPO, root=self.model_dir):
            emit(PreparationState.DOWNLOADING, message="ONNX segmentation-3.0")
        emit(PreparationState.LOADING, message="ONNX segmentation-3.0")
        self._segmenter._ensure_session()  # noqa: SLF001
        if cancelled():
            raise PreparationCancelled("Подготовка ONNX-диаризации отменена")
        if not onnx_model_is_local(ONNX_EMBEDDING_REPO, root=self.model_dir):
            emit(PreparationState.DOWNLOADING, message="WeSpeaker embeddings")
        emit(PreparationState.LOADING, message="WeSpeaker embeddings")
        self._embedding_extractor._ensure_model()  # noqa: SLF001
        return self

    def unload(self) -> None:
        self._segmenter.unload()
        self._embedding_extractor.unload()
