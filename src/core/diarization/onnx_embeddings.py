"""Speaker embedding extraction through the public onnx-asr WeSpeaker API."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...utils.model_cache import onnx_model_location
from ..asr.onnx_loading import load_speaker_embedding_model
from ..asr.onnx_provider import (
    available_onnx_providers,
    onnx_session_providers,
    resolve_onnx_providers,
)
from .onnx_segmentation import SegmentationResult

ONNX_EMBEDDING_REPO = "wespeaker/wespeaker-voxceleb-resnet34"


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: np.ndarray
    valid: np.ndarray
    window_indices: np.ndarray
    local_speakers: np.ndarray


class OnnxSpeakerEmbeddings:
    def __init__(
        self,
        *,
        provider: str = "auto",
        model_dir: str | None = None,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        min_speech_seconds: float = 0.5,
        batch_size: int | None = None,
        model_factory: Callable[..., Any] | None = None,
        available_provider_probe: Callable[[], tuple[str, ...]] | None = None,
    ) -> None:
        self.provider = provider
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_seconds = min_speech_seconds
        # Активации ResNet34 растут пропорционально батчу: на CPU батч 16 по 10 с
        # поднимал RSS примерно на гигабайт. Меняется без правки кода:
        # DIARIZATION_EMBEDDING_BATCH=4.
        if batch_size is None:
            batch_size = int(os.getenv("DIARIZATION_EMBEDDING_BATCH", "16"))
        self.batch_size = max(1, int(batch_size))
        self._model_factory = model_factory
        self._available_provider_probe = available_provider_probe or (
            lambda: available_onnx_providers(self.provider)
        )
        self._model: Any | None = None
        # Рабочий размер батча: уменьшается при нехватке памяти (см. _embed_batch).
        self._batch_limit = self.batch_size

    @staticmethod
    def _load_model(*, providers: list[str], model_dir, offline: bool | None = None):
        return load_speaker_embedding_model(
            ONNX_EMBEDDING_REPO,
            model_dir,
            offline=offline,
            providers=providers,
            preprocessor_config={"use_numpy_preprocessors": False},
        )

    def _ensure_model(self):
        if self._model is None:
            selection = resolve_onnx_providers(
                self.provider,
                available=self._available_provider_probe(),
            )
            factory = self._model_factory or self._load_model
            location = onnx_model_location(ONNX_EMBEDDING_REPO, root=self.model_dir)
            kwargs: dict[str, Any] = {
                "providers": onnx_session_providers(selection),
                "model_dir": location.path,
            }
            if location.offline is not None:
                kwargs["offline"] = location.offline
            self._model = factory(**kwargs)
        return self._model

    @staticmethod
    def _is_out_of_memory(exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}".lower()
        return "failed to allocate memory" in message or "out of memory" in message

    def _embed_in_batches(self, model, waveforms: list[np.ndarray]) -> list[np.ndarray]:
        """Считать эмбеддинги батчами, уменьшая батч при нехватке памяти.

        Свободную память ускорителя нельзя узнать заранее: ORT её не отдаёт, а
        рядом уже лежит ASR-модель. Поэтому вместо угадывания размера батча
        ловим отказ аллокации и делим батч пополам — это работает одинаково для
        CUDA, DirectML, CoreML и CPU.
        """
        batch_size = self.batch_size
        while True:
            chunks: list[np.ndarray] = []
            try:
                for start in range(0, len(waveforms), batch_size):
                    raw = np.asarray(
                        model.embedding(waveforms[start : start + batch_size]),
                        dtype=np.float32,
                    )
                    chunks.append(raw[None, :] if raw.ndim == 1 else raw)
                return chunks
            except Exception as exc:
                if batch_size <= 1 or not self._is_out_of_memory(exc):
                    raise
                batch_size = max(1, batch_size // 2)

    def _embed_batch(self, model, batch: list[np.ndarray]) -> np.ndarray:
        """Эмбеддинги одного батча; при нехватке памяти делит батч пополам."""
        try:
            raw = np.asarray(model.embedding(batch), dtype=np.float32)
        except Exception as exc:
            if len(batch) <= 1 or not self._is_out_of_memory(exc):
                raise
            half = max(1, len(batch) // 2)
            self._batch_limit = half
            return np.concatenate(
                [self._embed_batch(model, batch[:half]), self._embed_batch(model, batch[half:])],
                axis=0,
            )
        return raw[None, :] if raw.ndim == 1 else raw

    def extract(
        self,
        waveform: np.ndarray,
        segmentation: SegmentationResult,
        progress: Callable[[float], None] | None = None,
    ) -> EmbeddingResult:
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        probabilities = segmentation.probabilities
        rows = probabilities.shape[0] * probabilities.shape[2]
        valid = np.zeros(rows, dtype=np.bool_)
        window_indices = np.repeat(np.arange(probabilities.shape[0]), probabilities.shape[2])
        local_speakers = np.tile(np.arange(probabilities.shape[2]), probabilities.shape[0])
        min_samples = max(1, int(round(self.min_speech_seconds * self.sample_rate)))
        frame_samples = max(1, int(round(segmentation.frame_step * self.sample_rate)))

        # Строк с достаточной речью — только чтобы показывать прогресс.
        speech_seconds = (probabilities >= self.threshold).sum(axis=1) * segmentation.frame_step
        expected = int((speech_seconds >= self.min_speech_seconds).sum())

        # Фрагменты речи не копятся списком на всю запись (на 3 часах это
        # около гигабайта): набирается один батч, считается, и копии освобождаются.
        # Результат тот же, что у расчёта «все фрагменты → батчами».
        self._batch_limit = self.batch_size
        parts: list[np.ndarray] = []
        valid_rows: list[int] = []
        pending: list[np.ndarray] = []
        model = None
        done = 0

        def flush() -> None:
            nonlocal model, done
            if not pending:
                return
            if model is None:
                model = self._ensure_model()
            parts.append(self._embed_batch(model, pending))
            done += len(pending)
            pending.clear()
            if progress is not None and expected:
                progress(min(1.0, done / expected))

        for row, (window_index, speaker_index) in enumerate(
            zip(window_indices, local_speakers, strict=True)
        ):
            frame_mask = probabilities[window_index, :, speaker_index] >= self.threshold
            speech = float(frame_mask.sum()) * segmentation.frame_step
            if speech < self.min_speech_seconds:
                continue
            start = int(round(segmentation.window_starts[window_index] * self.sample_rate))
            mask = np.repeat(frame_mask, frame_samples)
            end = min(len(audio), start + len(mask))
            if end <= start:
                continue
            # Вырезаем активную речь, а не глушим остальное окно нулями:
            # иначе строка с 0.5 c речи отдавала бы в WeSpeaker ~9.5 c тишины,
            # fbank/CMN-статистики считались бы по ней и эмбеддинги разных
            # спикеров стягивались бы к «эмбеддингу тишины».
            chunk = np.ascontiguousarray(audio[start:end][mask[: end - start]])
            if len(chunk) < min_samples:
                continue
            valid[row] = True
            valid_rows.append(row)
            pending.append(chunk)
            if len(pending) >= self._batch_limit:
                flush()
        flush()

        if not parts:
            embeddings = np.empty((rows, 0), dtype=np.float32)
        else:
            raw = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            normalized = raw / np.maximum(norms, np.finfo(np.float32).eps)
            embeddings = np.zeros((rows, normalized.shape[1]), dtype=np.float32)
            embeddings[np.asarray(valid_rows)] = normalized

        return EmbeddingResult(
            embeddings=embeddings,
            valid=valid,
            window_indices=window_indices,
            local_speakers=local_speakers,
        )

    def unload(self) -> None:
        self._model = None
