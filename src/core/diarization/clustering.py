"""Constrained cosine agglomeration for local speaker embeddings."""

from __future__ import annotations

import os

import numpy as np

from .onnx_embeddings import EmbeddingResult

_EPS = float(np.finfo(np.float32).eps)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, _EPS)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """1 - косинусное сходство для строк единичной длины.

    Accelerate BLAS на Apple Silicon выставляет FPU-флаги overflow/invalid на
    полностью корректных входах (SIMD считает и padding-дорожки), из-за чего
    numpy печатал RuntimeWarning на каждый файл. Входы санитизируются до
    вызова, поэтому флаги здесь заведомо ложные и глушатся точечно.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        distance = np.asarray(left @ right, dtype=np.float32)
        # На месте: прежний `(1.0 - m).astype(float32)` держал одновременно три
        # матрицы M×M (произведение, разность, копия) — ~0.5 ГБ при M≈6500.
        np.subtract(1.0, distance, out=distance)
    return distance


def _closest_active_pair(
    centroids: np.ndarray,
    active: np.ndarray,
) -> tuple[int, int] | None:
    """Ближайшая пара активных кластеров без учёта cannot-link."""
    active_index = np.flatnonzero(active)
    if len(active_index) < 2:
        return None
    subset = centroids[active_index]
    distances = _cosine_distance(subset, subset.T)
    np.fill_diagonal(distances, np.inf)
    flat = int(np.argmin(distances))
    left, right = divmod(flat, len(active_index))
    return int(active_index[left]), int(active_index[right])


def _cluster_valid(
    result: EmbeddingResult,
    *,
    num_speakers: int | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Cluster valid rows while forbidding two speakers from one window to merge.

    Порог подобран под эмбеддинги wespeaker-voxceleb-resnet34: на реальном
    двухголосом интервью медиана попарных дистанций около 0.35, поэтому прежнее
    значение 0.35 резало одного спикера на несколько кластеров.

    Явно заданное число спикеров выполняется всегда. Cannot-link (два локальных
    спикера одного окна) — эвристика, и когда она заводит слияния в тупик выше
    запрошенного количества, ограничение ослабляется: пожелание пользователя
    важнее эвристики. Без этого запрос «2 спикера» молча возвращал 4.
    """

    if threshold is None:
        # Порог слияния при автоопределении числа спикеров. Больше — кластеров
        # меньше (риск склеить разных людей), меньше — больше (риск разрезать
        # одного). DIARIZATION_THRESHOLD=0.7 и т. п.
        threshold = float(os.getenv("DIARIZATION_THRESHOLD", "0.6"))

    valid_rows = np.flatnonzero(result.valid)
    labels = np.full(len(result.valid), -1, dtype=np.int64)
    if len(valid_rows) == 0:
        return labels

    count = len(valid_rows)
    target = None if num_speakers is None else max(1, min(int(num_speakers), count))

    sums = np.asarray(result.embeddings[valid_rows], dtype=np.float32).copy()
    if not np.isfinite(sums).all():
        # Нечисловой эмбеддинг мог бы разъехаться по всей матрице дистанций,
        # а глушение FPU-флагов ниже скрыло бы это.
        sums = np.nan_to_num(sums, nan=0.0, posinf=0.0, neginf=0.0)
    centroids = _unit_rows(sums)
    windows = np.asarray(result.window_indices)[valid_rows]

    # Дистанции пересчитываются только для изменившегося кластера, поэтому
    # стоимость агломерации линейна по числу слияний, а не кубична.
    distances = _cosine_distance(centroids, centroids.T)
    distances[windows[:, None] == windows[None, :]] = np.inf

    masks = [1 << int(window) for window in windows]
    members: list[list[int]] = [[int(row)] for row in valid_rows]
    active = np.ones(count, dtype=np.bool_)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(count), nearest]

    remaining = count
    while remaining > 1:
        if target is not None and remaining <= target:
            break
        candidates = np.where(active, nearest_distance, np.inf)
        left = int(np.argmin(candidates))
        best = float(candidates[left])
        if np.isfinite(best):
            if target is None and best > threshold:
                break
            right = int(nearest[left])
        else:
            # Разрешённых слияний не осталось: кластеры накопили общие окна.
            # Без цели останавливаемся, с целью — ослабляем cannot-link.
            if target is None:
                break
            forced = _closest_active_pair(centroids, active)
            if forced is None:
                break
            left, right = forced

        sums[left] += sums[right]
        centroids[left] = sums[left] / max(float(np.linalg.norm(sums[left])), _EPS)
        masks[left] |= masks[right]
        members[left].extend(members[right])
        active[right] = False
        remaining -= 1

        distances[right, :] = np.inf
        distances[:, right] = np.inf

        row = _cosine_distance(centroids, centroids[left])
        blocked = np.fromiter(
            ((masks[left] & mask) != 0 for mask in masks),
            dtype=np.bool_,
            count=count,
        )
        row[blocked | ~active] = np.inf
        row[left] = np.inf
        distances[left, :] = row
        distances[:, left] = row

        stale = np.flatnonzero(active & ((nearest == left) | (nearest == right)))
        for index in (*stale.tolist(), left):
            nearest[index] = int(np.argmin(distances[index]))
            nearest_distance[index] = distances[index, nearest[index]]

    clusters = [members[index] for index in np.flatnonzero(active)]
    clusters.sort(key=min)
    for label, cluster in enumerate(clusters):
        labels[cluster] = label
    return labels


def _assign_unreliable(
    result: EmbeddingResult,
    labels: np.ndarray,
    pending: np.ndarray,
) -> np.ndarray:
    """Раздать строки с короткой речью по ближайшим готовым центроидам.

    Кластеры уже построены по надёжным строкам, поэтому шумные короткие
    эмбеддинги не могут ни породить новый кластер, ни сдвинуть центроид.
    Cannot-link соблюдается, пока есть свободная метка: две дорожки одного окна
    не получают один голос.
    """
    labels = labels.copy()
    todo = np.flatnonzero(pending)
    clustered = np.flatnonzero(labels >= 0)
    if len(todo) == 0 or len(clustered) == 0:
        return labels

    embeddings = np.asarray(result.embeddings, dtype=np.float32)
    count = int(labels.max()) + 1
    sums = np.zeros((count, embeddings.shape[1]), dtype=np.float32)
    np.add.at(sums, labels[clustered], embeddings[clustered])
    centroids = _unit_rows(sums)
    windows = np.asarray(result.window_indices)

    taken: dict[int, set[int]] = {}
    for row in clustered:
        taken.setdefault(int(windows[row]), set()).add(int(labels[row]))

    distances = _cosine_distance(_unit_rows(embeddings[todo]), centroids.T)
    for position in np.argsort(distances.min(axis=1)):  # сначала самые уверенные
        row = int(todo[position])
        busy = taken.setdefault(int(windows[row]), set())
        ranked = [int(label) for label in np.argsort(distances[position])]
        label = next((item for item in ranked if item not in busy), ranked[0])
        labels[row] = label
        busy.add(label)
    return labels


def cluster_embeddings(
    result: EmbeddingResult,
    *,
    num_speakers: int | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Кластеризация; опционально в два этапа.

    DIARIZATION_RELIABLE_SEC > 0 (например 1.0): кластеры строятся только по
    строкам с речью не короче этого порога, остальные валидные строки потом
    раздаются по ближайшим центроидам. Без переменной — прежнее поведение.
    """
    reliable_sec = float(os.getenv("DIARIZATION_RELIABLE_SEC", "0"))
    durations = getattr(result, "durations", None)
    if reliable_sec > 0 and durations is not None:
        valid = np.asarray(result.valid, dtype=bool)
        reliable = valid & (np.asarray(durations) >= reliable_sec)
        if 2 <= int(reliable.sum()) < int(valid.sum()):
            base = EmbeddingResult(
                embeddings=result.embeddings,
                valid=reliable,
                window_indices=result.window_indices,
                local_speakers=result.local_speakers,
            )
            labels = _cluster_valid(base, num_speakers=num_speakers, threshold=threshold)
            return _assign_unreliable(result, labels, valid & ~reliable)
    return _cluster_valid(result, num_speakers=num_speakers, threshold=threshold)
