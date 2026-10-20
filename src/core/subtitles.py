"""Планирование коротких subtitle cues поверх технических ASR-сегментов."""

from __future__ import annotations

import math
from dataclasses import dataclass

_MAX_JOIN_GAP_SECONDS = 1.0
_PREFERRED_MIN_TEXT_WIDTH_WITH_SPEAKER = 16

# Видимый разделитель метки спикера в SRT. VTT использует невидимую разметку
# <v ...>, поэтому она не участвует в бюджете ширины строки.
SRT_SPEAKER_SEPARATOR = ": "


@dataclass(frozen=True)
class SubtitleOptions:
    """Пользовательские ограничения вывода SRT/VTT."""

    sentence_split: bool = True
    max_line_count: int = 2
    max_line_width: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.sentence_split, bool):
            raise TypeError("sentence_split должен быть boolean")
        if type(self.max_line_count) is not int:
            raise TypeError("max_line_count должен быть целым числом")
        if type(self.max_line_width) is not int:
            raise TypeError("max_line_width должен быть целым числом")
        if not 1 <= self.max_line_count <= 4:
            raise ValueError("max_line_count должен быть от 1 до 4")
        if not 20 <= self.max_line_width <= 100:
            raise ValueError("max_line_width должен быть от 20 до 100")


@dataclass(frozen=True)
class SubtitleCue:
    """Один временной блок субтитров до сериализации в SRT/VTT.

    speaker — полное имя спикера, семантическая атрибуция для VTT <v ...>.
    speaker_label — видимая метка для SRT, заполнена только на первом cue
    новой реплики спикера.
    """

    start: float
    end: float
    lines: tuple[str, ...]
    speaker: str | None = None
    speaker_label: str | None = None


def _ends_sentence(text: str) -> bool:
    return text.rstrip('"\'»”)]}').endswith((".", "!", "?", "…"))


def _normalized_words(utterance: dict) -> list[dict]:
    """Вернуть валидные монотонные words либо пустой список для fallback."""

    raw_words = utterance.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        return []

    normalized: list[dict] = []
    previous_start = -math.inf
    previous_end = -math.inf
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            return []
        raw_text = raw_word.get("text", "")
        if not isinstance(raw_text, str):
            return []
        text = raw_text.strip()
        try:
            start = float(raw_word["start"])
            end = float(raw_word["end"])
        except (KeyError, TypeError, ValueError):
            return []
        if (
            not text
            or not math.isfinite(start)
            or not math.isfinite(end)
            or end <= start
            or start < previous_start
            or end < previous_end
            or start < previous_end
        ):
            return []
        normalized.append({**raw_word, "text": text, "start": start, "end": end})
        previous_start = start
        previous_end = end
    transcription = utterance.get("transcription")
    if isinstance(transcription, str):
        source_text = "".join(transcription.split())
        word_text = "".join("".join(word["text"] for word in normalized).split())
        if source_text != word_text:
            return []
    return normalized


def _fallback_words(utterance: dict) -> list[dict]:
    """Приблизительно разметить слова внутри исходных boundaries."""

    transcription = utterance.get("transcription", "")
    if not isinstance(transcription, str):
        return []
    tokens = transcription.split()
    if not tokens:
        return []
    try:
        start, end = (float(value) for value in utterance.get("boundaries", (0.0, 0.0)))
    except (TypeError, ValueError):
        return []
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return []

    weights = [max(len(token), 1) for token in tokens]
    total_weight = sum(weights)
    elapsed = 0
    words: list[dict] = []
    for token, weight in zip(tokens, weights, strict=True):
        word_start = start + (end - start) * elapsed / total_weight
        elapsed += weight
        word_end = start + (end - start) * elapsed / total_weight
        words.append({"text": token, "start": word_start, "end": word_end})
    return words


def _split_long_words(words: list[dict], width: int) -> list[dict]:
    """Разделить токены длиннее строки с пропорциональными timestamps."""

    split_words: list[dict] = []
    for word in words:
        text = str(word["text"])
        if len(text) <= width:
            split_words.append(word)
            continue

        start = float(word["start"])
        duration = float(word["end"]) - start
        text_length = len(text)
        for offset in range(0, text_length, width):
            chunk_end = min(offset + width, text_length)
            split_words.append({
                **word,
                "text": text[offset:chunk_end],
                "start": start + duration * offset / text_length,
                "end": start + duration * chunk_end / text_length,
                "joins_previous": offset > 0,
            })
    return split_words


def _fit_speaker(speaker: object, width: int) -> str | None:
    """Обрезать имя спикера под видимую SRT-метку, сохранив место для текста."""

    # Единственный вызывающий код гарантирует speaker is not None и непустую
    # строку; guard остаётся на случай, если _fit_speaker станет использоваться иначе.
    if speaker is None:
        return None
    value = str(speaker).strip()
    if not value:
        return None
    separator_width = len(SRT_SPEAKER_SEPARATOR)
    reserved_text_width = min(
        _PREFERRED_MIN_TEXT_WIDTH_WITH_SPEAKER,
        width - separator_width - 1,
    )
    max_length = max(1, width - separator_width - reserved_text_width)
    if len(value) <= max_length:
        return value
    if max_length <= 2:
        # value уже strip-нут, поэтому хвост не может стать пустым; lstrip
        # убирает пробел, попавший в срез, чтобы строка субтитра не начиналась
        # с пробела.
        return value[-max_length:].lstrip()
    prefix_length = (max_length - 1) // 2
    suffix_length = max_length - prefix_length - 1
    return f"{value[:prefix_length]}…{value[-suffix_length:]}"


def _content_line_width(width: int, label: str | None) -> int:
    """Ширина текста строки: метка занимает место только там, где её печатают."""

    if label is None:
        return width
    return max(1, width - len(label) - len(SRT_SPEAKER_SEPARATOR))


def build_subtitle_cues(
    utterances: list[dict],
    options: SubtitleOptions | None = None,
) -> list[SubtitleCue]:
    """Преобразовать ASR utterances в семантические subtitle cues."""

    options = options or SubtitleOptions()
    cues: list[SubtitleCue] = []
    grouped_words: list[dict] = []
    grouped_speaker: str | None = None
    grouped_label: str | None = None
    labeled_speaker: str | None = None

    def flush_group() -> None:
        nonlocal grouped_words, grouped_label
        sentence: list[dict] = []
        for grouped_word in grouped_words:
            sentence.append(grouped_word)
            if options.sentence_split and _ends_sentence(grouped_word["text"]):
                cues.extend(_cues_from_words(
                    sentence,
                    grouped_speaker,
                    grouped_label,
                    options,
                ))
                grouped_label = None
                sentence = []
        if sentence:
            cues.extend(_cues_from_words(
                sentence,
                grouped_speaker,
                grouped_label,
                options,
            ))
        grouped_words = []

    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        transcription = utterance.get("transcription", "")
        if not isinstance(transcription, str) or not transcription.strip():
            continue
        words = _normalized_words(utterance) or _fallback_words(utterance)
        if not words:
            continue
        raw_speaker = utterance.get("speaker")
        speaker = str(raw_speaker).strip() if raw_speaker is not None else ""
        speaker = speaker or None
        can_join = bool(grouped_words) and speaker == grouped_speaker
        if can_join:
            gap = float(words[0]["start"]) - float(grouped_words[-1]["end"])
            can_join = 0.0 <= gap <= _MAX_JOIN_GAP_SECONDS
        if not can_join:
            flush_group()
            grouped_speaker = speaker
            grouped_label = None
            if speaker is not None and speaker != labeled_speaker:
                grouped_label = _fit_speaker(speaker, options.max_line_width)
                labeled_speaker = speaker
        # Пока метка не напечатана, любое слово группы может попасть в cue
        # с меткой, поэтому делим по суженной ширине всю группу целиком.
        split_width = _content_line_width(options.max_line_width, grouped_label)
        grouped_words.extend(_split_long_words(words, split_width))

    flush_group()

    return cues


def _wrap_words(words: list[dict], width: int) -> tuple[str, ...]:
    lines: list[str] = []
    current = ""
    for word in words:
        text = str(word["text"]).strip()
        if not text:
            continue
        # Продолжение разрезанного слова приклеивается без пробела: пробел внутри
        # слова меняет границы слов для потребителей субтитров.
        joins_previous = bool(word.get("joins_previous")) and bool(current)
        candidate = f"{current}{text}" if joins_previous else f"{current} {text}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = text
        else:
            current = candidate
    if current:
        lines.append(current)
    return tuple(lines)


def _cues_from_words(
    words: list[dict],
    speaker: str | None,
    label: str | None,
    options: SubtitleOptions,
) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    current: list[dict] = []
    pending_label = label
    line_width = _content_line_width(options.max_line_width, pending_label)
    for word in words:
        candidate = [*current, word]
        lines = _wrap_words(candidate, line_width)
        if current and len(lines) > options.max_line_count:
            cues.append(_cue_from_words(current, speaker, pending_label, line_width))
            pending_label = None
            line_width = _content_line_width(options.max_line_width, pending_label)
            current = [word]
        else:
            current = candidate
    if current:
        cues.append(_cue_from_words(current, speaker, pending_label, line_width))
    return cues


def _cue_from_words(
    words: list[dict],
    speaker: str | None,
    label: str | None,
    line_width: int,
) -> SubtitleCue:
    return SubtitleCue(
        start=float(words[0]["start"]),
        end=float(words[-1]["end"]),
        lines=_wrap_words(words, line_width),
        speaker=speaker,
        speaker_label=label,
    )
