"""Утилиты для Telegram-бота."""


def split_text_for_telegram(text: str, max_chars: int = 4000) -> list[str]:
    """Разбить текст на части не длиннее max_chars символов.

    Разбивает по абзацам, затем по словам, чтобы не разрывать слова.
    """
    if not text:
        return []

    chunks: list[str] = []
    current = ""

    for para in text.split("\n"):
        para = para.strip()
        if not para:
            para = "\n"

        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = ""

        # Разбиваем длинный абзац по словам
        words = para.split()
        for word in words:
            if len(current) + len(word) + 1 > max_chars:
                chunks.append(current.strip())
                current = ""
            current = f"{current} {word}".strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:max_chars]]
