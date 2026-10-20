"""Кастомный logging handler для отправки сообщений в Telegram.

Позволяет перенаправить все логи (прогресс обработки, ошибки, статусы)
прямо в Telegram-чат пользователя.
"""

import logging
import threading
from typing import Callable, Optional


class TelegramLogHandler(logging.Handler):
    """Logging handler, который отправляет сообщения в Telegram чат.
    
    Использует callback для асинхронной отправки сообщений.
    """

    def __init__(
        self,
        chat_id: int,
        send_callback: Callable[[str], None],
        level: int = logging.NOTSET,
        max_messages: int = 50,
    ):
        """
        Args:
            chat_id: ID Telegram чата для отправки сообщений
            send_callback: Асинхронный callback для отправки сообщения в чат
                          Должен принимать str и возвращать coroutine
            level: Уровень логирования (по умолчанию все сообщения)
            max_messages: Максимальное количество сообщений за сессию
        """
        super().__init__(level)
        self.chat_id = chat_id
        self.send_callback = send_callback
        self._message_count = 0
        self._max_messages = max_messages
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Отправить отформатированное сообщение в Telegram."""
        # Проверяем лимит сообщений
        with self._lock:
            if self._message_count >= self._max_messages:
                return
            self._message_count += 1

        try:
            msg = self.format(record)
            # Telegram лимит — 4096 символов
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            self.send_callback(msg)
        except Exception:
            # Не бросаем исключения из emit — это потеряет логи
            pass

    def reset(self) -> None:
        """Сбросить счётчик сообщений."""
        with self._lock:
            self._message_count = 0


class ProgressLogHandler(TelegramLogHandler):
    """Handler для сообщений прогресса.
    
    Отправляет статусные сообщения (прогресс, загрузка и т.д.)
    и обновляет существующее status_message вместо создания новых.
    """

    def __init__(
        self,
        chat_id: int,
        send_callback: Callable[[str], None],
        update_callback: Optional[Callable[[str], None]] = None,
        level: int = logging.INFO,
    ):
        """
        Args:
            chat_id: ID Telegram чата
            send_callback: Callback для отправки нового сообщения
            update_callback: Callback для обновления status_message (опционально)
        """
        super().__init__(chat_id, send_callback, level)
        self._update_callback = update_callback

    def emit(self, record: logging.LogRecord) -> None:
        """Отправить сообщение прогресса."""
        try:
            msg = self.format(record)
            if len(msg) > 4000:
                msg = msg[:3997] + "..."
            
            # Для progress сообщений — пробуем обновить status_message
            if self._update_callback and record.levelno <= logging.INFO:
                self._update_callback(msg)
            else:
                self.send_callback(msg)
        except Exception:
            pass
