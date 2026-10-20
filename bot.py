#!/usr/bin/env python3
"""Telegram-бот для распознавания речи с диаризацией (2+ спикера).

Скачивание:
- файлы <= 20 МБ: штатно через PTB (get_file + download_to_drive);
- файлы > 20 МБ: через Telethon (MTProto, до 2 ГБ). Telethon подключается
  лениво, только когда пришёл большой файл, и отключается после простоя.

В Docker-режиме:
- логи только в stdout
- временные файлы удаляются сразу (в т.ч. по таймауту, если режим не выбран)
- статистика не сохраняется
"""

import asyncio
import logging
import os
import tempfile
import time
from typing import Any, Awaitable, Callable

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, Conflict
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.config import LOG_LEVEL, TELEGRAM_BOT_TOKEN
from src.recognizer import build_recognizer
from src.utils_helpers import split_text_for_telegram

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger(__name__)
# httpx пишет полные URL (с токеном бота) на уровне INFO
logging.getLogger("httpx").setLevel(logging.WARNING)

BOT_API_DOWNLOAD_LIMIT = 20 * 1024 * 1024  # лимит облачного Bot API на скачивание
PENDING_TTL_SEC = 15 * 60  # сколько ждём выбора режима
TELETHON_IDLE_SEC = 120  # через сколько секунд простоя отключать Telethon
MAX_INLINE_CHARS = 12000  # длиннее — отправляем .txt файлом

MIME_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}
KNOWN_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm", ".mp4"}

import base64
import struct

_FILE_REFERENCE_FLAG = 1 << 25
_WEB_LOCATION_FLAG = 1 << 24
# VoiceNote, Video, Document, Audio, Animation, VideoNote, DocumentAsFile
_DOCUMENT_TYPES = {3, 4, 5, 9, 10, 13, 17}


def _rle_decode(data: bytes) -> bytes:
    """Decode RLE-compressed bot API file data."""
    out, zero = bytearray(), False
    for b in data:
        if b == 0:
            zero = True
            continue
        if zero:
            out += b"\x00" * b
            zero = False
        else:
            out.append(b)
    return bytes(out)


def _read_tl_bytes(buf: bytes, pos: int) -> tuple[bytes, int]:
    """Read a length-prefixed TL bytes field."""
    length = buf[pos]
    if length <= 253:
        start, end = pos + 1, pos + 1 + length
        consumed = 1 + length
    else:
        length = int.from_bytes(buf[pos + 1 : pos + 4], "little")
        start, end = pos + 4, pos + 4 + length
        consumed = 4 + length
    return buf[start:end], pos + consumed + (-consumed % 4)


def parse_bot_file_id(file_id: str) -> tuple[int, int, int, bytes]:
    """Parse Bot API file_id into (dc_id, media_id, access_hash, file_reference)."""
    raw = base64.urlsafe_b64decode(file_id + "=" * (-len(file_id) % 4))
    data = _rle_decode(raw)

    # хвост: версия формата (>= 4 — два байта: minor, major)
    data = data[:-2] if data[-1] >= 4 else data[:-1]

    file_type, dc_id = struct.unpack_from("<ii", data, 0)
    pos = 8
    has_ref = bool(file_type & _FILE_REFERENCE_FLAG)
    if file_type & _WEB_LOCATION_FLAG:
        raise ValueError("web location не поддерживается")
    file_type &= ~(_FILE_REFERENCE_FLAG | _WEB_LOCATION_FLAG)

    if file_type not in _DOCUMENT_TYPES:
        raise ValueError(f"Неподдерживаемый тип file_id: {file_type}")

    file_reference = b""
    if has_ref:
        file_reference, pos = _read_tl_bytes(data, pos)
    media_id, access_hash = struct.unpack_from("<qq", data, pos)
    return dc_id, media_id, access_hash, file_reference

class LargeFileDownloader:
    """Ленивый Telethon-клиент (bot-токен) только для скачивания больших файлов.

    Ничего не занимает, пока не пришёл файл > 20 МБ. После скачивания
    отключается через TELETHON_IDLE_SEC секунд простоя.
    """

    def __init__(self, bot_token: str) -> None:
        self._bot_token = bot_token
        self._api_id = os.getenv("TELEGRAM_API_ID")
        self._api_hash = os.getenv("TELEGRAM_API_HASH")
        # Файл сессии: чтобы не логиниться по токену при каждом запуске.
        # В Docker смонтируйте для него volume.
        self._session = os.getenv("TELETHON_SESSION", "telethon_bot")
        self._client: Any = None
        self._lock = asyncio.Lock()
        self._idle_handle: asyncio.TimerHandle | None = None

    @property
    def configured(self) -> bool:
        return bool(self._api_id and self._api_hash)

    async def _ensure_client(self) -> Any:
        async with self._lock:
            if self._client is None or not self._client.is_connected():
                from telethon import TelegramClient  # noqa: PLC0415

                client = TelegramClient(self._session, int(self._api_id), self._api_hash)
                await client.start(bot_token=self._bot_token)
                self._client = client
            return self._client

    def _schedule_idle_close(self) -> None:
        if self._idle_handle:
            self._idle_handle.cancel()
        loop = asyncio.get_running_loop()
        self._idle_handle = loop.call_later(
            TELETHON_IDLE_SEC, lambda: asyncio.create_task(self.close())
        )

    async def download(
        self, file_id: str, dest_path: str, file_size: int | None = None
    ) -> None:
        if not self.configured:
            raise RuntimeError(
                "Файл больше 20 МБ, а TELEGRAM_API_ID / TELEGRAM_API_HASH не заданы."
            )
        from telethon.tl.types import InputDocumentFileLocation  # noqa: PLC0415

        if self._idle_handle:
            self._idle_handle.cancel()

        dc_id, media_id, access_hash, file_reference = parse_bot_file_id(file_id)
        location = InputDocumentFileLocation(
            id=media_id,
            access_hash=access_hash,
            file_reference=file_reference,
            thumb_size="",
        )

        client = await self._ensure_client()
        try:
            await client.download_file(
                location, file=dest_path, file_size=file_size, dc_id=dc_id
            )
        finally:
            self._schedule_idle_close()

    async def close(self) -> None:
        if self._idle_handle:
            self._idle_handle.cancel()
            self._idle_handle = None
        async with self._lock:
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:
                    logger.debug("Telethon disconnect failed", exc_info=True)
                self._client = None
                logger.info("Telethon отключён (простой)")


class TelegramSpeechBot:
    """Бот распознавания голосовых с поддержкой диаризации."""

    def __init__(self, token: str | None = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        if not self.token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN не установлен. Добавьте его в .env или в окружение."
            )

        self.app = (
            Application.builder()
            .token(self.token)
            .post_shutdown(self._on_shutdown)
            .build()
        )
        self.recognizer = build_recognizer(enable_diarization=True, num_speakers=None)
        self.large_files = LargeFileDownloader(self.token)
        self.usage: dict[int, int] = {}
        # (chat_id, status_message_id) -> {audio_path, message, status, expire_task}
        self._pending_audio: dict[tuple[int, int], dict[str, Any]] = {}
        self.setup_handlers()

    async def _on_shutdown(self, app: Application) -> None:
        await self.large_files.close()
        for pending in self._pending_audio.values():
            pending["expire_task"].cancel()
            self._safe_unlink(pending["audio_path"])
        self._pending_audio.clear()

    def setup_handlers(self) -> None:
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("stats", self.stats))
        self.app.add_handler(
            MessageHandler(
                filters.VOICE | filters.AUDIO | filters.VIDEO_NOTE, self.handle_audio
            )
        )
        self.app.add_handler(
            CallbackQueryHandler(self.handle_diarization_choice, pattern="^diarization_")
        )

    # ------------------------------------------------------------------ команды

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        await update.message.reply_text(
            f"Привет, {user.first_name}! 👋\n\n"
            "Я бот для распознавания голосовых сообщений.\n"
            "Отправьте голосовое или аудиофайл — я верну текст с разметкой спикеров.\n\n"
            "Команды:\n"
            "/help — справка\n"
            "/stats — статистика использования"
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "📋 Как использовать:\n\n"
            "1. Отправьте голосовое сообщение или аудиофайл.\n"
            "2. Бот обработает его через GigaAM-v3 (ONNX).\n"
            "3. Вы получите расшифровку с разметкой спикеров.\n\n"
            "Особенности:\n"
            "- Поддерживаемые форматы: OGG, WAV, MP3, M4A, AAC, WEBM\n"
            "- Файлы до 2 ГБ\n"
            "- Автоматическое определение 2+ спикеров\n"
            "- Работает без PyTorch и токенов"
        )

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        total = sum(self.usage.values())
        user_id = update.effective_user.id

        await update.message.reply_text(
            "📊 Статистика использования\n\n"
            f"Ваши запросы сегодня: {self.usage.get(user_id, 0)}\n"
            f"Всего обработано запросов: {total}"
        )

    # -------------------------------------------------------------- приём аудио

    async def handle_audio(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.message
        if message is None:
            return

        chat_id = update.effective_chat.id
        media = message.voice or message.audio or message.video_note
        if media is None:
            return

        size_mb = (getattr(media, "file_size", None) or 0) / 1024 / 1024
        status_message = await message.reply_text(
            f"🎧 Загружаю аудио ({size_mb:.0f} МБ)..."
            if size_mb > 20
            else "🎧 Загружаю аудио..."
        )

        audio_path = None
        try:
            audio_path = await self._download_audio_to_temp(media, context.bot)

            key = (chat_id, status_message.message_id)
            self._pending_audio[key] = {
                "audio_path": audio_path,
                "message": message,
                "status": status_message,
                "expire_task": asyncio.create_task(self._expire_pending(key)),
            }
            mid = status_message.message_id

            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🏥 Врач-Пациент", callback_data=f"diarization_medical:{mid}"
                        ),
                        InlineKeyboardButton(
                            "🚗 Авто", callback_data=f"diarization_auto:{mid}"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "📝 Текст без разметки", callback_data=f"diarization_none:{mid}"
                        ),
                    ],
                ]
            )
            await status_message.edit_text(
                "Выберите режим обработки:\n\n"
                "🏥 Врач-Пациент — 2 спикера\n"
                "🚗 Автоматическое определение спикеров\n"
                "📝 Текст без разметки",
                reply_markup=keyboard,
            )

        except Exception as exc:
            logger.exception("Audio download failed")
            if audio_path:
                self._pending_audio.pop((chat_id, status_message.message_id), None)
                self._safe_unlink(audio_path)
            try:
                await status_message.edit_text(
                    f"❌ Не удалось загрузить аудио: {exc}\nОтправьте файл ещё раз."
                )
            except Exception:
                pass

    async def _expire_pending(self, key: tuple[int, int]) -> None:
        """Удалить файл, если режим не выбран за PENDING_TTL_SEC."""
        await asyncio.sleep(PENDING_TTL_SEC)
        pending = self._pending_audio.pop(key, None)
        if not pending:
            return
        self._safe_unlink(pending["audio_path"])
        try:
            await pending["status"].edit_text("⏰ Время вышло. Отправьте аудио заново.")
        except Exception:
            pass

    # ---------------------------------------------------------- выбор режима

    async def handle_diarization_choice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if not query or not query.message:
            return

        chat_id = query.message.chat.id
        choice, _, mid = (query.data or "").partition(":")
        key = (chat_id, int(mid)) if mid.isdigit() else (chat_id, query.message.message_id)

        pending = self._pending_audio.pop(key, None)
        if not pending:
            await query.answer("⏰ Время вышло! Отправьте аудио заново.", show_alert=True)
            return
        pending["expire_task"].cancel()

        audio_path = pending["audio_path"]
        message = pending["message"]

        if choice == "diarization_medical":
            diarize, mode_name, num_speakers = True, "Врач-Пациент", 2
            speaker_names = {"speaker_1": "Врач", "speaker_2": "Пациент"}
            icon = "🏥"
        elif choice == "diarization_auto":
            diarize, mode_name, num_speakers, speaker_names = True, "Авто", None, None
            icon = "🚗"
        else:
            diarize, mode_name, num_speakers, speaker_names = False, "Без диаризации", None, None
            icon = "📝"

        await query.answer()
        status_message = await query.message.edit_text(f"{icon} Загрузка модели...")
        user_id = update.effective_user.id

        async def update_progress_text(text: str) -> None:
            try:
                await status_message.edit_text(text)
            except Exception:
                pass

        try:
            start_time = time.time()
            try:
                transcript, stats = await self._transcribe_with_progress(
                    audio_path,
                    diarize=diarize,
                    num_speakers=num_speakers,
                    speaker_names=speaker_names,
                    progress_callback=update_progress_text,
                )
            finally:
                self._safe_unlink(audio_path)

            total_time = stats.get("total_time", time.time() - start_time)
            prep_time = stats.get("prep_time", 0.0)
            transcribe_time = stats.get("transcribe_time", 0.0)
            char_count = stats.get("char_count", len(transcript))

            try:
                await status_message.delete()
            except Exception:
                pass

            self.usage[user_id] = self.usage.get(user_id, 0) + 1

            await message.reply_text(
                f"✅ Режим: {mode_name}\n"
                f"📊 Объём текста: {char_count} символов\n"
                f"⏱ Готово за {total_time:.0f} сек "
                f"(подготовка звука {prep_time:.1f} с, "
                f"распознавание {transcribe_time:.1f} с)"
            )

            if len(transcript) > MAX_INLINE_CHARS:
                await message.reply_document(
                    document=transcript.encode("utf-8"),
                    filename="transcript.txt",
                    caption="Расшифровка длинная, отправляю файлом.",
                )
            else:
                for chunk in split_text_for_telegram(transcript, max_chars=4000):
                    await message.reply_text(chunk)

        except Exception as exc:
            logger.exception("Audio processing failed")
            try:
                await status_message.delete()
            except Exception:
                pass
            await message.reply_text(
                f'❌ Ошибка при обработке в режиме "{mode_name}"\n\n'
                f"Техническая информация: {exc}"
            )

    async def _transcribe_with_progress(
        self,
        audio_path: str,
        *,
        diarize: bool,
        num_speakers: int | None,
        speaker_names: dict[str, str] | None,
        progress_callback: Callable[[str], Awaitable[None]],
    ) -> tuple[str, dict]:
        """Синхронное распознавание в потоке, event loop остаётся свободным."""
        loop = asyncio.get_running_loop()

        def send_progress(msg: str) -> None:
            try:
                asyncio.run_coroutine_threadsafe(progress_callback(msg), loop)
            except Exception:
                pass

        return await asyncio.to_thread(
            self.recognizer.transcribe,
            audio_path,
            diarize=diarize,
            num_speakers=num_speakers,
            speaker_names=speaker_names,
            progress_callback=send_progress,
        )

    # ------------------------------------------------------------- скачивание

    @staticmethod
    def _guess_suffix(media) -> str:
        file_name = getattr(media, "file_name", None)
        if file_name:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in KNOWN_EXT:
                return ext
        return MIME_TO_EXT.get(getattr(media, "mime_type", None) or "", ".ogg")

    async def _download_audio_to_temp(self, media, bot) -> str:
        """Скачать аудио во временный файл.

        <= 20 МБ — через PTB; больше — через Telethon.
        """
        fd, temp_path = tempfile.mkstemp(suffix=self._guess_suffix(media))
        os.close(fd)

        try:
            size = getattr(media, "file_size", None)
            if size is None or size <= BOT_API_DOWNLOAD_LIMIT:
                try:
                    tg_file = await bot.get_file(media.file_id)
                    await tg_file.download_to_drive(temp_path)
                    return temp_path
                except BadRequest as exc:
                    # размер неизвестен, а файл оказался большим
                    if "too big" not in str(exc).lower():
                        raise

            logger.info("Большой файл, скачиваю через Telethon")
            await self.large_files.download(media.file_id, temp_path, size)
            return temp_path
        except BaseException:
            self._safe_unlink(temp_path)
            raise

    @staticmethod
    def _safe_unlink(path: str | None) -> None:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                logger.warning("Не удалось удалить %s", path)

    # ------------------------------------------------------------------ запуск

    def run(self) -> None:
        logger.info("Telegram bot started")
        try:
            self.app.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
            )
        except Conflict:
            logger.error(
                "Конфликт: другой экземпляр бота запущен.\n"
                "Закройте другой процесс с bot.py"
            )
        except KeyboardInterrupt:
            logger.info("Получен KeyboardInterrupt, остановка...")
        finally:
            logger.info("Бот остановлен")


def main() -> None:
    bot = TelegramSpeechBot()
    bot.run()


if __name__ == "__main__":
    main()