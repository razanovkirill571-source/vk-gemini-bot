import hmac
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import vk_api
from google import genai
from google.genai import types
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vk_gemini_bot")

# --- Переменные окружения ---
VK_TOKEN = os.environ["VK_TOKEN"]
VK_GROUP_ID = int(os.environ["VK_GROUP_ID"])
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PROXY_URL = os.environ.get("PROXY_URL", "").strip()

# Доступ к боту:
# ALLOWED_USER_IDS — ID пользователей ВК через запятую (всегда имеют доступ), например: "123456,789012"
# BOT_ACCESS_KEY   — секретный ключ для входа с других аккаунтов командой: /auth <ключ>
# Если обе переменные пустые — доступ открыт всем, кто может написать в сообщество.
ALLOWED_USER_IDS = {
    int(uid.strip())
    for uid in os.environ.get("ALLOWED_USER_IDS", "").split(",")
    if uid.strip().isdigit()
}
BOT_ACCESS_KEY = os.environ.get("BOT_ACCESS_KEY", "").strip()
AUTH_STORAGE_PATH = Path(os.environ.get("AUTH_STORAGE_PATH", "authorized_users.json"))

# --- Актуальные бесплатные модели Gemini API (Free Tier) ---
MODEL_NAME = "gemini-4-hydrogen"
FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

# --- Лимиты под Bothost Базовый (512 МБ – 1 ГБ RAM) ---
MAX_CHATS = 30                   # Максимум активных диалогов в ОЗУ
CHAT_TTL_SECONDS = 6 * 3600      # Автоочистка неактивных диалогов через 6 часов
MAX_HISTORY_MESSAGES = 30        # Максимум сообщений в истории одного чата (15 пар вопрос-ответ)
MAX_PHOTO_BYTES = 10 * 1024 * 1024  # Максимум 10 МБ на одно фото
VK_MESSAGE_LIMIT = 4000

SYSTEM_PROMPT = (
    "Ты отвечаешь в мессенджере ВКонтакте, где НЕТ поддержки LaTeX и Markdown. "
    "Никогда не пиши $, \\frac, \\cdot, ** и т.п. "
    "Записывай дроби в три короткие строки (не длиннее 25 символов в строке, чтобы не переносились на телефоне): "
    "числитель сверху, полоска из ───── посередине, знаменатель снизу; "
    "для выравнивания используй неразрывный пробел. Степени пиши как x², x³."
)

http_opts = types.HttpOptions(client_args={"proxy": PROXY_URL}) if PROXY_URL else None
client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts)
GEN_CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

vk_session = vk_api.VkApi(token=VK_TOKEN)
vk = vk_session.get_api()

# --- Хранилище сессий и блокировок ---
# chat_sessions[peer_id] = {"chat": Chat, "model": str, "last_active": float}
chat_sessions: dict[int, dict] = {}
sessions_lock = threading.Lock()
peer_locks: dict[int, threading.Lock] = {}
auth_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)


def load_authorized_users() -> set[int]:
    if not AUTH_STORAGE_PATH.exists():
        return set()
    try:
        data = json.loads(AUTH_STORAGE_PATH.read_text(encoding="utf-8"))
        return {int(uid) for uid in data if isinstance(uid, int) or str(uid).isdigit()}
    except Exception:
        logger.exception("Не удалось прочитать %s", AUTH_STORAGE_PATH)
        return set()


def save_authorized_users(users: set[int]) -> None:
    try:
        AUTH_STORAGE_PATH.write_text(
            json.dumps(sorted(users), ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Не удалось сохранить %s", AUTH_STORAGE_PATH)


authorized_users: set[int] = load_authorized_users()


def is_access_restricted() -> bool:
    return bool(ALLOWED_USER_IDS or BOT_ACCESS_KEY)


def is_user_authorized(user_id: int) -> bool:
    if not is_access_restricted():
        return True
    if user_id in ALLOWED_USER_IDS:
        return True
    with auth_lock:
        return user_id in authorized_users


def get_peer_lock(peer_id: int) -> threading.Lock:
    with sessions_lock:
        lock = peer_locks.get(peer_id)
        if lock is None:
            lock = threading.Lock()
            peer_locks[peer_id] = lock
        return lock


def trim_history(history: list) -> list:
    if len(history) <= MAX_HISTORY_MESSAGES:
        return list(history)
    trimmed = list(history[-MAX_HISTORY_MESSAGES:])
    # История Gemini должна начинаться с реплики пользователя
    while trimmed and getattr(trimmed[0], "role", None) != "user":
        trimmed.pop(0)
    return trimmed


def cleanup_sessions() -> None:
    now = time.time()
    with sessions_lock:
        expired = [
            pid
            for pid, info in chat_sessions.items()
            if now - info["last_active"] > CHAT_TTL_SECONDS
        ]
        for pid in expired:
            chat_sessions.pop(pid, None)

        while len(chat_sessions) > MAX_CHATS:
            oldest_pid = min(chat_sessions, key=lambda p: chat_sessions[p]["last_active"])
            chat_sessions.pop(oldest_pid, None)


def get_or_create_session(peer_id: int) -> dict:
    cleanup_sessions()
    with sessions_lock:
        session = chat_sessions.get(peer_id)
        if session is None:
            session = {
                "chat": client.chats.create(model=MODEL_NAME, config=GEN_CONFIG),
                "model": MODEL_NAME,
                "last_active": time.time(),
            }
            chat_sessions[peer_id] = session
        else:
            session["last_active"] = time.time()
        return session


def split_message(text: str, limit: int = VK_MESSAGE_LIMIT) -> list[str]:
    if not text:
        return []
    chunks = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = remaining.rfind("\n", 0, limit)
        if cut == -1:
            cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunk = remaining[:cut].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].lstrip("\n ")
    if remaining:
        chunks.append(remaining)
    return chunks


def send_vk_message(peer_id: int, text: str) -> None:
    for chunk in split_message(text):
        try:
            vk.messages.send(
                peer_id=peer_id,
                message=chunk,
                random_id=random.randint(1, 2**31 - 1),
            )
        except Exception:
            logger.exception("Ошибка отправки сообщения в VK (peer_id=%s)", peer_id)
            break


def send_typing(peer_id: int) -> None:
    try:
        vk.messages.setActivity(peer_id=peer_id, type="typing")
    except Exception:
        logger.debug("Не удалось отправить статус typing (peer_id=%s)", peer_id)


def is_retryable_error(err: Exception) -> bool:
    s = str(err).lower()
    markers = (
        "503",
        "429",
        "unavailable",
        "overloaded",
        "high demand",
        "resource_exhausted",
        "quota",
        "rate limit",
    )
    return any(m in s for m in markers)


def ask_gemini_with_retry(peer_id: int, contents: list) -> str:
    session = get_or_create_session(peer_id)
    saved_history = trim_history(session["chat"].get_history())

    # Если история обрезалась или прошлый ответ был от запасной модели,
    # пробуем аккуратно вернуться на основную модель с сохранением контекста.
    if session["model"] != MODEL_NAME or len(saved_history) < len(session["chat"].get_history()):
        primary_chat = client.chats.create(
            model=MODEL_NAME,
            config=GEN_CONFIG,
            history=saved_history,
        )
    else:
        primary_chat = session["chat"]

    last_error: Exception | None = None

    for attempt in range(2):
        try:
            response = primary_chat.send_message(contents)
            text_reply = response.text or "Модель вернула пустой ответ."
            with sessions_lock:
                chat_sessions[peer_id] = {
                    "chat": primary_chat,
                    "model": MODEL_NAME,
                    "last_active": time.time(),
                }
            return f"{text_reply}\n\n[Модель: {MODEL_NAME}]"
        except Exception as e:
            last_error = e
            if is_retryable_error(e):
                logger.warning("Модель %s недоступна (попытка %d): %s", MODEL_NAME, attempt + 1, e)
                if attempt == 0:
                    time.sleep(2)
                continue
            raise

    # Переключение на запасные модели с полным сохранением истории диалога
    for fallback_model in FALLBACK_MODELS:
        try:
            fallback_chat = client.chats.create(
                model=fallback_model,
                config=GEN_CONFIG,
                history=saved_history,
            )
            response = fallback_chat.send_message(contents)
            text_reply = response.text or "Модель вернула пустой ответ."
            with sessions_lock:
                chat_sessions[peer_id] = {
                    "chat": fallback_chat,
                    "model": fallback_model,
                    "last_active": time.time(),
                }
            return f"{text_reply}\n\n[Модель: {fallback_model}]"
        except Exception as e:
            last_error = e
            if is_retryable_error(e):
                logger.warning("Запасная модель %s недоступна: %s", fallback_model, e)
                continue
            raise

    assert last_error is not None
    raise last_error


def download_photo(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        if len(resp.content) > MAX_PHOTO_BYTES:
            logger.warning("Фото пропущено из-за размера: %d байт", len(resp.content))
            return None
        return resp.content
    except Exception:
        logger.exception("Не удалось скачать фото из VK")
        return None


def collect_message_payload(msg: dict, depth: int = 0) -> tuple[list[str], list[str]]:
    """Собирает текст и URL фотографий из сообщения, ответа на него и пересланных сообщений."""
    texts: list[str] = []
    photo_urls: list[str] = []

    text = (msg.get("text") or "").strip()
    if text:
        prefix = "[Пересланное сообщение] " if depth > 0 else ""
        texts.append(f"{prefix}{text}")

    for att in msg.get("attachments", []):
        if att.get("type") == "photo":
            sizes = att["photo"].get("sizes") or []
            if sizes:
                best_photo = max(sizes, key=lambda s: s["width"] * s["height"])
                if best_photo.get("url"):
                    photo_urls.append(best_photo["url"])

    if depth < 3:
        reply_msg = msg.get("reply_message")
        if isinstance(reply_msg, dict):
            r_texts, r_photos = collect_message_payload(reply_msg, depth=depth + 1)
            texts.extend(r_texts)
            photo_urls.extend(r_photos)

        for fwd in msg.get("fwd_messages", []):
            if isinstance(fwd, dict):
                f_texts, f_photos = collect_message_payload(fwd, depth=depth + 1)
                texts.extend(f_texts)
                photo_urls.extend(f_photos)

    return texts, photo_urls


def handle_message(msg: dict) -> None:
    peer_id = msg["peer_id"]
    user_id = msg["from_id"]
    raw_text = (msg.get("text") or "").strip()

    # Игнорируем сообщения от самого сообщества
    if user_id < 0:
        return

    # --- Авторизация по ключу ---
    if raw_text.lower().startswith("/auth"):
        if not BOT_ACCESS_KEY:
            send_vk_message(peer_id, "Авторизация по ключу не включена.")
            return
        parts = raw_text.split(maxsplit=1)
        provided_key = parts[1].strip() if len(parts) > 1 else ""
        if provided_key and hmac.compare_digest(provided_key, BOT_ACCESS_KEY):
            with auth_lock:
                authorized_users.add(user_id)
                save_authorized_users(authorized_users)
            send_vk_message(peer_id, "Доступ открыт! Теперь можно пользоваться ботом.")
        else:
            send_vk_message(peer_id, "Неверный ключ доступа.")
        return

    if raw_text.lower() == "/logout":
        with auth_lock:
            removed = user_id in authorized_users
            authorized_users.discard(user_id)
            if removed:
                save_authorized_users(authorized_users)
        with sessions_lock:
            chat_sessions.pop(peer_id, None)
        send_vk_message(peer_id, "Сессия завершена, контекст очищен.")
        return

    if not is_user_authorized(user_id):
        if BOT_ACCESS_KEY:
            send_vk_message(peer_id, "Доступ ограничен. Введи: /auth <ключ>")
        else:
            send_vk_message(peer_id, "Доступ к этому боту ограничен.")
        return

    if raw_text.lower() in ("/new", "/clear", "сброс"):
        with sessions_lock:
            chat_sessions.pop(peer_id, None)
        send_vk_message(peer_id, "Контекст диалога очищен.")
        return

    lock = get_peer_lock(peer_id)
    with lock:
        texts, photo_urls = collect_message_payload(msg)
        contents: list = []

        if texts:
            contents.append("\n\n".join(texts))

        failed_photos = 0
        for url in photo_urls:
            img_bytes = download_photo(url)
            if img_bytes:
                contents.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                )
            else:
                failed_photos += 1

        if not contents:
            if failed_photos:
                send_vk_message(peer_id, "Не удалось загрузить фото из ВКонтакте. Попробуй отправить ещё раз.")
            return

        send_typing(peer_id)
        try:
            reply = ask_gemini_with_retry(peer_id, contents)
            send_vk_message(peer_id, reply)
        except Exception:
            logger.exception("Ошибка при запросе к Gemini (peer_id=%s, user_id=%s)", peer_id, user_id)
            send_vk_message(
                peer_id,
                "Сейчас не удалось получить ответ от Gemini (возможно, перегрузка или лимит). Попробуй через минуту.",
            )


def main() -> None:
    logger.info("Бот запущен и слушает сообщения... Основная модель: %s", MODEL_NAME)
    while True:
        try:
            longpoll = VkBotLongPoll(vk_session, VK_GROUP_ID)
            for event in longpoll.listen():
                if event.type != VkBotEventType.MESSAGE_NEW:
                    continue
                msg = event.object.message
                executor.submit(handle_message, msg)
        except Exception:
            logger.exception("Сбой LongPoll, переподключаюсь через 5 секунд...")
            time.sleep(5)


if __name__ == "__main__":
    main()
