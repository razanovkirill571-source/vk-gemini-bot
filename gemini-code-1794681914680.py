import os
import random
import time
import requests
import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from google import genai
from google.genai import types

VK_TOKEN = os.environ["VK_TOKEN"]
VK_GROUP_ID = int(os.environ["VK_GROUP_ID"])
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

MODEL_NAME = "gemini-4-hydrogen"
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash"]

SYSTEM_PROMPT = (
    "Ты отвечаешь в мессенджере ВКонтакте, где НЕТ поддержки LaTeX и Markdown. "
    "Никогда не пиши $, \\frac, \\cdot, ** и т.п. "
    "Записывай дроби в три короткие строки (не длиннее 25 символов в строке, чтобы не переносились на телефоне): "
    "числитель сверху, полоска из ───── посередине, знаменатель снизу; "
    "для выравнивания используй неразрывный пробел. Степени пиши как x², x³."
)

PROXY_URL = os.environ.get("PROXY_URL", "")
http_opts = types.HttpOptions(client_args={"proxy": PROXY_URL}) if PROXY_URL else None
client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts)
GEN_CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

vk_session = vk_api.VkApi(token=VK_TOKEN)
vk = vk_session.get_api()
longpoll = VkBotLongPoll(vk_session, VK_GROUP_ID)

user_chats = {}


def get_chat(user_id):
    if user_id not in user_chats:
        user_chats[user_id] = client.chats.create(model=MODEL_NAME, config=GEN_CONFIG)
    return user_chats[user_id]


def send_vk_message(peer_id, text):
    for i in range(0, len(text), 4000):
        vk.messages.send(
            peer_id=peer_id,
            message=text[i : i + 4000],
            random_id=random.randint(1, 2**31 - 1),
        )


def is_overloaded(err):
    s = str(err).lower()
    return "503" in s or "unavailable" in s or "overloaded" in s or "high demand" in s


def ask_gemini_with_retry(user_id, contents):
    chat = get_chat(user_id)
    last_error = None

    for attempt in range(2):
        try:
            response = chat.send_message(contents)
            text_reply = response.text or "Модель вернула пустой ответ."
            return f"{text_reply}\n\n[Модель: {MODEL_NAME}]"
        except Exception as e:
            last_error = e
            if is_overloaded(e):
                time.sleep(2)
                continue
            raise

    for fallback_model in FALLBACK_MODELS:
        try:
            resp = client.models.generate_content(
                model=fallback_model,
                contents=contents,
                config=GEN_CONFIG,
            )
            text_reply = resp.text or "Модель вернула пустой ответ."
            return f"{text_reply}\n\n[Модель: {fallback_model}]"
        except Exception as e:
            last_error = e
            if is_overloaded(e):
                continue
            raise

    raise last_error


print("Бот запущен и слушает сообщения...", flush=True)

for event in longpoll.listen():
    if event.type != VkBotEventType.MESSAGE_NEW:
        continue

    msg = event.object.message
    peer_id = msg["peer_id"]
    user_id = msg["from_id"]
    text = (msg.get("text") or "").strip()

    if text.lower() in ("/new", "/clear", "сброс"):
        user_chats.pop(user_id, None)
        send_vk_message(peer_id, "Контекст диалога очищен.")
        continue

    contents = [text] if text else []

    for att in msg.get("attachments", []):
        if att.get("type") == "photo":
            sizes = att["photo"]["sizes"]
            best_photo = max(sizes, key=lambda s: s["width"] * s["height"])
            img_bytes = requests.get(best_photo["url"], timeout=15).content
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

    if not contents:
        continue

    try:
        reply = ask_gemini_with_retry(user_id, contents)
        send_vk_message(peer_id, reply)
    except Exception as e:
        send_vk_message(peer_id, f"Ошибка при запросе к Gemini: {e}")
