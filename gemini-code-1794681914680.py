import os
import random
import requests
import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from google import genai
from google.genai import types

VK_TOKEN = os.environ["VK_TOKEN"]
VK_GROUP_ID = int(os.environ["VK_GROUP_ID"])
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_NAME = "gemini-3.8-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

vk_session = vk_api.VkApi(token=VK_TOKEN)
vk = vk_session.get_api()
longpoll = VkBotLongPoll(vk_session, VK_GROUP_ID)

user_chats = {}


def get_chat(user_id):
    if user_id not in user_chats:
        user_chats[user_id] = client.chats.create(model=MODEL_NAME)
    return user_chats[user_id]


def send_vk_message(peer_id, text):
    for i in range(0, len(text), 4000):
        vk.messages.send(
            peer_id=peer_id,
            message=text[i : i + 4000],
            random_id=random.randint(1, 2**31 - 1),
        )


print("Бот запущен и слушает сообщения...")

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
        chat = get_chat(user_id)
        response = chat.send_message(contents)
        reply = response.text or "Модель вернула пустой ответ."
        send_vk_message(peer_id, reply)
    except Exception as e:
        send_vk_message(peer_id, f"Ошибка при запросе к Gemini: {e}")