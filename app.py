#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import uuid
import json
import secrets
from datetime import datetime
from flask import Flask, request, session, render_template, jsonify, redirect, url_for
from flask_session import Session
from g4f.client import Client
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytesseract

# === Configuration ===
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = secrets.token_hex(32)
Session(app)

# Create folders and db if missing
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
if not os.path.exists('chat_history.json'):
    with open('chat_history.json', 'w') as f:
        json.dump({}, f)

# === Load image model ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# === Utility Functions ===
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_db():
    with open('chat_history.json') as f:
        return json.load(f)

def save_db(data):
    with open('chat_history.json', 'w') as f:
        json.dump(data, f, indent=2)

def gpt_respond(messages):
    try:
        client = Client()
        result = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        reply = result.choices[0].message.content.strip()
        year_now = str(datetime.now().year)
        return reply.replace("2023", year_now).replace("2022", year_now)
    except Exception as e:
        return f"❌ GPT Error: {e}"

def generate_title(messages):
    title_prompt = [{
        "role": "user",
        "content": "Give a short descriptive title for this conversation:\n" +
                   "\n".join(f"{m['role']}: {m['content']}" for m in messages[:2])
    }]
    return gpt_respond(title_prompt)[:40] or "Untitled Chat"

def get_caption(path):
    image = Image.open(path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def get_ocr(path):
    return pytesseract.image_to_string(Image.open(path))

# === Flask Routes ===

@app.before_request
def ensure_session():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

@app.route('/')
def index():
    db = load_db()
    user_chats = db.get(session['user_id'], [])
    return render_template("index.html", chats=user_chats)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    msg = data.get("message")
    chat_id = data.get("chat_id")

    if not msg:
        return jsonify({"error": "Empty message"})

    db = load_db()
    user_id = session['user_id']
    chats = db.setdefault(user_id, [])

    chat = next((c for c in chats if c['id'] == chat_id), None)
    if not chat:
        chat = {"id": str(uuid.uuid4()), "title": "", "messages": []}
        chats.append(chat)

    user_msg = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": msg,
        "timestamp": now()
    }
    chat['messages'].append(user_msg)

    messages_for_gpt = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]
    ai_reply = gpt_respond(messages_for_gpt)

    bot_msg = {
        "id": str(uuid.uuid4()),
        "role": "bot",
        "content": ai_reply,
        "timestamp": now()
    }
    chat['messages'].append(bot_msg)

    if not chat["title"]:
        chat["title"] = generate_title(chat["messages"])

    save_db(db)
    return jsonify({
        "content": bot_msg["content"],
        "timestamp": bot_msg["timestamp"],
        "chat_id": chat["id"],
        "title": chat["title"],
        "bot_msg_id": bot_msg["id"]
    })

@app.route('/history/<chat_id>')
def history(chat_id):
    db = load_db()
    user_id = session['user_id']
    chats = db.get(user_id, [])
    chat = next((c for c in chats if c["id"] == chat_id), None)
    return jsonify(chat.get("messages", []) if chat else [])

@app.route('/edit/<chat_id>/<msg_id>', methods=['POST'])
def edit(chat_id, msg_id):
    db = load_db()
    new_text = request.json.get("new_content")
    user_id = session['user_id']
    for chat in db.get(user_id, []):
        if chat["id"] == chat_id:
            for msg in chat["messages"]:
                if msg["id"] == msg_id and msg["role"] == "user":
                    msg["content"] = new_text
                    msg["timestamp"] = now()
                    save_db(db)
                    return jsonify({"success": True, "new_content": new_text, "timestamp": msg["timestamp"]})
    return jsonify({"error": "Message not found"})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."})

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        caption = get_caption(filepath)
        ocr = get_ocr(filepath)
        prompt = f"Image details:\nCaption: {caption}\nOCR Text: {ocr}\nDescribe the image meaningfully."
        result = gpt_respond([{"role": "user", "content": prompt}])
        return jsonify({"caption": caption, "ocr": ocr, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/delete/<chat_id>', methods=['POST'])
def delete(chat_id):
    db = load_db()
    user_id = session['user_id']
    db[user_id] = [c for c in db.get(user_id, []) if c['id'] != chat_id]
    save_db(db)
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
