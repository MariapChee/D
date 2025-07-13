#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template, jsonify
from g4f.client import Client
import typing
import os

app = Flask(__name__)

# GPT-4 free function
def gpt_4_free_client(messages: typing.List[typing.Dict[str, str]]) -> typing.Union[str, None]:
    client = Client()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        return response.choices[0].message.content
    except:
        return None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get("message")
    conversation = [{"role": "user", "content": user_input}]
    result = gpt_4_free_client(conversation)
    if result:
        return jsonify({"content": result})
    else:
        return jsonify({"error": "AI failed to respond."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
