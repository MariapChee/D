#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import requests
import typing
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from g4f.client import Client

app = Flask(__name__)

# GPT-4 Free (g4f)
def gpt_4_free_client(messages: typing.List[typing.Dict[str, str]]) -> typing.Union[str, None]:
    try:
        client = Client()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        return response.choices[0].message.content
    except:
        return None

# Remix AI fallback
def remix_ai(messages: typing.List[typing.Dict[str, str]]) -> typing.Union[str, None]:
    url = "https://openai-gpt.remixproject.org/"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "user-agent": "Mozilla/5.0"
    }
    data = {"prompt": str(messages)}
    try:
        result = requests.post(url, json=data, headers=headers)
        return result.json()['choices'][0]['message']['content']
    except:
        return None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get("message")
    messages = [{"role": "user", "content": user_input}]
    response = gpt_4_free_client(messages) or remix_ai(messages)

    if not response:
        return jsonify({"error": "AI failed to respond."})

    # Time-based correction
    now = datetime.now()
    current_year = str(now.year)
    current_month = now.strftime("%B")
    current_day = str(now.day)

    # Replace outdated years with current year
    response = re.sub(r'\b(in|on|by|before|after|during|until)?\s?(20(1[5-9]|2[0-4]))\b',
                      lambda m: f"{m.group(1) + ' ' if m.group(1) else ''}{current_year}",
                      response)
    for y in ['2021', '2022', '2023', '2024']:
        response = re.sub(rf'\b{y}\b', current_year, response)

    # Replace months and days
    month_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b'
    response = re.sub(month_pattern, current_month, response)
    response = re.sub(r'\b(0?[1-9]|[12][0-9]|3[01])(st|nd|rd|th)?\b', current_day, response)

    return jsonify({"content": response})
