from flask import Flask, request, render_template, jsonify
import requests

app = Flask(__name__)

API_URL = "https://oi-vscode-server-2.onrender.com/v1/chat/completions"

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get("message")
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": user_input}],
        "stream": False
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
