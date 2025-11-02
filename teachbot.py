from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/nvidia/nemotron-nano-12b-v2-vl"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route('/')
def home():
    return "Chat Teaching Bot Running!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    output = query({"inputs": user_input})
    return jsonify({"response": output[0]["generated_text"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
