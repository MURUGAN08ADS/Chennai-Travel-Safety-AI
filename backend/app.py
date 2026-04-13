from flask import Flask, jsonify, request
from flask_cors import CORS
from safety_score import get_safety_score
from gpt_chat import ask_travel_assistant

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Travel Safety AI is running!"})

@app.route("/safety", methods=["GET"])
def safety():
    area = request.args.get("area", "")
    if not area:
        return jsonify({"error": "Please provide an area name"}), 400
    result = get_safety_score(area)
    return jsonify(result)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Please provide a question"}), 400
    answer = ask_travel_assistant(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)