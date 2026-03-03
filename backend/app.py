from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os

app = Flask(__name__, template_folder='../frontend')
CORS(app) # Taake aapka frontend backend se connect ho sakay

# Model load karein
model = pickle.load(open("model.pkl", "rb"))

@app.route("/stats", methods=["GET"])
def get_stats():
    # Accuracy read karein
    with open("accuracy.txt", "r") as f:
        acc = f.read()
    return jsonify({
        "accuracy": acc,
        "graph_url": "http://127.0.0.1:5000/static/boundary_plot.png"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Prediction logic yahan dalo
    return jsonify({"prediction": 1}) 

# Ye zaroori hai graph dikhane ke liye
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route("/")
def home():
    # Ab 404 nahi aayega, balki index.html load hogi
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)