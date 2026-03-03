from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Admission Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    exam1 = data["exam1"]
    exam2 = data["exam2"]

    prediction = model.predict([[exam1, exam2]])

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)