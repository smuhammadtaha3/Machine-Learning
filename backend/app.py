from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='../frontend')
CORS(app) 

# Model load karein
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error: {e}")

# Polynomial mapping function
def map_feature_single(x1, x2):
    degree = 6
    out = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((x1**(i - j) * (x2**j)))
    return np.array(out)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        raw_x1 = float(data["exam1"])
        raw_x2 = float(data["exam2"])
        
        # Scaling logic for 40+ scores
        x1_scaled = (raw_x1 / 60.0) - 0.5 
        x2_scaled = (raw_x2 / 60.0) - 0.5
        
        # Prediction calculation
        x_mapped = map_feature_single(x1_scaled, x2_scaled)
        z = np.dot(x_mapped, model["weights"]) + model["bias"]
        prob = 1 / (1 + np.exp(-z)) 
        prediction = 1 if prob >= 0.5 else 0

        # --- DYNAMIC PLOT WITH BACKGROUND DATA ---
        plt.figure(figsize=(6, 5))
        
        # 1. Background Data Load aur Plot karein
        # Make sure ex2data2.txt is in the same folder or provide full path
        try:
            data_points = np.loadtxt("../model_training/ex2data2.txt", delimiter=',')
            X_bg, y_bg = data_points[:, :2], data_points[:, 2]
            pos = y_bg == 1
            neg = y_bg == 0
            plt.scatter(X_bg[pos, 0], X_bg[pos, 1], marker='+', c='black', label='y=1 (Admitted)')
            plt.scatter(X_bg[neg, 0], X_bg[neg, 1], marker='o', c='y', edgecolors='k', label='y=0 (Not Admitted)')
        except:
            print("Warning: Background data file not found.")

        # 2. Decision Boundary (Green Line)
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z_grid = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                mapped = map_feature_single(u[i], v[j])
                z_grid[i,j] = np.dot(mapped, model["weights"]) + model["bias"]
        plt.contour(u, v, z_grid.T, levels=[0], colors="green")

        # 3. User Point (Bright Red highlighted)
        plt.scatter(x1_scaled, x2_scaled, c='red', s=200, edgecolors='white', linewidth=2, label="Your Input", zorder=5)
        
        plt.title(f"Live Position: {'Admitted' if prediction == 1 else 'Not Admitted'}")
        plt.xlabel("Microchip Test 1 (Scaled)")
        plt.ylabel("Microchip Test 2 (Scaled)")
        plt.legend(loc="upper right", fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.3)

        graph_path = os.path.join("static", "user_prediction.png")
        plt.savefig(graph_path)
        plt.close()

        return jsonify({
            "prediction": int(prediction),
            "new_graph_url": "http://127.0.0.1:5000/static/user_prediction.png"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/stats", methods=["GET"])
def get_stats():
    try:
        with open("accuracy.txt", "r") as f:
            acc = "".join(filter(lambda x: x.isdigit() or x == '.', f.read()))
        return jsonify({
            "accuracy": acc, 
            "default_graph": "http://127.0.0.1:5000/static/boundary_plot.png"
        })
    except:
        return jsonify({"error": "Stats not found"}), 404

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True)