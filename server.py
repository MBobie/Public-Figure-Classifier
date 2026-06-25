import os
import json
import base64
import joblib
import numpy as np
import cv2
from wavelet import w2d
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=".")

# ── Load artifacts once at startup ───────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "artifacts/class_dictionary.json")) as f:
    CLASS_NAME_TO_NUM = json.load(f)
    CLASS_NUM_TO_NAME = {v: k for k, v in CLASS_NAME_TO_NUM.items()}

MODEL = joblib.load(os.path.join(BASE, "artifacts/saved_model.pkl"))

FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(BASE, "opencv/haarcascades/haarcascade_frontalface_default.xml"))
EYE_CASCADE  = cv2.CascadeClassifier(
    os.path.join(BASE, "opencv/haarcascades/haarcascade_eye.xml"))

print("✅ Model and artifacts loaded.")

# ── Preprocessing helpers ─────────────────────────────────────────────────────
def b64_to_cv2(b64str):
    encoded = b64str.split(",")[1] if "," in b64str else b64str
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def get_cropped_faces(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
    cropped = []
    for (x, y, w, h) in faces:
        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped.append(roi_color)
    return cropped

def classify_image(b64str):
    img   = b64_to_cv2(b64str)
    faces = get_cropped_faces(img)

    if not faces:
        return [{"error": "Could not detect a face with two eyes. Try a clearer front-facing photo."}]

    results = []
    for face in faces:
        raw  = cv2.resize(face, (32, 32))
        har  = w2d(face, 'db1', 5)
        har  = cv2.resize(har, (32, 32))
        feat = np.vstack((raw.reshape(32*32*3, 1), har.reshape(32*32, 1)))
        feat = feat.reshape(1, 32*32*3 + 32*32).astype(float)

        pred  = MODEL.predict(feat)[0]
        proba = MODEL.predict_proba(feat)

        results.append({
            "class": CLASS_NUM_TO_NAME[pred],
            "class_probability": np.around(proba * 100, 2).tolist()[0],
            "class_dictionary": CLASS_NAME_TO_NUM,
        })
    return results

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "app.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

@app.route("/classify_image", methods=["POST"])
def classify():
    b64 = request.form.get("image_data", "")
    if not b64:
        return jsonify({"error": "No image data received."}), 400
    try:
        result = classify_image(b64)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    resp = jsonify(result)
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
