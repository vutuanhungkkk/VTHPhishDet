from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)
BACKEND_URL = "http://127.0.0.1:8000"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/scan/url", methods=["POST"])
def scan_url():
    url = (request.json or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        r = requests.post(f"{BACKEND_URL}/scan/url", json={"url": url}, timeout=30)
        result = r.json()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/email", methods=["POST"])
def scan_email():
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No email text"}), 400
    try:
        r = requests.post(f"{BACKEND_URL}/scan/email", json={"text": text}, timeout=30)
        result = r.json()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/image", methods=["POST"])
def scan_image():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        files = {"file": (file.filename, file.read(), file.content_type)}
        resp = requests.post(f"{BACKEND_URL}/scan/image", files=files, timeout=120)
        result = resp.json()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/email_image", methods=["POST"])
def scan_email_image():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        files = {"file": (file.filename, file.read(), file.content_type)}
        resp = requests.post(f"{BACKEND_URL}/scan/email_image", files=files, timeout=120)
        result = resp.json()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/scan/qr", methods=["POST"])
def scan_qr():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file"}), 400
        files = {"file": (file.filename, file.read(), file.content_type)}
        r = requests.post(f"{BACKEND_URL}/scan/qr", files=files, timeout=30)
        result = r.json()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)