from flask import Flask, request, jsonify
import requests
import uuid
import os
import time

import league_parser

app = Flask(__name__)

UPLOAD_DIR = "temp_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

INGEST_API_URL = os.environ.get(
    "INGEST_API_URL",
    "https://eclchina.lol/api/ingest-match-game",
)


@app.route("/", methods=["GET"])
def health():
    return "OCR Service Running", 200


@app.route("/ocr", methods=["POST"])
def run_ocr():
    temp_filepath = None
    started_at = time.time()

    try:
        print("📥 /ocr request received", flush=True)

        data = request.get_json(silent=True) or {}
        print("📦 Request JSON:", data, flush=True)

        match_id = data.get("matchId")
        game_number = data.get("gameNumber")
        screenshot_url = data.get("screenshotUrl")

        if not match_id or game_number is None or not screenshot_url:
            print("❌ Missing required fields", flush=True)
            return jsonify({
                "success": False,
                "error": "Missing matchId, gameNumber, or screenshotUrl",
            }), 400

        filename = f"{uuid.uuid4()}.jpg"
        temp_filepath = os.path.join(UPLOAD_DIR, filename)

        print("🌐 Downloading image:", screenshot_url, flush=True)
        image_response = requests.get(
            screenshot_url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 ECL-OCR-Service"},
        )
        image_response.raise_for_status()

        with open(temp_filepath, "wb") as f:
            f.write(image_response.content)

        print("✅ Image downloaded:", temp_filepath, flush=True)

        print("🧠 Starting OCR parse...", flush=True)
        payload = league_parser.parse_image(
            image_path=temp_filepath,
            match_id=match_id,
            game_number=game_number,
        )
        print("✅ OCR parse finished", flush=True)

        print("📤 Posting payload to ingest API:", INGEST_API_URL, flush=True)
        ingest_response = requests.post(
            INGEST_API_URL,
            json=payload,
            timeout=60,
        )

        try:
            response_body = ingest_response.json()
        except Exception:
            response_body = ingest_response.text

        print("📨 Ingest status:", ingest_response.status_code, flush=True)
        print("📨 Ingest response:", response_body, flush=True)

        ingest_response.raise_for_status()

        total_time = round(time.time() - started_at, 2)
        print(f"✅ /ocr finished in {total_time}s", flush=True)

        return jsonify({
            "success": True,
            "payload": payload,
            "ingestStatus": ingest_response.status_code,
            "ingestResponse": response_body,
            "elapsedSeconds": total_time,
        }), 200

    except Exception as e:
        total_time = round(time.time() - started_at, 2)
        print(f"❌ /ocr failed after {total_time}s: {str(e)}", flush=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "elapsedSeconds": total_time,
        }), 500

    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                print("🧹 Temp file removed:", temp_filepath, flush=True)
            except Exception as cleanup_err:
                print("⚠️ Failed to remove temp file:", str(cleanup_err), flush=True)
