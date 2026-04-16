from flask import Flask, request, jsonify
import requests
import uuid
import os

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

    try:
        data = request.get_json(silent=True) or {}

        match_id = data.get("matchId")
        game_number = data.get("gameNumber")
        screenshot_url = data.get("screenshotUrl")

        if not match_id or game_number is None or not screenshot_url:
            return jsonify({
                "success": False,
                "error": "Missing matchId, gameNumber, or screenshotUrl",
            }), 400

        filename = f"{uuid.uuid4()}.png"
        temp_filepath = os.path.join(UPLOAD_DIR, filename)

        image_response = requests.get(screenshot_url, timeout=30)
        image_response.raise_for_status()

        with open(temp_filepath, "wb") as f:
            f.write(image_response.content)

        payload = league_parser.parse_image(
            image_path=temp_filepath,
            match_id=match_id,
            game_number=game_number,
        )

        ingest_response = requests.post(
            INGEST_API_URL,
            json=payload,
            timeout=60,
        )

        response_body = None
        try:
            response_body = ingest_response.json()
        except Exception:
            response_body = ingest_response.text

        ingest_response.raise_for_status()

        return jsonify({
            "success": True,
            "payload": payload,
            "ingestStatus": ingest_response.status_code,
            "ingestResponse": response_body,
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500

    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
