import os
from math import asin, cos, radians, sin, sqrt
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from safety_score import get_safety_score
from gpt_chat import ask_travel_assistant

app = Flask(__name__)
allowed_origin = os.getenv("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": allowed_origin}})

EMERGENCY_POIS = [
    {
        "name": "Adyar Police Station",
        "type": "police",
        "lat": 13.0034,
        "lng": 80.2571,
        "area": "Adyar",
    },
    {
        "name": "Egmore Police Station",
        "type": "police",
        "lat": 13.0800,
        "lng": 80.2600,
        "area": "Egmore",
    },
    {
        "name": "Apollo Hospital",
        "type": "hospital",
        "lat": 13.0569,
        "lng": 80.2521,
        "area": "Greams Road",
    },
    {
        "name": "Fortis Malar",
        "type": "hospital",
        "lat": 13.0007,
        "lng": 80.2541,
        "area": "Adyar",
    },
    {
        "name": "MIOT International",
        "type": "hospital",
        "lat": 13.0107,
        "lng": 80.1759,
        "area": "Manapakkam",
    },
]


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    a = sin(d_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(d_lng / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def _extract_location(payload: dict[str, Any]) -> tuple[float, float] | None:
    lat = payload.get("lat")
    lng = payload.get("lng")

    if lat is None or lng is None:
        return None

    try:
        lat_value = float(lat)
        lng_value = float(lng)
    except (TypeError, ValueError):
        return None

    if not (-90 <= lat_value <= 90 and -180 <= lng_value <= 180):
        return None
    return lat_value, lng_value


def _nearest_emergency_poi(lat: float, lng: float, poi_type: str) -> dict[str, Any]:
    options = [poi for poi in EMERGENCY_POIS if poi["type"] == poi_type]
    nearest = min(
        options,
        key=lambda poi: _haversine_km(lat, lng, poi["lat"], poi["lng"]),
    )

    distance = _haversine_km(lat, lng, nearest["lat"], nearest["lng"])
    return {
        "name": nearest["name"],
        "type": nearest["type"],
        "area": nearest["area"],
        "lat": nearest["lat"],
        "lng": nearest["lng"],
        "distance_km": round(distance, 2),
    }

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
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    runtime_context = data.get("context")
    language = data.get("language")

    if not isinstance(question, str):
        return jsonify({"error": "Question must be a string"}), 400

    if runtime_context is not None and not isinstance(runtime_context, dict):
        return jsonify({"error": "context must be a JSON object"}), 400

    if language is not None and not isinstance(language, str):
        return jsonify({"error": "language must be a string"}), 400

    question = question.strip()
    if not question:
        return jsonify({"error": "Please provide a question"}), 400

    if len(question) > 1500:
        return jsonify({"error": "Question is too long (max 1500 characters)"}), 400

    answer = ask_travel_assistant(
        question,
        runtime_context=runtime_context,
        preferred_language=language,
    )
    return jsonify({"answer": answer})


@app.route("/sos", methods=["POST"])
def sos():
    data = request.get_json(silent=True) or {}
    reason = data.get("reason", "manual_sos")
    language = data.get("language")
    context = data.get("context")
    contacts = data.get("contacts")

    if reason is not None and not isinstance(reason, str):
        return jsonify({"error": "reason must be a string"}), 400

    if language is not None and not isinstance(language, str):
        return jsonify({"error": "language must be a string"}), 400

    if context is not None and not isinstance(context, dict):
        return jsonify({"error": "context must be a JSON object"}), 400

    if contacts is not None and not isinstance(contacts, list):
        return jsonify({"error": "contacts must be a list"}), 400

    location_payload = data.get("location")
    if location_payload is not None and not isinstance(location_payload, dict):
        return jsonify({"error": "location must be a JSON object"}), 400

    location = _extract_location(location_payload or {})
    if not location and isinstance(context, dict):
        gps_context = context.get("gps")
        if isinstance(gps_context, dict):
            location = _extract_location(gps_context)

    if not location:
        return jsonify({"error": "Valid location (lat,lng) is required"}), 400

    lat, lng = location
    nearest_police = _nearest_emergency_poi(lat, lng, "police")
    nearest_hospital = _nearest_emergency_poi(lat, lng, "hospital")

    sos_context: dict[str, Any] = dict(context or {})
    sos_context.update(
        {
            "gps": {"lat": lat, "lng": lng},
            "emergency_mode": True,
            "sos_reason": reason,
            "nearest_police": nearest_police,
            "nearest_hospital": nearest_hospital,
        }
    )

    guidance_prompt = (
        "I need immediate travel safety guidance in Chennai. "
        f"Reason: {reason}. Provide urgent next actions, location-sharing advice, "
        "and mention nearest police/hospital priorities."
    )
    guidance = ask_travel_assistant(
        guidance_prompt,
        runtime_context=sos_context,
        preferred_language=language,
    )

    contact_count = len(contacts) if isinstance(contacts, list) else 0
    return jsonify(
        {
            "status": "sos_triggered",
            "reason": reason,
            "location": {"lat": lat, "lng": lng},
            "nearest_police": nearest_police,
            "nearest_hospital": nearest_hospital,
            "share_live_location_recommended": True,
            "contacts_requested": contact_count,
            "contact_dispatch": "not_configured",
            "guidance": guidance,
        }
    )

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5001"))
    app.run(host=host, port=port, debug=debug_mode)