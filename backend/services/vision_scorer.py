import base64
import io
import json
import re
from PIL import Image
import google.generativeai as genai

from config import VISION_PROMPT, GEMINI_API_KEY, GEMINI_MODEL_VERSION


def _decode_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _extract_json(text: str) -> dict:
    if match := re.search(r"\{.*\}", text, re.S):
        return json.loads(match.group(0))
    raise ValueError("Model did not return JSON")


def _infer_image_score(image_base64: str, url: str = "") -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in config. Please set it to use the vision model.")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL_VERSION)
    
    image = _decode_image(image_base64)

    prompt = VISION_PROMPT
    if url:
        prompt = f"{VISION_PROMPT}\nThe page URL is: {url}"

    response = model.generate_content([prompt, image])
    generated_text = response.text
    
    parsed = _extract_json(generated_text)
    risk_score = float(parsed.get("risk_score", 0.5))
    risk_score = max(0.0, min(1.0, risk_score))
    label = str(parsed.get("label", "phishing" if risk_score >= 0.5 else "safe")).lower()
    rationale = str(parsed.get("rationale", ""))

    return {
        "image_score": round(risk_score, 4),
        "vision_response": {
            "label": label,
            "risk_score": round(risk_score, 4),
            "rationale": rationale,
            "raw_output": generated_text,
            "model_id": GEMINI_MODEL_VERSION,
        },
        "error": None,
    }


def score_image(image_base64: str, url: str = "") -> dict:
    if not image_base64:
        return {
            "image_score": None,
            "vision_response": None,
            "error": "No image provided",
        }

    try:
        return _infer_image_score(image_base64, url=url)
    except Exception as exc:
        return {
            "image_score": None,
            "vision_response": None,
            "error": str(exc),
        }