import base64
import io
import json
import re
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

from config import VISION_MAX_NEW_TOKENS, VISION_MODEL_ID, VISION_PROMPT, VISION_USE_4BIT

try:
    from transformers import AutoModelForImageTextToText as VisionModelClass
except ImportError:  # pragma: no cover
    try:
        from transformers import AutoModelForVision2Seq as VisionModelClass
    except ImportError:  # pragma: no cover
        from transformers import AutoModelForCausalLM as VisionModelClass


def _decode_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _extract_json(text: str) -> dict:
    if match := re.search(r"\{.*\}", text, re.S):
        return json.loads(match.group(0))
    raise ValueError("Model did not return JSON")


def _build_inputs(processor, image: Image.Image, prompt: str):
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(text=chat_prompt, images=image, return_tensors="pt")

    return processor(text=prompt, images=image, return_tensors="pt")


def _infer_image_score(image_base64: str, url: str = "") -> dict:
    processor, model = _load_model()
    image = _decode_image(image_base64)

    prompt = VISION_PROMPT
    if url:
        prompt = f"{VISION_PROMPT} The page URL is: {url}"

    inputs = _build_inputs(processor, image, prompt)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=VISION_MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
            "model_id": VISION_MODEL_ID,
        },
        "error": None,
    }


@lru_cache(maxsize=1)
def _load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("Local 4-bit vision inference requires a CUDA GPU")

    quant_config = None
    if VISION_USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, trust_remote_code=True)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = VisionModelClass.from_pretrained(VISION_MODEL_ID, **model_kwargs)
    model.eval()
    return processor, model


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