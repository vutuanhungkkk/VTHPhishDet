import base64
import numpy as np
import cv2
import os

# Bypass SSL issues on Windows when paddleocr tries to import aiohttp and download models
import ssl
try:
    ssl.create_default_context = ssl._create_unverified_context
except AttributeError:
    pass

# Attempt to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    # Initialize PaddleOCR with Vietnamese support (also works for English and numbers)
    print("Initializing PaddleOCR with Vietnamese support...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,   # replaces old use_angle_cls=True
        lang='vi'
    )
except ImportError:
    print("paddleocr not installed. Please install it.")
    ocr = None
except Exception as e:
    print(f"Error loading PaddleOCR: {e}")
    ocr = None

def extract_text_from_image(image_b64: str) -> dict:
    if not ocr:
        return {"text": None, "error": "OCR model is not initialized or installed."}

    try:
        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"text": None, "error": "Failed to decode image"}

        # PaddleOCR 3.x: use predict() instead of ocr(det=..., rec=...)
        result = ocr.predict(img)

        extracted_text = []
        if result:
            for res in result:
                # res behaves like a dict; 'rec_texts' holds the recognized strings
                rec_texts = res.get("rec_texts") if hasattr(res, "get") else getattr(res, "rec_texts", [])
                if rec_texts:
                    extracted_text.extend(rec_texts)

        full_text = "\n".join(extracted_text).strip()

        if not full_text:
            return {"text": None, "error": "No text found in image"}

        return {"text": full_text}
    except Exception as e:
        return {"text": None, "error": f"OCR extraction error: {str(e)}"}
