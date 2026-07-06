import cv2
import numpy as np
import base64
import io
from PIL import Image
from pyzbar.pyzbar import decode

def extract_url_from_qr(image_base64: str) -> dict:
    """
    Decodes a QR code image and extracts the URL inside it.
    Uses pyzbar for highly robust QR code detection.
    Input: base64 encoded image string.
    """
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(image)
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Attempt decoding
        decoded_objects = decode(img_gray)
        
        # Fallback 1: Thresholding
        if not decoded_objects:
            _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            decoded_objects = decode(img_thresh)
            
        # Fallback 2: Inverted image
        if not decoded_objects:
            img_inv = cv2.bitwise_not(img_gray)
            decoded_objects = decode(img_inv)

        if not decoded_objects:
            return {"url": None, "error": "No QR code found in image"}

        data = decoded_objects[0].data.decode('utf-8')

        if data.startswith("http") or data.startswith("www"):
            return {"url": data, "error": None}
        else:
            return {"url": data, "error": "QR content is not a URL"}

    except Exception as e:
        return {"url": None, "error": str(e)}