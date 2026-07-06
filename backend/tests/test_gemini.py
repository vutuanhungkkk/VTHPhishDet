import os
from PIL import Image, ImageDraw
import google.generativeai as genai
from config import VISION_PROMPT

# Fake phishing image
img = Image.new('RGB', (800, 600), color=(240, 242, 245))
d = ImageDraw.Draw(img)
d.text((320, 100), 'Faceboook', fill=(24, 119, 242)) # typo
d.text((270, 420), 'Warning: Your account will be suspended if you do not log in immediately!', fill=(255, 0, 0))

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: GEMINI_API_KEY environment variable is not set.")
    print("Please set it in your terminal before running this script.")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

print("Sending test image to Gemini API...")
response = model.generate_content([VISION_PROMPT, img])
print("---")
print(response.text)
print("---")
