import os
from dotenv import load_dotenv

# Nạp file .env ngay từ đầu
load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_phishing_model.pkl")
ROBERTA_MODEL_PATH = os.path.join(MODELS_DIR, "RoBERTa_model")
LLAMA_MODEL_PATH = os.path.join(BASE_DIR, "..", "build_models", "models", "SLM_adapter")
print(XGBOOST_MODEL_PATH)
print(ROBERTA_MODEL_PATH)
print(LLAMA_MODEL_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_VERSION = os.getenv("GEMINI_MODEL_VERSION", "gemini-1.5-flash")
VISION_PROMPT = os.getenv(
	"VISION_PROMPT",
	"You are an expert cybersecurity analyst specializing in visual phishing detection. "
	"Examine the provided screenshot (webpage, email, or message) very carefully for suspicious indicators: "
	"1) Brand impersonation (fake logos, blurry images, slight misspellings). "
	"2) Deceptive UI elements (fake 'Login' windows, 'Update Required' popups, or fake security warnings). "
	"3) Urgency or threats (e.g., 'Account suspended', 'Action required immediately'). "
	"4) Inconsistencies (mismatch between the stated brand and the overall design quality). "
	"Return ONLY valid JSON with keys: risk_score (float 0.0 to 1.0, where 1.0 is definitely phishing), "
	"label ('phishing', 'suspicious', or 'safe'), and rationale (a brief explanation of the specific suspicious signs found). "
	"Do not output markdown formatting around the JSON, just the raw JSON object."
)


PHISHING_THRESHOLD = 0.5


WEIGHT_URL = 0.35
WEIGHT_EMAIL = 0.40
WEIGHT_IMAGE = 0.25


WHOIS_YOUNG_DOMAIN_DAYS = 180