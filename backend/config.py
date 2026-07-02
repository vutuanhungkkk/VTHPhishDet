import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_phishing_model.pkl")
ROBERTA_MODEL_PATH = os.path.join(MODELS_DIR, "RoBERTa_model")
print(XGBOOST_MODEL_PATH)
print(ROBERTA_MODEL_PATH)

VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
VISION_MAX_NEW_TOKENS = int(os.getenv("VISION_MAX_NEW_TOKENS", "128"))
VISION_USE_4BIT = os.getenv("VISION_USE_4BIT", "1") not in {"0", "false", "False"}
VISION_PROMPT = os.getenv(
	"VISION_PROMPT",
	"You are a cybersecurity analyst. Inspect the provided image (which may be a webpage, email, or message) and determine if it is phishing or safe. "
	"Return ONLY valid JSON with keys risk_score, label, and rationale. "
	"risk_score must be a float from 0.0 to 1.0 (1.0 means high risk of phishing, 0.0 means completely safe). "
	"Use label 'safe' for normal, legitimate communications (like job rejections, news, standard notifications). "
	"Use label 'phishing' only if the image shows malicious intent, demands urgent action, asks for passwords, or impersonates a brand."
)


PHISHING_THRESHOLD = 0.5


WEIGHT_URL = 0.35
WEIGHT_EMAIL = 0.40
WEIGHT_IMAGE = 0.25


WHOIS_YOUNG_DOMAIN_DAYS = 180