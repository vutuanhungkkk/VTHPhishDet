import ssl
import certifi

_original_create_default_context = ssl.create_default_context
def _patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    if cafile is None and capath is None and cadata is None:
        cafile = certifi.where()
    return _original_create_default_context(purpose=purpose, cafile=cafile, capath=capath, cadata=cadata)
ssl.create_default_context = _patched_create_default_context

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64

from services.url_scorer import score_url
from services.email_scorer import score_email
from services.llama_scorer import score_email_llama
from services.whois_service import get_domain_info
from services.vision_scorer import score_image
from services.ocr_service import extract_text_from_image
from utils.score_aggregator import aggregate_scores
from utils.qr_extractor import extract_url_from_qr
import requests

def resolve_url(url: str) -> str:
    """Follows redirects to find the final destination URL."""
    try:
        # Some servers block HEAD requests, but it's faster.
        r = requests.head(url, allow_redirects=True, timeout=5)
        if r.status_code == 405:
            r = requests.get(url, allow_redirects=True, timeout=5, stream=True)
            r.close()
        return r.url
    except Exception:
        return url

app = FastAPI(
    title="Multimodal Phishing Detector",
    description="Detects phishing using XGBoost (URL), RoBERTa (Email)",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class URLRequest(BaseModel):
    url: str

class EmailRequest(BaseModel):
    text: str
    model: Optional[str] = "roberta"

class FullScanRequest(BaseModel):
    url: Optional[str] = None
    email_text: Optional[str] = None
    email_model: Optional[str] = "roberta"


@app.get("/")
def root():
    return {"status": "running", "version": "1.1.0"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan/url")
def scan_url(request: URLRequest):
    """Score URL using XGBoost + WHOIS domain age intelligence."""
    url_result = score_url(request.url)
    whois_result = get_domain_info(request.url)

    url_score = url_result.get("url_score")


    if url_score is not None and whois_result.get("whois_score") is not None:
        w_score = whois_result["whois_score"]
        if whois_result.get("is_young_domain"):
            
            boost = (w_score - 0.1) * 0.3   
            url_score = min(1.0, url_score + boost)

    return {
        "url": request.url,
        "url_score": round(url_score, 4) if url_score is not None else None,
        "label": "phishing" if (url_score or 0) >= 0.5 else "safe",
        "features_used": url_result.get("features_used"),
        "whois": whois_result
    }


@app.post("/scan/email")
def scan_email(request: EmailRequest):
    """Score email using RoBERTa or Llama. Returns phishing score + correct confidence."""
    if request.model == "llama":
        result = score_email_llama(request.text)
    else:
        result = score_email(request.text)
        
    return {
        "email_score": result.get("email_score"),
        "safe_score": result.get("safe_score"),
        "label": result.get("label"),
        "confidence": result.get("confidence")
    }


@app.post("/scan/email_image")
async def scan_email_image(file: UploadFile = File(...), model: str = Form("roberta")):
    """Extract text from email image using PaddleOCR and score using RoBERTa or Llama."""
    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode("utf-8")
    
    # 1. Extract text via PaddleOCR
    ocr_result = extract_text_from_image(image_b64)
    if ocr_result.get("error") or not ocr_result.get("text"):
        return {"error": ocr_result.get("error", "No text found in image")}
    
    extracted_text = ocr_result["text"]
    
    # 2. Score extracted text via RoBERTa or Llama
    if model == "llama":
        result = score_email_llama(extracted_text)
    else:
        result = score_email(extracted_text)
    
    return {
        "email_score": result.get("email_score"),
        "safe_score": result.get("safe_score"),
        "label": result.get("label"),
        "confidence": result.get("confidence"),
        "extracted_text": extracted_text
    }


@app.post("/scan/image")
async def scan_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode("utf-8")
    result = score_image(image_b64)
    result["screenshot_base64"] = image_b64
    return result


@app.post("/scan/full")
def full_scan(request: FullScanRequest):
    url_score = None
    email_score = None
    whois_result = None

    if request.url:
        url_result = score_url(request.url)
        url_score = url_result.get("url_score")
        whois_result = get_domain_info(request.url)
        if url_score is not None and whois_result.get("is_young_domain"):
            w_score = whois_result.get("whois_score", 0.5)
            boost = (w_score - 0.1) * 0.3
            url_score = min(1.0, url_score + boost)

    if request.email_text:
        if request.email_model == "llama":
            email_result = score_email_llama(request.email_text)
        else:
            email_result = score_email(request.email_text)
        email_score = email_result.get("email_score")

    verdict = aggregate_scores(url_score=url_score, email_score=email_score)

    return {
        "verdict": verdict,
        "whois": whois_result,
    }


@app.post("/scan/qr")
async def scan_qr(file: UploadFile = File(...)):
    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode("utf-8")
    qr_result = extract_url_from_qr(image_b64)

    if qr_result.get("error") or not qr_result.get("url"):
        return {"qr_url": None, "error": qr_result.get("error", "No QR URL found")}

    extracted_url = qr_result["url"]
    final_url = resolve_url(extracted_url)

    url_result = score_url(final_url)
    whois_result = get_domain_info(final_url)

    url_score = url_result.get("url_score")
    if url_score is not None and whois_result.get("is_young_domain"):
        w_score = whois_result.get("whois_score", 0.5)
        boost = (w_score - 0.1) * 0.3
        url_score = min(1.0, url_score + boost)

    verdict = aggregate_scores(url_score=url_score)

    return {
        "qr_url": final_url,
        "url_score": round(url_score, 4) if url_score is not None else None,
        "label": "phishing" if (url_score or 0) >= 0.5 else "safe",
        "features_used": url_result.get("features_used"),
        "whois": whois_result,
        "verdict": verdict
    }