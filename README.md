<div align="center">
  <h1>🛡️ PhishGuard</h1>
  <h3>AI-Powered Multimodal Phishing Detection Framework</h3>
</div>

<br />

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi">
  <img src="https://img.shields.io/badge/Flask-Frontend-000000?style=flat-square&logo=flask">
</div>

## 📖 What is PhishGuard?
Traditional phishing detection mechanisms rely on a single vector (like checking a URL against a blacklist). **PhishGuard** takes a multimodal approach, employing Artificial Intelligence to analyze phishing threats across multiple dimensions simultaneously: URL characteristics, email text semantics, website visual structures, and embedded QR codes.

By aggregating intelligence from multiple domains, PhishGuard significantly reduces false positives and provides a robust defense against modern, sophisticated phishing attacks.

---

## ⚡ Core Capabilities
- **URL & Domain Intelligence:** Evaluates URL structure and WHOIS data to determine the trustworthiness of a link.
- **Semantic Email Analysis:** Understands the context and urgency of email contents to catch social engineering attempts.
- **Visual Webpage Inspection:** Uses Vision-Language Models to detect fake login pages and brand impersonation.
- **QR Code "Quishing" Detection:** Extracts and scans hidden URLs embedded within QR codes.
- **Image-to-Text (OCR) Processing:** Extracts text directly from screenshots/emails to analyze hidden payloads.
- **Unified Risk Scoring:** Aggregates individual scores into a single, reliable verdict.

---

## 🧠 Under The Hood: AI Models & Performance

PhishGuard orchestrates multiple machine learning models. *Note: The exact performance metrics below are left blank to be filled with the latest benchmark results.*

### 1. URL Analysis (XGBoost)
Detects malicious patterns based on 10+ extracted features (URL length, entropy, special characters, etc.).
- **Accuracy:** [ To be filled ]
- **Precision / Recall:** [ To be filled ]
- **F1-Score:** [ To be filled ]

### 2. Email NLP Classification (RoBERTa)
Analyzes the text of an email to detect phishing semantics.
- **Accuracy:** [ To be filled ]
- **Precision / Recall:** [ To be filled ]
- **ROC-AUC:** [ To be filled ]

### 3. Visual Analysis (LLaVA / Qwen-VL)
Performs visual reasoning on webpage or email screenshots to identify credential harvesting and fake warnings.
- **Screenshot Coverage:** [ To be filled ]
- **Average Inference Time:** [ To be filled ]

### 4. OCR Processing (PaddleOCR)
Supports multiple languages (including Vietnamese) to extract text directly from images before passing it to the NLP classification layer.

---

## 🏗️ Architecture & Data Flow

PhishGuard separates concerns cleanly between a web frontend and a heavily AI-driven API backend.

1. **User Input:** User submits a URL, text, image, or QR code via the **Flask** web application.
2. **Backend Processing:** The request is routed to the **FastAPI** backend.
3. **Multi-Model Inference:** 
   - URLs are parsed and sent to the XGBoost module and WHOIS service.
   - Text is tokenized and sent to the RoBERTa module.
   - Images are processed via PaddleOCR or the Vision Model.
4. **Aggregation Layer:** Individual risk scores are weighted (e.g., URL: 35%, Email: 40%, Visual: 25%) and combined.
5. **Verdict:** A final JSON payload with the unified verdict and reasoning is returned to the user interface.

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Python 3.10 environment (Conda is recommended):
```bash
git clone https://github.com/vutuanhungkkk/VTHPhishDet.git
cd VTHPhishDet

conda create -n llm python=3.10
conda activate llm
```

### 2. Install Dependencies
You need to install packages for both the backend and frontend:
```bash
# Cài đặt requirements cho project
pip install -r requirements.txt
```
*(Nếu bạn đã lỡ xóa file requirements.txt, hãy báo mình để mình tạo lại nhé!)*

### 3. Running the Application
The application runs as two separate services. You will need two terminal windows:

**Terminal 1 (Start Backend API):**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 (Start Frontend Web UI):**
```bash
cd frontend
python app.py
```
*Access the UI at: `http://localhost:5000`*

---

## 💾 Model Weights & Datasets

### External Model Weights
Due to size constraints, the heavily trained `.pkl` and `.bin`/`.safetensors` files are hosted externally.

Place the downloaded files in the appropriate directories:
```
backend/models/RoBERTa_model/
backend/models/xgboost_phishing_model.pkl
```

### Datasets
- **URL Training Data:** Contains ~450k URLs (77% Safe, 23% Phishing).
- **Email Training Data:** Contains ~18k balanced samples.

---

## 👥 Authors & Credits
- **Vũ Tuấn Hùng** (vutuanhungkkk)
- GitHub: [https://github.com/vutuanhungkkk](https://github.com/vutuanhungkkk)

### License
This project is licensed under the MIT License.
