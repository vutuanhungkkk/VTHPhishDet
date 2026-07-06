<div align="center">
  <h1>🛡️ VTHPhishDet (PhishGuard)</h1>
  <h3>AI-Powered Multimodal Phishing Detection Framework</h3>
</div>
<div align="center">
  <b>DEMO: 🔗 <a href="https://drive.google.com/file/d/1J-_69e6KT_klt2xPMkVhdI8xCQhdF9ZM/view?usp=drive_link">Demo Video</a></b><br><br>
  <img src="gif/test.gif" alt="PhishGuard Demo" width="800" />
</div>

<br />

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi">
  <img src="https://img.shields.io/badge/Flask-Frontend-000000?style=flat-square&logo=flask">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow?style=flat-square">
  <img src="https://img.shields.io/badge/Gemini_1.5_Flash-Vision-orange?style=flat-square">
</div>

---

## 📑 Table of Contents
- [📖 Introduction](#-introduction)
- [⚙️ Core Pipeline & Technologies](#-core-pipeline--technologies)
- [📊 Training & Evaluation Results](#-training--evaluation-results)
- [🏗️ Architecture & Data Flow](#-architecture--data-flow)
- [🚀 Setup & Installation](#-setup--installation)
- [💡 Usage](#-usage)
- [👥 Authors](#-authors)

---

## 📖 Introduction
Traditional phishing detection mechanisms rely on a single vector (like checking a URL against a blacklist). **VTHPhishDet (PhishGuard)** takes a cutting-edge multimodal approach, employing Artificial Intelligence to analyze phishing threats across multiple dimensions simultaneously: URL characteristics, email text semantics, website visual structures, and embedded QR codes.

By aggregating intelligence from multiple domains, PhishGuard significantly reduces false positives and provides a robust defense against modern, sophisticated phishing attacks.

---

## ⚙️ Core Pipeline & Technologies

This framework utilizes a hybrid pipeline of specialized machine learning models and APIs:

- **URL Intelligence (XGBoost + WHOIS):** Analyzes lexical features of URLs (length, entropy, special chars) and domain age using WHOIS data to catch newly registered phishing domains.
- **Email NLP (RoBERTa & Llama 3.2 SLM):** Processes the semantic meaning, urgency, and context of email bodies. Users can switch between a fast RoBERTa classifier or a highly capable Llama 3.2 Small Language Model.
- **Visual Analysis (Gemini 1.5 Flash API):** Analyzes screenshots of suspicious emails or web pages to detect brand impersonation, deceptive UI, and fake security warnings using Google's generative multimodal AI.
- **OCR Integration (PaddleOCR):** Extracts text embedded inside images/screenshots to feed into the NLP pipeline.
- **QR Code Scanning:** Safely extracts hidden URLs from QR codes before routing them through the URL intelligence module.

**Tech Stack:**
- **Backend:** FastAPI, Uvicorn, Python 3.10
- **Frontend:** Flask, HTML5, CSS3, Vanilla JavaScript
- **AI/ML:** XGBoost, HuggingFace Transformers, Unsloth (for Llama finetuning), Google Generative AI (Gemini), PaddleOCR.

---

## 📊 Training & Evaluation Results

The models in this repository have been fine-tuned on custom datasets. You can find the notebooks, training scripts, and evaluation metrics in the `build_models/` directory.

### 1. URL Analysis (XGBoost)
Trained on a massive dataset of **651,191** URLs (65.74% Safe, 34.26% Phishing), extracting specific lexical features for structural analysis.
- **Accuracy:** `92.09%`
- **F1 Score:** `0.9198`
- **ROC-AUC:** `0.9714`
- **Location:** `build_models/train_xgboost/`

### 2. Email NLP (RoBERTa)
- **Accuracy:** `98.62%`
- **F1 Score:** `0.9862`
- **ROC-AUC:** `0.9999`
- **Location:** `build_models/train_roberta/`

### 3. Email NLP (Llama 3.2 SLM)
Fine-tuned using Unsloth and PEFT (Parameter-Efficient Fine-Tuning) on the same balanced dataset (59,976 train samples, 12,852 test samples) to deeply understand complex social engineering tactics.
- **Accuracy:** `98.33%`
- **Weighted F1-Score:** `0.99`
- **Location:** `build_models/train_SLM/finetune-slm-email-phishing.ipynb`

---

## 🏗️ Architecture & Data Flow

1. **User Input:** User submits a URL, text, image, or QR code via the **Flask** web UI.
2. **Backend Processing:** The request is routed to the **FastAPI** backend endpoint.
3. **Multi-Model Inference:** 
   - URLs are parsed and evaluated by **XGBoost** and WHOIS APIs.
   - Text is evaluated by the loaded **RoBERTa** or **Llama 3.2** weights.
   - Screenshots are analyzed directly by **Gemini 1.5 Flash API**.
   - Images with text go through **PaddleOCR**.
4. **Aggregation Layer:** Individual risk scores are weighted (e.g., URL: 35%, Email: 40%, Visual: 25%) and aggregated.
5. **Verdict:** A JSON payload with the unified risk score and rationale is returned.

---

## 🚀 Setup & Installation

### 1. Prerequisites
Clone the repository and set up a Python 3.10 conda environment:
```bash
git clone https://github.com/vutuanhungkkk/VTHPhishDet.git
cd VTHPhishDet

conda create -n llm python=3.10
conda activate llm
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Environment Variables (Gemini API)
The vision scoring module uses Google's Gemini API. You must configure your API key.
1. In the `backend/` directory, create or edit the `.env` file.
2. Add your key:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 4. Running the Application
The application runs as two separate services. You will need two terminal windows:

**Terminal 1 (Backend API):**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 (Frontend Web UI):**
```bash
cd frontend
python app.py
```
*Access the UI at: `http://localhost:5000`*

---

## 💾 Model Weights & Datasets

### 1. Download Pre-trained Models
Due to size constraints, the trained models are hosted externally:
- **[Models Download Link](https://drive.google.com/drive/folders/1SkoxJcZf04F1eXcGYUU4b_ALkbDbJ6w5?usp=sharing)** (Includes SLM adapters, XGBoost `.pkl`, and RoBERTa weights).
Place the downloaded files into their respective directories (`backend/models/` and `build_models/models/`).

### 2. Training & Evaluation Data
- **URL Training Data (XGBoost):** 651,191 URLs (65.74% Safe, 34.26% Phishing).
  - 🔗 **[Download XGBoost Data](https://drive.google.com/drive/folders/12lSGzYj6NBPkcsuPJNZQkeHmYrMum1iI?usp=sharing)**
- **Email Training Data (RoBERTa & SLM):** 85,680 balanced samples (50% Safe, 50% Phishing), cleaned from 367,912 raw emails.
  - 🔗 **[Download Email Data](https://drive.google.com/drive/folders/14GFn-DF0RuJU0uQMJjX75hbX0C79VPnf?usp=sharing)**

---

## 💡 Usage
- **URL Tab:** Paste any suspicious link. The system will extract features and query the domain age.
- **Email Tab:** Paste email text or upload an image of an email. Select between RoBERTa (fast) or Llama 3.2 (advanced).
- **Screenshot Tab:** Upload a screenshot of a suspicious login page or system warning. The Gemini API will visually scan for brand impersonation.
- **QR Code Tab:** Upload an image of a QR code to extract and scan the underlying link.

---

## 👥 Authors
- **Vũ Tuấn Hùng** (vutuanhungkkk)
- GitHub: [https://github.com/vutuanhungkkk](https://github.com/vutuanhungkkk)
