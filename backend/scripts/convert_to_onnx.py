"""
PaddleOCR -> ONNX Conversion Script
====================================
Downloads PaddleOCR inference models (PP-OCRv3 Det + PP-OCRv4 Rec)
and converts them to ONNX format using paddle2onnx CLI.

Usage:
    python convert_to_onnx.py
"""

import os
import sys
import subprocess
import urllib.request
import tarfile

# ── Model URLs (Official PaddleOCR Inference Models) ──────────────────────────
DET_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
REC_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"

MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "ocr")
)


def download_and_extract(url, extract_to):
    """Download a tar file and extract it."""
    os.makedirs(extract_to, exist_ok=True)
    filename = url.split("/")[-1]
    tar_path = os.path.join(extract_to, filename)

    if os.path.exists(tar_path):
        print(f"  [SKIP] {filename} already downloaded.")
    else:
        print(f"  [DOWN] Downloading {filename}...")
        urllib.request.urlretrieve(url, tar_path)

    print(f"  [UNPACK] Extracting {filename}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_to)

    extracted_dir = os.path.join(extract_to, filename.replace(".tar", ""))
    return extracted_dir


def convert_to_onnx(model_dir, save_file):
    """Convert a PaddlePaddle inference model to ONNX using paddle2onnx CLI."""
    model_file = os.path.join(model_dir, "inference.pdmodel")
    params_file = os.path.join(model_dir, "inference.pdiparams")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Params file not found: {params_file}")

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    print(f"  [CONVERT] {os.path.basename(model_dir)} -> {save_file}")

    # Resolve paddle2onnx executable from the current Python environment
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    paddle2onnx_exe = os.path.join(scripts_dir, "paddle2onnx.exe")
    if not os.path.exists(paddle2onnx_exe):
        # Fallback: try PATH
        paddle2onnx_exe = "paddle2onnx"

    # Use paddle2onnx CLI (more reliable than the Python API)
    cmd = [
        paddle2onnx_exe,
        "--model_dir", model_dir,
        "--model_filename", "inference.pdmodel",
        "--params_filename", "inference.pdiparams",
        "--save_file", save_file,
        "--opset_version", "11",
        "--enable_onnx_checker", "True",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"paddle2onnx failed (exit code {result.returncode}) "
            f"for model: {model_dir}"
        )

    if os.path.exists(save_file):
        size_mb = os.path.getsize(save_file) / (1024 * 1024)
        print(f"  [OK] Saved {save_file} ({size_mb:.1f} MB)")
    else:
        raise FileNotFoundError(f"Expected output not found: {save_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("PaddleOCR to ONNX Conversion Tool")
    print("=" * 60)

    # Step 1: Download and extract PaddleOCR models
    print("\n[1/2] Downloading PaddleOCR inference models...")
    det_dir = download_and_extract(DET_URL, MODELS_DIR)
    rec_dir = download_and_extract(REC_URL, MODELS_DIR)

    # Step 2: Convert to ONNX using CLI
    print("\n[2/2] Converting to ONNX format...")
    det_onnx = os.path.join(MODELS_DIR, "en_det.onnx")
    rec_onnx = os.path.join(MODELS_DIR, "en_rec.onnx")

    try:
        convert_to_onnx(det_dir, det_onnx)
        convert_to_onnx(rec_dir, rec_onnx)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n  [ERROR] {e}")
        print("  Please install: pip install paddlepaddle paddle2onnx")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"All ONNX models saved to: {MODELS_DIR}")
    print(f"  - en_det.onnx  (Text Detection)")
    print(f"  - en_rec.onnx  (Text Recognition)")
    print("=" * 60)
