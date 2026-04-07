"""
DeepTrace Final — No API Key Required
======================================
Full deepfake detection pipeline running 100% locally:

  Layer 1: FFT frequency analysis     (numpy/scipy)
  Layer 2: ELA compression analysis   (Pillow)
  Layer 3: Noise residual analysis    (scipy)
  Layer 4: Facial geometry            (OpenCV)
  Layer 5: EfficientNet-B4 CNN        (PyTorch + timm)
  Layer 6: Signal aggregation         (weighted ensemble)
  Layer 7: Auto-generated text report (from real signal values)

Zero external API calls. Runs entirely on your machine.
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from core.frequency_analysis import run_full_analysis
from core.detector import run_inference
from core.aggregator import aggregate_signals, build_report_text

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="DeepTrace — Local Deepfake Detection", version="3.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
MAX_SIZE      = 20 * 1024 * 1024   # 20MB


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    t0 = time.time()

    # Read & validate
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(413, "File too large — max 20MB")

    ctype = file.content_type or ""
    if ctype not in ALLOWED_TYPES:
        ext = Path(file.filename or "").suffix.lower()
        ext_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png", ".webp": "image/webp"}
        ctype = ext_map.get(ext, "")
        if not ctype:
            raise HTTPException(400, f"Unsupported file type. Use JPG, PNG, or WEBP.")

    # ── Layer 1-4: Signal extraction ────────────────────────────────────────
    try:
        analysis = run_full_analysis(contents)
    except ValueError as e:
        raise HTTPException(422, f"Could not read image: {e}")

    # ── Layer 5: EfficientNet CNN ────────────────────────────────────────────
    try:
        efficientnet = run_inference(contents)
    except Exception as e:
        efficientnet = {"fake_probability": 0.5, "real_probability": 0.5,
                        "feature_stats": {}, "error": str(e), "model": "EfficientNet-B4"}

    # ── Layer 6: Aggregate ───────────────────────────────────────────────────
    analysis["efficientnet"] = efficientnet
    report = aggregate_signals(analysis, efficientnet)

    # ── Layer 7: Generate text from real signal values (no API) ─────────────
    text = build_report_text(report, analysis)

    elapsed = round(time.time() - t0, 2)

    return JSONResponse({
        **report.to_dict(),
        **text,
        "analysis_time_s": elapsed,
        "pipeline_scores": report.signal_scores,
    })


@app.post("/api/analyze-sample")
async def analyze_sample(request: Request):
    body     = await request.json()
    scenario = body.get("scenario", "portrait")
    samples  = {
        "portrait":  _sample_authentic(),
        "stylegan":  _sample_stylegan(),
        "faceswap":  _sample_faceswap(),
        "diffusion": _sample_diffusion(),
    }
    if scenario not in samples:
        raise HTTPException(400, "Unknown scenario")
    return JSONResponse(samples[scenario])


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "api_key_required": False,
        "pipeline": ["FFT", "ELA", "NoiseResidual", "FacialGeometry", "EfficientNet-B4"],
        "version": "3.0.0"
    }


# ── Sample scenarios ───────────────────────────────────────────────────────────
def _sample_authentic():
    return {
        "verdict": "AUTHENTIC", "confidence": 92,
        "composite_score": 0.181, "gan_probability": 7,
        "diffusion_probability": 6, "face_manipulation_score": 9,
        "frequency_anomaly_score": 8, "cnn_score": 14,
        "metadata_integrity": "INTACT", "model_type_detected": "Authentic Camera Photo",
        "faces_detected": 1,
        "summary": "Composite score 0.181 is well below the 0.40 threshold. EfficientNet-B4 assigns 14% fake probability. Spectral power law α=2.18 is consistent with real camera pink-noise distribution. ELA residuals are uniform — no manipulation boundaries detected.",
        "gan_analysis": "Spectral power law α=2.18 is consistent with authentic camera imagery. Periodic peak score 0.04 is far below the 0.3 threshold — no GAN transposed-convolution artifacts in the frequency domain.",
        "recommendation": "Image appears authentic — no significant manipulation detected across all analysis layers.",
        "flags": [{"severity": "LOW", "text": "No significant artifacts detected across all analysis layers"}],
        "pipeline_scores": {"efficientnet": 0.14, "fft": 0.07, "ela": 0.09, "noise": 0.08, "geo": 0.06, "composite": 0.181},
    }


def _sample_stylegan():
    return {
        "verdict": "DEEPFAKE", "confidence": 97,
        "composite_score": 0.847,
        "gan_probability": 93, "diffusion_probability": 12,
        "face_manipulation_score": 58, "frequency_anomaly_score": 91, "cnn_score": 94,
        "metadata_integrity": "COMPROMISED", "model_type_detected": "StyleGAN2 / StyleGAN3",
        "faces_detected": 1,
        "summary": "Composite score 0.847 — high-confidence GAN-generated image. EfficientNet-B4 assigns 94% fake probability. Spectral power law α=1.18 significantly deviates from real camera distribution. Periodic peak score 0.81 confirms GAN upsampling artifacts at 2^7 and 2^8 frequency bins. No EXIF metadata present.",
        "gan_analysis": "Spectral power law α=1.18 (expected ≥1.8) and periodic peak score 0.81 (threshold: 0.30) are definitive GAN fingerprints. Spectral peaks at 2^7 (128px) and 2^8 (256px) are characteristic of StyleGAN3's alias-free upsampling layers.",
        "recommendation": "High-confidence GAN-generated face — flag as synthetic media, do not use as evidence of a real person.",
        "flags": [
            {"severity": "HIGH", "text": "EfficientNet-B4 CNN: 94% fake probability — deep feature artifacts detected"},
            {"severity": "HIGH", "text": "Spectral slope α=1.18 (real images: ≥1.8) — GAN upsampling signature"},
            {"severity": "HIGH", "text": "Periodic spectral peaks at 2^n frequencies (score=0.81) — transposed convolution artifact"},
            {"severity": "MED",  "text": "Over-smooth skin texture (variance=0.07) — GAN-generated face characteristic"},
            {"severity": "MED",  "text": "Non-Gaussian noise (similarity=0.28) — inconsistent with real camera PRNU"},
        ],
        "pipeline_scores": {"efficientnet": 0.94, "fft": 0.91, "ela": 0.44, "noise": 0.69, "geo": 0.58, "composite": 0.847},
    }


def _sample_faceswap():
    return {
        "verdict": "DEEPFAKE", "confidence": 94,
        "composite_score": 0.791,
        "gan_probability": 61, "diffusion_probability": 28,
        "face_manipulation_score": 89, "frequency_anomaly_score": 67, "cnn_score": 88,
        "metadata_integrity": "COMPROMISED", "model_type_detected": "DeepFaceLab / FaceSwap",
        "faces_detected": 1,
        "summary": "Composite score 0.791 — high-confidence face manipulation. EfficientNet-B4 assigns 88% fake probability. ELA spatial inconsistency 1.41 reveals compression boundary between replaced face and original background. Face-background ΔE=41.3 LAB units — far above the 15-unit threshold.",
        "gan_analysis": "Moderate GAN frequency signal (α=1.64, peak=0.44) — the face region is generated by DeepFaceLab's SAEHD encoder. ELA is more diagnostic here: spatial inconsistency 1.41 confirms a pasted region with different compression history than the background.",
        "recommendation": "Face-swap detected — the face region has been replaced. Suitable for forensic flagging.",
        "flags": [
            {"severity": "HIGH", "text": "EfficientNet-B4 CNN: 88% fake probability — deep feature artifacts detected"},
            {"severity": "HIGH", "text": "Spatial ELA inconsistency=1.41 — distinct compression boundaries suggest face pasting"},
            {"severity": "HIGH", "text": "Face-background color mismatch ΔE=41.3 LAB units — face paste artifact"},
            {"severity": "HIGH", "text": "Sharp face boundary (score=0.74) — blending seam at jaw edge"},
            {"severity": "MED",  "text": "Periodic spectral peaks (score=0.44) — transposed convolution artifact"},
        ],
        "pipeline_scores": {"efficientnet": 0.88, "fft": 0.62, "ela": 0.79, "noise": 0.67, "geo": 0.89, "composite": 0.791},
    }


def _sample_diffusion():
    return {
        "verdict": "DEEPFAKE", "confidence": 89,
        "composite_score": 0.683,
        "gan_probability": 37, "diffusion_probability": 82,
        "face_manipulation_score": 54, "frequency_anomaly_score": 61, "cnn_score": 79,
        "metadata_integrity": "SUSPICIOUS", "model_type_detected": "Stable Diffusion / DALL-E",
        "faces_detected": 1,
        "summary": "Composite score 0.683 — diffusion model generated image. EfficientNet-B4 assigns 79% fake probability. The spectral slope α=1.69 is mildly anomalous, but noise kurtosis 5.9 (Gaussian: 3.0) and ELA probability 0.63 are strongly diagnostic of latent diffusion synthesis. Skin texture variance 0.08 confirms over-smoothed synthetic texture.",
        "gan_analysis": "Mild GAN frequency signature (α=1.69, peak=0.26) — diffusion models don't use transposed convolution so the FFT fingerprint is weaker than GANs. Dominant signals are noise non-Gaussianity (kurtosis=5.9) and ELA inconsistency, which better characterize latent diffusion models.",
        "recommendation": "High-confidence diffusion-generated image — lacks real camera sensor fingerprints.",
        "flags": [
            {"severity": "HIGH", "text": "EfficientNet-B4 CNN: 79% fake probability — deep feature artifacts detected"},
            {"severity": "HIGH", "text": "ELA manipulation probability 63% — inconsistent JPEG compression"},
            {"severity": "HIGH", "text": "Noise kurtosis=5.9 (Gaussian: 3.0) — heavy-tailed noise, not a camera sensor"},
            {"severity": "MED",  "text": "Over-smooth skin texture (variance=0.08) — synthetic texture generation"},
            {"severity": "MED",  "text": "Non-Gaussian noise (similarity=0.34) — neural network generation signature"},
        ],
        "pipeline_scores": {"efficientnet": 0.79, "fft": 0.53, "ela": 0.65, "noise": 0.70, "geo": 0.54, "composite": 0.683},
    }
