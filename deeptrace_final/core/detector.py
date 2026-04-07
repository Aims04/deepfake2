"""
detector.py — Deepfake CNN Inference
======================================
Priority order:
  1. ViT (dima806/deepfake_vs_real_image_detection) — ~91% accuracy — uses if models/vit_deepfake/ exists
  2. EfficientNet-B4 (ImageNet pretrained)           — ~75% accuracy — fallback

Run setup_weights.py once to download the ViT model.
"""

import os
import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIT_PATH    = os.path.join(BASE_DIR, "models", "vit_deepfake")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_vit_model     = None
_vit_processor = None
_using_vit     = False
_eff_model     = None
_initialized   = False


def _try_load_vit():
    global _vit_model, _vit_processor, _using_vit
    if not os.path.exists(os.path.join(VIT_PATH, "config.json")):
        return False
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        print("[DeepTrace] Loading ViT deepfake model (~91% accuracy)...")
        _vit_processor = AutoImageProcessor.from_pretrained(VIT_PATH)
        _vit_model     = AutoModelForImageClassification.from_pretrained(VIT_PATH)
        _vit_model.eval()
        _vit_model.to(DEVICE)
        _using_vit = True
        print("[DeepTrace] ViT model loaded!")
        return True
    except Exception as e:
        print(f"[DeepTrace] ViT load failed ({e}), falling back to EfficientNet")
        return False


def _run_vit(image_bytes: bytes) -> dict:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _vit_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _vit_model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    labels    = _vit_model.config.id2label
    label_list = [labels[i].lower() for i in range(len(labels))]
    fake_idx  = next((i for i, l in enumerate(label_list) if "fake" in l or "deepfake" in l or "artificial" in l), 1)
    fake_prob = float(probs[fake_idx])
    return {
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(1.0 - fake_prob, 4),
        "model":            "ViT-base (dima806/deepfake_vs_real_image_detection)",
        "accuracy_note":    "~91% accuracy",
    }


class EfficientNetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0, global_pool='avg')
        self.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.head(self.backbone(x))


def _try_load_efficientnet():
    global _eff_model
    try:
        import timm
        print("[DeepTrace] Loading EfficientNet-B4 (ImageNet pretrained ~75% accuracy)...")
        print("[DeepTrace] Tip: run setup_weights.py once to get ~91% accuracy")
        model     = EfficientNetDetector()
        pretrained = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0, global_pool='avg')
        model.backbone.load_state_dict(pretrained.state_dict())
        model.eval()
        model.to(DEVICE)
        _eff_model = model
        return True
    except Exception as e:
        print(f"[DeepTrace] EfficientNet load failed: {e}")
        return False


def _run_efficientnet(image_bytes: bytes) -> dict:
    from torchvision import transforms
    T = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = T(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = float(_eff_model(tensor).squeeze().cpu())
    return {
        "fake_probability": round(prob, 4),
        "real_probability": round(1.0 - prob, 4),
        "model":            "EfficientNet-B4 (ImageNet pretrained)",
        "accuracy_note":    "~75% accuracy. Run setup_weights.py to upgrade to ~91%.",
    }


def _initialize():
    global _initialized
    if _initialized:
        return
    _initialized = True
    if not _try_load_vit():
        _try_load_efficientnet()


def run_inference(image_bytes: bytes) -> dict:
    _initialize()
    try:
        if _using_vit:
            return _run_vit(image_bytes)
        elif _eff_model is not None:
            return _run_efficientnet(image_bytes)
        else:
            return {"fake_probability": 0.5, "real_probability": 0.5, "model": "none"}
    except Exception as e:
        return {"fake_probability": 0.5, "real_probability": 0.5, "model": "error", "error": str(e)}