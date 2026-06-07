"""
xynapse_inference.py
────────────────────
Standalone inference script for the Xynapse multimodal chest X-ray classifier.
Give it a chest X-ray image (and optionally a radiology report) and it returns
probability scores for 5 pathologies.

Usage
-----
  # Image only
  python xynapse_inference.py --image chest_xray.png

  # Image + report text
  python xynapse_inference.py --image chest_xray.png --report "Cardiac silhouette is enlarged."

  # Batch — CSV with columns: image_path, report (report column optional)
  python xynapse_inference.py --csv samples.csv --out predictions.csv

Requirements
------------
  pip install torch torchvision torchxrayvision transformers pillow pandas tqdm

Model files needed (same folder as this script, or pass --weights / --embeddings):
  xynapse_best.pt        — trained model weights
  text_embeddings.pt     — ClinicalBERT embedding cache (optional; will re-embed if missing)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

# ── Labels & thresholds (from per-class threshold optimisation in notebook) ──
TARGET_LABELS = ["Cardiomegaly", "Pleural Effusion", "Pneumonia", "Pneumothorax", "Consolidation"]
DEFAULT_THRESHOLDS = {
    "Cardiomegaly":    0.46,
    "Pleural Effusion": 0.83,
    "Pneumonia":       0.84,
    "Pneumothorax":    0.75,
    "Consolidation":   0.75,
}
IMG_SIZE = 224
BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


# ─────────────────────────────── Model definition ──────────────────────────────
# Must match architecture in the notebook exactly.

import torchxrayvision as xrv

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256, freeze_backbone=True):
        super().__init__()
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(1024, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def unfreeze_top_block(self):
        for name, p in self.backbone.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                p.requires_grad = True

    def forward(self, x):
        feats = self.backbone.features(x)
        feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        return self.proj(feats)


class TextProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, txt_emb):
        return self.proj(txt_emb)


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, img, txt):
        cat   = torch.cat([img, txt], dim=1)
        alpha = self.gate(cat)
        gated = alpha * img + (1 - alpha) * txt
        return torch.cat([gated, img, txt], dim=1)


class FusionModel(nn.Module):
    def __init__(self, img_dim=256, txt_dim=768, proj_dim=256, num_labels=5):
        super().__init__()
        self.image_enc = ImageEncoder(out_dim=proj_dim)
        self.text_proj = TextProjection(in_dim=txt_dim, out_dim=proj_dim)
        self.fusion    = GatedFusion(dim=proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_labels),
        )
        # Auxiliary head used during training only; defined here so the checkpoint loads cleanly
        self.aux_head = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels),
        )

    def forward(self, images, txt_emb):
        img_feat = self.image_enc(images)
        txt_feat = self.text_proj(txt_emb)
        fused    = self.fusion(img_feat, txt_feat)
        logits     = self.mlp(fused)
        aux_logits = self.aux_head(img_feat)
        return logits, aux_logits


# ─────────────────────────────── Preprocessing ─────────────────────────────────

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def load_image(path: str) -> torch.Tensor:
    """Load a chest X-ray PNG/JPEG → (1, 1, 224, 224) in torchxrayvision range."""
    img = Image.open(path).convert("L")
    t   = val_transform(img)         # (1, 224, 224) in [0, 1]
    t   = t * 2048 - 1024            # torchxrayvision normalisation
    return t.unsqueeze(0)            # (1, 1, 224, 224)


def embed_text(text: str, device: str = "cpu") -> torch.Tensor:
    """Embed a single report string with ClinicalBERT → (1, 768)."""
    from transformers import AutoTokenizer, AutoModel
    print("  Loading ClinicalBERT (first call only)...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    bert      = AutoModel.from_pretrained(BERT_MODEL).to(device).eval()
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=128, padding=True).to(device)
        out = bert(**enc)
        emb = out.last_hidden_state[:, 0, :].cpu()   # (1, 768)
    del bert
    torch.cuda.empty_cache()
    return emb


# ─────────────────────────────── Inference ─────────────────────────────────────

def predict(
    image_path: str,
    report_text: str = "",
    weights_path: str = "xynapse_best.pt",
    embed_cache_path: str = "text_embeddings.pt",
    thresholds: dict = None,
    device: str = None,
) -> dict:
    """
    Run inference on a single X-ray + optional report.

    Returns a dict with keys:
      probs        — {label: float probability}
      predictions  — {label: 0 or 1}
      detected     — list of positive label names
      modality     — "multimodal" or "image_only"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Load model
    print(f"Loading model from {weights_path} on {device}...")
    model = FusionModel().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Prepare image
    img = load_image(image_path).to(device)

    # Prepare text embedding
    if report_text.strip():
        modality = "multimodal"
        cache_key = report_text.strip()
        cache = {}
        if os.path.exists(embed_cache_path):
            cache = torch.load(embed_cache_path, weights_only=False)
        txt_emb = cache.get(cache_key, None)
        if txt_emb is None:
            print("  Report not in cache — embedding with ClinicalBERT...")
            txt_emb = embed_text(report_text, device)
            cache[cache_key] = txt_emb
            torch.save(cache, embed_cache_path)
    else:
        modality = "image_only"
        txt_emb  = torch.zeros(1, 768)   # zero text → model leans on image only

    txt_emb = txt_emb.to(device)

    # Forward pass
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits, _ = model(img, txt_emb)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # Build output
    prob_dict = {label: round(float(probs[i]), 4) for i, label in enumerate(TARGET_LABELS)}
    pred_dict = {label: int(probs[i] >= thresholds[label]) for i, label in enumerate(TARGET_LABELS)}
    detected  = [label for label in TARGET_LABELS if pred_dict[label] == 1]

    return {
        "probs":       prob_dict,
        "predictions": pred_dict,
        "detected":    detected,
        "modality":    modality,
    }


def print_result(result: dict, image_path: str):
    print("\n" + "═" * 52)
    print(f"  Xynapse prediction — {os.path.basename(image_path)}")
    print(f"  Mode: {result['modality']}")
    print("═" * 52)
    for label in TARGET_LABELS:
        p   = result['probs'][label]
        pos = result['predictions'][label]
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        status = "✓ POSITIVE" if pos else "  negative"
        print(f"  {label:<20} {bar}  {p:.2f}  {status}")
    print("─" * 52)
    if result["detected"]:
        print(f"  Detected: {', '.join(result['detected'])}")
    else:
        print("  No acute findings detected.")
    print("═" * 52 + "\n")


# ─────────────────────────────── Batch mode ────────────────────────────────────

def batch_predict(
    csv_path: str,
    out_path: str,
    weights_path: str = "xynapse_best.pt",
    embed_cache_path: str = "text_embeddings.pt",
    thresholds: dict = None,
    device: str = None,
):
    """Run inference over a CSV with columns: image_path, report (optional)."""
    from tqdm import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise ValueError("CSV must have an 'image_path' column.")
    if "report" not in df.columns:
        df["report"] = ""

    # Load model once
    print(f"Loading model on {device}...")
    model = FusionModel().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Load or init embedding cache
    cache = {}
    if os.path.exists(embed_cache_path):
        cache = torch.load(embed_cache_path, weights_only=False)
        cache = {str(k): v for k, v in cache.items()}

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        img_path = str(row["image_path"])
        report   = str(row.get("report", "")).strip()

        if not os.path.exists(img_path):
            print(f"  Skipping missing image: {img_path}")
            continue

        img = load_image(img_path).to(device)

        if report:
            txt_emb = cache.get(report, None)
            if txt_emb is None:
                txt_emb = embed_text(report, device)
                cache[report] = txt_emb
        else:
            txt_emb = torch.zeros(1, 768)

        txt_emb = txt_emb.to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits, _ = model(img, txt_emb)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        out_row = {"image_path": img_path, "report": report}
        for i, label in enumerate(TARGET_LABELS):
            out_row[f"prob_{label.lower().replace(' ', '_')}"] = round(float(probs[i]), 4)
            out_row[f"pred_{label.lower().replace(' ', '_')}"] = int(probs[i] >= thresholds[label])

        detected = [label for i, label in enumerate(TARGET_LABELS) if probs[i] >= thresholds[label]]
        out_row["detected"] = "|".join(detected)
        rows.append(out_row)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(pred_df)} predictions → {out_path}")
    return pred_df


# ─────────────────────────────── CLI ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xynapse chest X-ray inference")
    parser.add_argument("--image",      type=str, help="Path to a single chest X-ray image")
    parser.add_argument("--report",     type=str, default="", help="Optional radiology report text")
    parser.add_argument("--csv",        type=str, help="CSV with image_path (+ optional report) for batch mode")
    parser.add_argument("--out",        type=str, default="xynapse_predictions.csv", help="Output CSV path (batch mode)")
    parser.add_argument("--weights",    type=str, default="xynapse_best.pt", help="Path to xynapse_best.pt")
    parser.add_argument("--embeddings", type=str, default="text_embeddings.pt", help="Path to text_embeddings.pt cache")
    parser.add_argument("--device",     type=str, default=None, help="cuda or cpu (auto-detected if omitted)")
    args = parser.parse_args()

    if args.csv:
        batch_predict(
            csv_path=args.csv,
            out_path=args.out,
            weights_path=args.weights,
            embed_cache_path=args.embeddings,
            device=args.device,
        )
    elif args.image:
        result = predict(
            image_path=args.image,
            report_text=args.report,
            weights_path=args.weights,
            embed_cache_path=args.embeddings,
            device=args.device,
        )
        print_result(result, args.image)
    else:
        parser.print_help()