"""
Xynapse REST API Server
────────────────────────────────────────────────────────────────
A Flask REST API for chest X-ray classification using the Xynapse
multimodal fusion model. Designed to be deployed on Railway and
called from a separate React frontend hosted on Cloudflare Pages.

Usage:
    python app.py
    → listens on 0.0.0.0:${PORT:-5000}
"""

import os
import sys
import uuid
import time
from io import StringIO

# Load .env automatically when running locally (no-op if file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set externally

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import torchxrayvision as xrv
import onnxruntime as ort
import groq

# ── CPU thread tuning for faster ONNX inference ──
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

TARGET_LABELS = ["Cardiomegaly", "Pleural Effusion", "Pneumonia", "Pneumothorax", "Consolidation"]

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
IMG_SIZE = 224
BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

# Thresholds — optimised on the validation set WITH real text embeddings (multimodal)
THRESHOLDS_NEW = {
    "Cardiomegaly":     0.46,
    "Pleural Effusion": 0.83,
    "Pneumonia":        0.84,
    "Pneumothorax":     0.75,
    "Consolidation":    0.75,
}

# Thresholds for IMAGE-ONLY mode (zero text embedding).
# The model's output distribution shifts lower without a real report, so we
# use reduced thresholds (~0.85× of the multimodal values, minimum 0.40).
THRESHOLDS_NEW_IMAGE_ONLY = {
    "Cardiomegaly":     0.40,   # 0.46 → 0.40
    "Pleural Effusion": 0.70,   # 0.83 → 0.70
    "Pneumonia":        0.72,   # 0.84 → 0.72
    "Pneumothorax":     0.64,   # 0.75 → 0.64
    "Consolidation":    0.64,   # 0.75 → 0.64
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════════════════════

class ImageEncoderNew(nn.Module):
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


class FusionModelNew(nn.Module):
    """New model — has aux_head, returns (logits, aux_logits)."""
    def __init__(self, img_dim=256, txt_dim=768, proj_dim=256, num_labels=5):
        super().__init__()
        self.image_enc = ImageEncoderNew(out_dim=proj_dim)
        self.text_proj = TextProjection(in_dim=txt_dim, out_dim=proj_dim)
        self.fusion    = GatedFusion(dim=proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 3, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_labels),
        )
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


# ═══════════════════════════════════════════════════════════════════════════════
# Global ONNX session — loaded ONCE at startup, reused across all requests
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_ort_session: ort.InferenceSession = None  # populated in load_model_at_startup()


def load_model_at_startup():
    """Load xynapse.onnx via ONNX Runtime once at server startup."""
    global _ort_session
    onnx_path = os.path.join(BASE_DIR, "xynapse.onnx")
    print(f"[startup] Loading ONNX model from {onnx_path}...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    _ort_session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    print("[startup] ONNX model loaded successfully.")


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Text embedding — resolve ONCE, shared by both models
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_text_embedding(report_text, device):
    """
    Resolve the text embedding for the report, checking the cache first.
    Returns (txt_emb, modality, embed_log) where txt_emb is a (1, 768) tensor.
    """
    log_lines = []

    if not report_text.strip():
        log_lines.append("No report text — using image-only mode (zero embedding).")
        return torch.zeros(1, 768), "image_only", "\n".join(log_lines)

    cache_key = report_text.strip()

    # Check embedding cache — always use the canonical path
    cache_path = os.path.join(BASE_DIR, "text_embeddings.pt")
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        txt_emb = cache.get(cache_key, None)
        if txt_emb is not None:
            log_lines.append("Text embedding found in cache (text_embeddings.pt).")
            return txt_emb, "multimodal", "\n".join(log_lines)

    # Not in cache — embed with ClinicalBERT
    log_lines.append("Report not in cache — embedding with ClinicalBERT...")
    txt_emb = embed_text(report_text, device)
    log_lines.append("Text embedded successfully.")

    # Save to cache (load existing entries first to avoid overwriting them)
    cache = {}
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
    cache[cache_key] = txt_emb
    torch.save(cache, cache_path)  # fixed: was referencing undefined old_cache_path
    log_lines.append("Saved embedding to cache.")

    return txt_emb, "multimodal", "\n".join(log_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference runner — uses the globally loaded model
# ═══════════════════════════════════════════════════════════════════════════════

def run_new_model(image_path, txt_emb, modality, device):
    """Run inference with the globally loaded NEW model. Returns (result_dict, log_str, elapsed)."""
    log = StringIO()
    _print = lambda *a, **kw: print(*a, **kw, file=log)

    t0 = time.time()
    # Select threshold set based on modality
    thresholds = THRESHOLDS_NEW_IMAGE_ONLY if modality == "image_only" else THRESHOLDS_NEW

    _print(f"[NEW] Using ONNX Runtime session (CPUExecutionProvider)...")

    img = load_image(image_path)  # (1, 1, 224, 224) tensor
    img_np = img.numpy().astype(np.float32)
    _print(f"[NEW] Image loaded: {os.path.basename(image_path)}")
    _print(f"[NEW] Modality: {modality}")

    txt_np = txt_emb.numpy().astype(np.float32)  # (1, 768)

    _print(f"[NEW] Running ONNX forward pass...")
    ort_inputs = {
        _ort_session.get_inputs()[0].name: img_np,
        _ort_session.get_inputs()[1].name: txt_np,
    }
    ort_outs = _ort_session.run(None, ort_inputs)
    logits_np = ort_outs[0]  # shape (1, 5) — main logits
    probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
    probs = probs.squeeze(0)  # (5,)

    prob_dict = {label: round(float(probs[i]), 4) for i, label in enumerate(TARGET_LABELS)}
    pred_dict = {label: int(probs[i] >= thresholds[label]) for i, label in enumerate(TARGET_LABELS)}
    detected  = [label for label in TARGET_LABELS if pred_dict[label] == 1]

    elapsed = time.time() - t0

    threshold_note = " (image-only thresholds)" if modality == "image_only" else ""
    _print(f"\n{'═'*52}")
    _print(f"  Xynapse NEW — {os.path.basename(image_path)}")
    _print(f"  Mode: {modality}{threshold_note}")
    _print(f"{'═'*52}")
    for label in TARGET_LABELS:
        p   = prob_dict[label]
        pos = pred_dict[label]
        thr = thresholds[label]
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        status = "✓ POSITIVE" if pos else "  negative"
        _print(f"  {label:<20} {bar}  {p:.2f}  [thr={thr:.2f}]  {status}")
    _print(f"{'─'*52}")
    if detected:
        _print(f"  Detected: {', '.join(detected)}")
    else:
        _print("  No acute findings detected.")
    _print(f"{'═'*52}")
    _print(f"\n[NEW] Inference completed in {elapsed:.2f}s")

    return {
        "probs": prob_dict,
        "predictions": pred_dict,
        "detected": detected,
        "modality": modality,
        "thresholds": thresholds,
    }, log.getvalue(), elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# Flask App
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Load model at module level so both waitress and direct `python app.py`
# runs have the model ready before the first request is served.
load_model_at_startup()


@app.route("/health")
def health():
    """Health check endpoint for Railway."""
    return jsonify({"status": "ok"})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serve uploaded X-ray images (still useful for debugging / preview links)."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    # Get image
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # Save uploaded file
    ext = os.path.splitext(file.filename)[1] or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    report_text = request.form.get("report", "").strip()

    # ── Step 1: Resolve text embedding ONCE ──
    try:
        txt_emb, modality, embed_log = resolve_text_embedding(report_text, DEVICE)
    except Exception as e:
        embed_log = f"ERROR embedding text: {str(e)}"
        txt_emb = torch.zeros(1, 768)
        modality = "image_only"

    # ── Step 2: Run NEW model ──
    try:
        new_result, new_log, new_time = run_new_model(filepath, txt_emb, modality, DEVICE)
        new_log = f"[EMBED] {embed_log}\n\n{new_log}"
    except Exception as e:
        new_result = None
        new_log = f"[EMBED] {embed_log}\n\n[NEW] ERROR: {str(e)}"
        new_time = 0

    return jsonify({
        "image_url": f"/uploads/{filename}",
        "report": report_text,
        "device": DEVICE,
        "new_model": {
            "name": "Xynapse NEW (xynapse_best.pt)",
            "result": new_result,
            "log": new_log,
            "time": round(new_time, 2),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Medical Chatbot — /api/chat
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Medical chatbot endpoint backed by Groq (llama-3.1-8b-instant).

    Request JSON:
        {
          "message": "string",
          "history": [{"role": "user"|"assistant", "content": "string"}, ...],
          "scan": {
            "detected": ["Pneumonia", ...],
            "probs": {"Cardiomegaly": 0.72, ...}
          }
        }

    Response JSON:
        { "reply": "string" }
    """
    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 500

    body = request.get_json(force=True, silent=True) or {}
    user_message = body.get("message", "").strip()
    history      = body.get("history", [])
    scan         = body.get("scan", {})

    if not user_message:
        return jsonify({"error": "message field is required"}), 400

    # ── Build scan context ──
    detected = scan.get("detected", [])
    probs    = scan.get("probs", {})

    detected_str = ", ".join(detected) if detected else "None"

    scores_lines = []
    for label in TARGET_LABELS:
        pct = round(probs.get(label, 0.0) * 100, 1)
        scores_lines.append(f"  - {label}: {pct}%")
    scores_str = "\n".join(scores_lines)

    # ── Location-first rule (only injected on the very first turn) ──
    location_rule = ""
    if len(history) == 0:
        location_rule = (
            "IMPORTANT: Before answering ANY question, you MUST first ask the user "
            "for their city and country. Do not provide any medical information until "
            "they have supplied their location. Once they do, recommend 3-4 real hospitals "
            "or health institutes in that city that specifically treat the detected "
            "condition(s), using your training knowledge only — no external APIs."
        )

    # ── System prompt ──
    system_prompt = f"""You are Xynapse's medical assistant — a concise, plain-language AI \
specialised in chest X-ray findings.

SCAN RESULTS
============
Detected conditions : {detected_str}
Confidence scores   :
{scores_str}

BEHAVIOUR RULES
===============
1. TOPIC RESTRICTION — Only discuss topics directly related to chest X-rays, chest scans, 
   or the conditions detected above. If the user asks about anything unrelated, politely 
   decline and redirect them back to their scan results.
2. CONCISENESS — Keep every reply to 3-5 sentences maximum. Use plain language; avoid 
   medical jargon. If a term is necessary, briefly explain it.
3. DISCLAIMER — Every single response MUST end with this exact one-line reminder on its own 
   line: "⚠️ This is not a substitute for professional diagnosis — please consult a licensed physician."
{location_rule}"""

    # ── Assemble message list ──
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    # ── Call Groq ──
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.5,
            max_tokens=512,
        )
        reply = completion.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Xynapse API on port {port}")
    serve(app, host="0.0.0.0", port=port, threads=4)
