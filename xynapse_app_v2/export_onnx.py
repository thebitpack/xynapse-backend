"""
export_onnx.py
──────────────────────────────────────────────────────────────
Run this script ONCE locally to convert xynapse_best.pt to
xynapse.onnx for fast CPU inference via ONNX Runtime.

Usage:
    python export_onnx.py

Produces:
    xynapse.onnx  (same directory as this script)

Dependencies needed (no Flask required):
    pip install torch torchvision torchxrayvision onnxruntime
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions (copied from app.py — kept in sync manually)
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
# Export
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "xynapse_best.pt")
ONNX_PATH    = os.path.join(BASE_DIR, "xynapse.onnx")


def main():
    print(f"[export] Loading weights from: {WEIGHTS_PATH}")
    model = FusionModelNew()
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[export] Weights loaded.")

    # Dummy inputs matching the shapes used at inference time
    dummy_image = torch.zeros(1, 1, 224, 224, dtype=torch.float32)
    dummy_txt   = torch.zeros(1, 768,          dtype=torch.float32)

    print(f"[export] Exporting to ONNX (opset 14) -> {ONNX_PATH}")
    # dynamo=False forces the legacy TorchScript exporter.
    # PyTorch 2.12+ defaults to the dynamo exporter which requires
    # the optional 'onnxscript' package — avoid that dependency.
    torch.onnx.export(
        model,
        args=(dummy_image, dummy_txt),
        f=ONNX_PATH,
        input_names=["images", "txt_emb"],
        output_names=["logits", "aux_logits"],
        dynamic_axes={
            "images":     {0: "batch_size"},
            "txt_emb":    {0: "batch_size"},
            "logits":     {0: "batch_size"},
            "aux_logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] Done. ONNX model saved to: {ONNX_PATH}")

    # Quick sanity-check with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        img_np = dummy_image.numpy().astype(np.float32)
        txt_np = dummy_txt.numpy().astype(np.float32)
        ort_inputs = {
            sess.get_inputs()[0].name: img_np,
            sess.get_inputs()[1].name: txt_np,
        }
        outs = sess.run(None, ort_inputs)
        print(f"[export] Sanity-check passed. Output shapes: {[o.shape for o in outs]}")
    except ImportError:
        print("[export] onnxruntime not installed — skipping sanity-check.")


if __name__ == "__main__":
    main()
