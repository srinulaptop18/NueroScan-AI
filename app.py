import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER
import torchvision.transforms as transforms
import torchvision.models as tv_models
import base64
import os
import gdown
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import timm

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — updated 2026-03-15 with new .pth files
# ══════════════════════════════════════════════════════════════════════════════

# Individual Drive file IDs for each model (from 20260315_100903 training run)
MODEL_DRIVE_IDS = {
    "resnet50_best.pth":        "1mUjefMpXU9vIFZl7j8aY3PohnavbqSJB",
    "efficientnet-b4_best.pth": "12ie2mSqBTzxOZdwfHvxom9Q2IdoUt0LX",
    "vit-b16_best.pth":         "1D_Kit-vFC8PBPYFk3kG4ZVBcmNzVkJBw",
    "minisegnet_best.pth":      "1YLPUe4d82NMZTtuaFN8Eq9l75IaOP6Wg",
    "hybridnet_rv_best.pth":    "1D_HT-DCMvmqTLVxnjG3y814Q9qH05sZs",
    "hybridnet_ev_best.pth":    "1uXL90WvSm7akZbgpYZCinGAzLUhwhl6i",
}

MODEL_FILES = {
    "HybridNet_RV (ResNet50 + ViT)":     "hybridnet_rv_best.pth",
    "HybridNet_EV (EfficientNet + ViT)": "hybridnet_ev_best.pth",
    "ResNet50":                           "resnet50_best.pth",
    "EfficientNet-B4":                    "efficientnet-b4_best.pth",
    "ViT-B16":                            "vit-b16_best.pth",
    "MiniSegNet":                         "minisegnet_best.pth",
}

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Legacy ResNetViT (matches new_ntau.pth exactly) ──────────────────────────
# Architecture must match EXACTLY what was used to save new_ntau.pth
# Original app.py (PDF page 29) used ResNetBackbone + PatchEmbedding classes

class _ResNetBackbone(nn.Module):
    """Matches original ResNetBackbone class from new_ntau.pth training."""
    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)
        for p in resnet.parameters():
            p.requires_grad = False
        for p in resnet.layer4.parameters():
            p.requires_grad = True
        self.features = nn.Sequential(*list(resnet.children())[:-2])
    def forward(self, x):
        return self.features(x)

class _PatchEmbedding(nn.Module):
    """Matches original PatchEmbedding class from new_ntau.pth training."""
    def __init__(self, in_channels=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 1)
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)

class _TransformerEncoder(nn.Module):
    """Matches original TransformerEncoder class from new_ntau.pth training."""
    def __init__(self, embed_dim=768, heads=8, depth=6):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=2048, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
    def forward(self, x):
        return self.encoder(x)

class ResNetViT_Legacy(nn.Module):
    """
    Must match the original ResNetViT class EXACTLY as saved in new_ntau.pth.
    Keys: backbone.features.X, patch_embed.proj.X, transformer.encoder.X,
          norm.X, dropout, fc.X
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone    = _ResNetBackbone()
        self.patch_embed = _PatchEmbedding()
        self.transformer = _TransformerEncoder()
        self.norm        = nn.LayerNorm(768)
        self.dropout     = nn.Dropout(0.3)
        self.fc          = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.fc(x)


# ── Shared conv block ────────────────────────────────────────────────────────
class ConvBNReLU(nn.Module):
    def __init__(self, ic, oc, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, k, padding=p, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)


# ── MiniSegNet ───────────────────────────────────────────────────────────────
class MiniSegNet(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNReLU(3,  32), ConvBNReLU(32,  32))
        self.enc2 = nn.Sequential(ConvBNReLU(32, 64), ConvBNReLU(64,  64))
        self.enc3 = nn.Sequential(ConvBNReLU(64,128), ConvBNReLU(128,128))
        self.pool = nn.MaxPool2d(2, 2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(128), nn.Dropout(0.40),
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(0.25), nn.Linear(64, nc)
        )
    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        return self.head(self.gap(x))


# ── CrossAttentionFusion ─────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    def __init__(self, d=512, h=8, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.ffn  = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(drop), nn.Linear(d*2, d)
        )
    def forward(self, cnn_t, vit_t):
        a, _ = self.attn(cnn_t, vit_t, vit_t)
        x = self.n1(cnn_t + a)
        return self.n2(x + self.ffn(x))


# ── HybridNet ────────────────────────────────────────────────────────────────
class HybridNet(nn.Module):
    def __init__(self, nc, backbone_type='resnet50', d_model=512, n_heads=8, dropout=0.2):
        super().__init__()
        self.backbone_type = backbone_type
        if backbone_type == 'resnet50':
            cnn_full = tv_models.resnet50(weights=None)
            self.cnn = nn.Sequential(*list(cnn_full.children())[:-2])
            cnn_ch   = 2048
        else:
            self.cnn = timm.create_model('efficientnet_b4', pretrained=False, features_only=True)
            cnn_ch   = self.cnn.feature_info[-1]['num_chs']
        self.vit      = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.cnn_proj = nn.Linear(cnn_ch, d_model)
        self.vit_proj = nn.Linear(768, d_model)
        self.fusion   = CrossAttentionFusion(d_model, n_heads, dropout)
        self.gap      = nn.AdaptiveAvgPool1d(1)
        self.head     = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(256, nc)
        )

    def _cnn_feat(self, x):
        if self.backbone_type == 'resnet50':
            return self.cnn(x)
        return self.cnn(x)[-1]

    def forward(self, x):
        cf    = self._cnn_feat(x)
        ct    = self.cnn_proj(cf.flatten(2).transpose(1, 2))
        vt    = self.vit_proj(self.vit.forward_features(x)[:, 1:, :])
        fused = self.fusion(ct, vt)
        return self.head(self.gap(fused.transpose(1, 2)).squeeze(-1))

    def get_gradcam_layer(self):
        if self.backbone_type == 'resnet50':
            return self.cnn[-1][-1].conv3
        return self.cnn.blocks[-1][-1].conv_pwl


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_model(path: str) -> None:
    """Download model from its specific Drive file ID if not already present."""
    if not os.path.exists(path):
        fname   = os.path.basename(path)
        file_id = MODEL_DRIVE_IDS.get(fname)
        if file_id is None:
            st.error(f"No Drive file ID configured for: {fname}")
            st.stop()
        with st.spinner(f"Downloading {fname} — please wait…"):
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    path, quiet=False,
                )
                st.success(f"Downloaded {fname}")
            except Exception as exc:
                st.error(f"Download failed: {exc}")
                st.info("Ensure the Google Drive file is shared as 'Anyone with the link'.")
                st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL  — handles OLD (raw state_dict) and NEW (full checkpoint) formats
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI model…")
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ck = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    # ── Detect checkpoint format ──────────────────────────────────────────
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        # NEW format — full checkpoint saved from training notebook
        state_dict = ck['model_state_dict']
        nc         = ck.get('config', {}).get('num_classes', 2)
        mn         = ck.get('model_name', 'ResNet50')
        bt         = ck.get('backbone_type', None)
        # Notebook saves raw folder names e.g. ['normal','parkinson']
        # Map to display names while PRESERVING index order exactly
        raw_cls = ck.get('classes', ['normal', 'parkinson'])
        cls_map = {
            'normal':               'Normal',
            'healthy':              'Normal',
            'parkinson':            "Parkinson\'s Disease",
            'parkinsons':           "Parkinson\'s Disease",
            "parkinson\'s disease":"Parkinson\'s Disease",
        }
        classes = [cls_map.get(c.lower(), c.title()) for c in raw_cls]

    elif isinstance(ck, dict):
        # OLD format — raw state_dict only (new_ntau.pth)
        state_dict = ck
        nc         = 2
        bt         = None
        keys       = list(state_dict.keys())
        if any('backbone.' in k for k in keys) and any('transformer.' in k for k in keys):
            mn = 'ResNetViT_Legacy'
        elif any('transformer.encoder' in k for k in keys):
            mn = 'ResNetViT_Legacy'
        elif any('backbone.features' in k for k in keys):
            mn = 'ResNetViT_Legacy'
        elif any('enc1' in k for k in keys):
            mn = 'MiniSegNet'
        elif any('fusion.attn' in k for k in keys):
            mn = 'HybridNet_RV'
        elif any('patch_embed.proj' in k for k in keys):
            mn = 'ViT-B16'
        elif any('layer4' in k for k in keys):
            mn = 'ResNet50'
        else:
            mn = 'ResNetViT_Legacy'
        # Legacy new_ntau.pth: class 0=Normal, class 1=Parkinson's
        # This matches the original app.py class order
        classes = ['Normal', "Parkinson\'s Disease"]
    else:
        st.error("Unrecognised checkpoint format.")
        st.stop()

    # ── Build correct architecture ────────────────────────────────────────
    if mn == 'ResNetViT_Legacy':
        m = ResNetViT_Legacy(num_classes=nc)
    elif mn == 'ResNet50':
        m   = tv_models.resnet50(weights=None)
        inf = m.fc.in_features
        m.fc = nn.Sequential(
            nn.BatchNorm1d(inf), nn.Dropout(0.45),
            nn.Linear(inf, 512), nn.ReLU(True),
            nn.BatchNorm1d(512), nn.Dropout(0.30),
            nn.Linear(512, nc)
        )
    elif mn == 'EfficientNet-B4':
        m = timm.create_model('efficientnet_b4', pretrained=False, num_classes=nc)
    elif mn == 'ViT-B16':
        m = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=nc)
    elif mn == 'MiniSegNet':
        m = MiniSegNet(nc)
    elif mn in ('HybridNet_RV', 'HybridNet_EV'):
        backbone = bt or ('resnet50' if 'RV' in mn else 'efficientnet_b4')
        m = HybridNet(nc, backbone_type=backbone)
    else:
        st.error(f"Unknown model name in checkpoint: {mn}")
        st.stop()

    # Load weights — strict=True catches mismatches immediately
    try:
        m.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # Show which keys mismatched so we can debug
        missing  = [k for k in state_dict if k not in m.state_dict()]
        extra    = [k for k in m.state_dict() if k not in state_dict]
        st.error(
            f"Weight mismatch loading {mn}.\n"
            f"Missing keys ({len(missing)}): {missing[:5]}\n"
            f"Extra keys ({len(extra)}): {extra[:5]}\n"
            f"Full error: {e}"
        )
        st.stop()
    return m.to(device).eval(), classes, mn, device


# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._acts = self._grads = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, '_acts', o))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, '_grads', go[0]))

    def generate(self, img_tensor, class_idx, device):
        # Reset stored grads/acts before each forward pass
        self._acts  = None
        self._grads = None
        self.model.eval()
        img_tensor = img_tensor.clone().to(device).requires_grad_(True)
        with torch.enable_grad():
            out = self.model(img_tensor)
            self.model.zero_grad()
            out[0, class_idx].backward(retain_graph=True)
        if self._grads is None or self._acts is None:
            return np.zeros((224, 224))
        weights = self._grads.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, (224, 224), mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().detach().numpy()
        lo, hi  = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            return np.zeros((224, 224))
        return cam


def get_gradcam_layer(model, mn):
    """Returns the target conv layer for GradCAM. Robust to model structure."""
    try:
        if mn == 'ResNetViT_Legacy':
            # backbone.features is nn.Sequential of ResNet children[:-2]
            # last element is layer4, last Bottleneck block
            layer4     = model.backbone.features[-1]
            last_block = list(layer4.children())[-1]
            return last_block.conv3

        elif mn == 'ResNet50':
            return model.layer4[-1].conv3

        elif mn == 'EfficientNet-B4':
            # timm EfficientNet — conv_head is the last conv before classifier
            return model.conv_head

        elif mn == 'ViT-B16':
            # ViT has no conv layers — hook on last block norm1 for attention proxy
            return model.blocks[-1].norm1

        elif mn == 'MiniSegNet':
            # enc3 is nn.Sequential of two ConvBNReLU, each has .net
            # enc3[1].net[0] is the Conv2d of the second ConvBNReLU
            return model.enc3[1].net[0]

        elif mn in ('HybridNet_RV', 'HybridNet_EV'):
            return model.get_gradcam_layer()

    except Exception as e:
        pass
    return None


def apply_colormap(orig_pil, cam):
    heatmap  = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    orig_arr = np.array(orig_pil.resize((224, 224))).astype(np.float32)
    overlay  = np.clip(0.55 * orig_arr + 0.45 * heatmap.astype(np.float32), 0, 255).astype(np.uint8)
    return Image.fromarray(overlay), Image.fromarray(heatmap)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORM + PREDICT
# ══════════════════════════════════════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, device, classes, mn, pil_image):
    img_rgb    = pil_image.convert('RGB')
    img_tensor = TRANSFORM(img_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out      = model(img_tensor)
        probs    = F.softmax(out, dim=1)
        probs_np = probs[0].cpu().numpy()
        class_idx      = int(probs_np.argmax())
        confidence_pct = float(probs_np[class_idx] * 100)

    # ── Dynamically find Normal vs Parkinson index ────────────────────────
    # Works regardless of class order saved in the checkpoint
    normal_idx = parkinson_idx = None
    for i, c in enumerate(classes):
        cl = c.lower()
        if 'normal' in cl or 'healthy' in cl:
            normal_idx = i
        if 'parkinson' in cl:
            parkinson_idx = i
    if normal_idx    is None: normal_idx    = 0
    if parkinson_idx is None: parkinson_idx = 1 if len(classes) > 1 else 0

    normal_prob    = float(probs_np[normal_idx]    * 100)
    parkinson_prob = float(probs_np[parkinson_idx] * 100)
    is_parkinson   = (class_idx == parkinson_idx)
    risk = ('High' if confidence_pct >= 85 else 'Moderate') if is_parkinson else 'Low'

    # ── GradCAM — fresh tensor, separate from inference ───────────────────
    overlay = heatmap = None
    gc_layer = get_gradcam_layer(model, mn)
    if gc_layer is not None and mn != 'ViT-B16':
        try:
            gc      = GradCAM(model, gc_layer)
            fresh   = TRANSFORM(img_rgb).unsqueeze(0)
            cam_map = gc.generate(fresh, class_idx, device)
            if cam_map is not None and cam_map.max() > 0:
                overlay, heatmap = apply_colormap(img_rgb, cam_map)
        except Exception as _gc_err:
            # Store error for display in UI
            overlay  = None
            heatmap  = None
            _gradcam_error = str(_gc_err)

    return {
        'prediction':     classes[class_idx],
        'class_idx':      class_idx,
        'is_parkinson':   is_parkinson,
        'confidence':     confidence_pct,
        'normal_prob':    normal_prob,
        'parkinson_prob': parkinson_prob,
        'risk_level':     risk,
        'cam_overlay':    overlay,
        'cam_heatmap':    heatmap,
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image':          img_rgb,
        'model_name':     mn,
        '_probs_np':      probs_np,   # raw probs for ensemble
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
def build_pdf(patient, result):
    """Gold-themed PDF report matching the royal UI aesthetic."""
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=letter,
                               leftMargin=0.75*inch, rightMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
    story  = []
    styles = getSampleStyleSheet()

    # ── Gold colour palette ───────────────────────────────────────────────────
    GOLD        = colors.HexColor('#b8860b')
    GOLD_DARK   = colors.HexColor('#7a4f00')
    GOLD_LIGHT  = colors.HexColor('#fdf3dc')
    PARCHMENT   = colors.HexColor('#fdf8f0')
    DEEP        = colors.HexColor('#2c1a00')
    MIDGOLD     = colors.HexColor('#c9a84c')
    CRIMSON     = colors.HexColor('#c0392b')
    EMERALD     = colors.HexColor('#1a6a38')
    GREY_TEXT   = colors.HexColor('#7a5a1a')

    # ── Paragraph styles ──────────────────────────────────────────────────────
    title_s = ParagraphStyle('T', parent=styles['Heading1'], fontSize=22,
        textColor=GOLD_DARK, spaceAfter=2, alignment=TA_CENTER,
        fontName='Helvetica-Bold')
    subtitle_s = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=9,
        textColor=GOLD, spaceAfter=4, alignment=TA_CENTER,
        fontName='Helvetica')
    project_s = ParagraphStyle('Proj', parent=styles['Normal'], fontSize=10,
        textColor=DEEP, spaceAfter=16, alignment=TA_CENTER,
        fontName='Helvetica-Bold')
    head_s = ParagraphStyle('H', parent=styles['Heading2'], fontSize=12,
        textColor=GOLD_DARK, spaceAfter=6, spaceBefore=12,
        fontName='Helvetica-Bold')
    body_s = ParagraphStyle('B', parent=styles['Normal'], fontSize=10,
        textColor=DEEP, leading=14)
    warn_s = ParagraphStyle('W', parent=styles['Normal'], fontSize=9,
        textColor=colors.HexColor('#7f1d1d'), leading=13,
        backColor=colors.HexColor('#fff1f1'), borderPad=6)
    foot_s = ParagraphStyle('F', parent=styles['Normal'], fontSize=8,
        textColor=GREY_TEXT, alignment=TA_CENTER)

    # ── Gold divider helper ───────────────────────────────────────────────────
    def gold_line():
        t = Table([['', '', '']], colWidths=[0.5*inch, 6.5*inch, 0.5*inch])
        t.setStyle(TableStyle([
            ('LINEABOVE',  (1,0),(1,0), 1, MIDGOLD),
            ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0),(-1,-1), 0),
            ('BOTTOMPADDING',(0,0),(-1,-1), 4),
        ]))
        return t

    # ── Gold table helper ─────────────────────────────────────────────────────
    def gold_tbl(rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            # Header row — dark gold
            ('BACKGROUND',     (0,0),(-1,0), GOLD_DARK),
            ('TEXTCOLOR',      (0,0),(-1,0), colors.white),
            ('FONTNAME',       (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',       (0,0),(-1,0), 10),
            ('BOTTOMPADDING',  (0,0),(-1,0), 8),
            ('TOPPADDING',     (0,0),(-1,0), 8),
            # Alternating parchment / white rows
            ('ROWBACKGROUNDS', (0,1),(-1,-1), [GOLD_LIGHT, PARCHMENT]),
            # Grid in gold
            ('GRID',           (0,0),(-1,-1), 0.4, MIDGOLD),
            ('FONTNAME',       (0,1),(-1,-1), 'Helvetica'),
            ('FONTSIZE',       (0,1),(-1,-1), 10),
            ('TEXTCOLOR',      (0,1),(-1,-1), DEEP),
            ('TOPPADDING',     (0,1),(-1,-1), 6),
            ('BOTTOMPADDING',  (0,1),(-1,-1), 6),
            ('LEFTPADDING',    (0,0),(-1,-1), 10),
            # Left column label styling
            ('FONTNAME',       (0,1),(0,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR',      (0,1),(0,-1), GOLD_DARK),
        ]))
        return t

    # ════════════════════════════════════════════════
    #  HEADER
    # ════════════════════════════════════════════════
    story.append(Paragraph('NeuroScan AI', title_s))
    story.append(Paragraph("Parkinson's Disease MRI Analysis Report", subtitle_s))
    story.append(Paragraph(
        'PARKINSONS DISEASE DETECTION USING DEEP LEARNING ON BRAIN MRI',
        project_s))
    story.append(gold_line())
    story.append(Spacer(1, 4))

    # Diagnosis verdict banner
    is_parkinson = result.get('is_parkinson', False)
    diag_color   = CRIMSON if is_parkinson else EMERALD
    diag_text    = result['prediction'].upper()
    verdict_tbl  = Table([[diag_text]], colWidths=[7*inch])
    verdict_tbl.setStyle(TableStyle([
        ('BACKGROUND',     (0,0),(-1,-1), diag_color),
        ('TEXTCOLOR',      (0,0),(-1,-1), colors.white),
        ('FONTNAME',       (0,0),(-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',       (0,0),(-1,-1), 16),
        ('ALIGN',          (0,0),(-1,-1), 'CENTER'),
        ('TOPPADDING',     (0,0),(-1,-1), 10),
        ('BOTTOMPADDING',  (0,0),(-1,-1), 10),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    story.append(verdict_tbl)
    story.append(Spacer(1, 10))

    # ════════════════════════════════════════════════
    #  PATIENT INFORMATION
    # ════════════════════════════════════════════════
    story.append(Paragraph('Patient Information', head_s))
    story.append(gold_line())
    story.append(gold_tbl([
        ['Field',            'Details'],
        ['Patient Name',     patient['name']],
        ['Patient ID',       patient['patient_id']],
        ['Age',              str(patient['age'])],
        ['Gender',           patient['gender']],
        ['Scan Date',        patient['scan_date']],
        ['Referring Doctor', patient.get('doctor', '—')],
    ], [2.2*inch, 4.8*inch]))

    if patient.get('medical_history', '').strip():
        story.append(Spacer(1, 8))
        story.append(Paragraph('Medical History', head_s))
        story.append(gold_line())
        story.append(Paragraph(patient['medical_history'], body_s))

    # ════════════════════════════════════════════════
    #  AI RESULTS
    # ════════════════════════════════════════════════
    story.append(Paragraph('AI Analysis Results', head_s))
    story.append(gold_line())
    story.append(gold_tbl([
        ['Metric',                   'Value'],
        ['Diagnosis',                result['prediction']],
        ['Confidence Score',         f"{result['confidence']:.2f}%"],
        ['Normal Probability',       f"{result['normal_prob']:.2f}%"],
        ["Parkinson's Probability",  f"{result['parkinson_prob']:.2f}%"],
        ['Risk Level',               result['risk_level']],
        ['Analysis Time',            result['timestamp']],
        ['AI Model',                 result['model_name']],
    ], [2.8*inch, 4.2*inch]))

    # ════════════════════════════════════════════════
    #  ALL MODELS RESULTS (if available)
    # ════════════════════════════════════════════════
    all_mr = result.get('_all_model_results')
    if all_mr:
        story.append(Spacer(1, 10))
        story.append(Paragraph('All Models Comparison', head_s))
        story.append(gold_line())
        rows = [['Model', 'Prediction', 'Confidence', "Parkinson's %", 'Normal %', 'Risk']]
        for mname, mr in all_mr.items():
            rows.append([
                mname,
                mr['prediction'],
                f"{mr['confidence']:.1f}%",
                f"{mr['parkinson_prob']:.1f}%",
                f"{mr['normal_prob']:.1f}%",
                mr['risk_level'],
            ])
        story.append(gold_tbl(rows, [1.6*inch, 1.1*inch, 1*inch, 1.1*inch, 1*inch, 0.8*inch]))

    # ════════════════════════════════════════════════
    #  BRAIN MRI + GRAD-CAM
    # ════════════════════════════════════════════════
    story.append(Spacer(1, 10))
    story.append(Paragraph('Brain MRI Scan & Grad-CAM Heatmap', head_s))
    story.append(gold_line())

    img_buf = io.BytesIO()
    result['image'].save(img_buf, 'PNG')
    img_buf.seek(0)

    if result.get('cam_overlay') is not None:
        cam_buf = io.BytesIO()
        result['cam_overlay'].save(cam_buf, 'PNG')
        cam_buf.seek(0)
        img_tbl = Table([[
            RLImage(img_buf,  width=2.8*inch, height=2.8*inch),
            RLImage(cam_buf,  width=2.8*inch, height=2.8*inch),
        ]], colWidths=[3.2*inch, 3.2*inch])
        cap_txt = 'Left: Original MRI &nbsp;&nbsp;|&nbsp;&nbsp; Right: Grad-CAM Overlay'
    else:
        img_tbl = Table([[
            RLImage(img_buf, width=2.8*inch, height=2.8*inch),
        ]], colWidths=[3.2*inch])
        cap_txt = 'Original MRI Scan'

    img_tbl.setStyle(TableStyle([
        ('ALIGN',            (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',           (0,0),(-1,-1), 'MIDDLE'),
        ('BOX',              (0,0),(-1,-1), 1.5, MIDGOLD),
        ('BACKGROUND',       (0,0),(-1,-1), PARCHMENT),
    ]))
    story.append(img_tbl)
    cap_s = ParagraphStyle('C', parent=styles['Normal'], fontSize=8,
        textColor=GOLD, alignment=TA_CENTER, spaceBefore=4)
    story.append(Paragraph(cap_txt, cap_s))

    # ════════════════════════════════════════════════
    #  DISCLAIMER + FOOTER
    # ════════════════════════════════════════════════
    story.append(Spacer(1, 14))
    story.append(gold_line())
    story.append(Paragraph(
        '⚠️ DISCLAIMER: This report is generated by an AI system for research and '
        'educational purposes only. It must NOT replace clinical diagnosis by a '
        'qualified medical professional. Always consult a licensed neurologist.',
        warn_s))
    story.append(Spacer(1, 16))
    story.append(gold_line())
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"NeuroScan AI | Parkinson's Disease Detection | BVC College of Engineering, Palacharla | "
        # (collapsed into line above)
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        foot_s))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_logo_b64(path):
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = base64.b64encode(f.read()).decode()
            ext  = path.rsplit('.', 1)[-1].lower()
            mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'
            return f'data:{mime};base64,{data}'
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroScan AI — Parkinson's MRI Analysis",
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ══════════════════════════════════════════════════════════════════════════════
#  ROYAL CSS  (identical to original)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,900;1,400;1,600&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Cinzel:wght@400;500;600;700;900&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');
:root {
  --midnight:#fdf8f0;--navy-deep:#fff9f2;--navy:#fef6ec;--navy-mid:#fdf3e7;
  --navy-light:#fceedd;--navy-card:#fffaf4;
  --gold:#b8860b;--gold-bright:#9a6e00;--gold-dim:#c9a84c;
  --gold-muted:rgba(184,134,11,0.10);--gold-glow:rgba(184,134,11,0.25);
  --gold-line:rgba(184,134,11,0.20);--gold-border:rgba(184,134,11,0.40);
  --crimson:#c0392b;--crimson-dim:rgba(192,57,43,0.10);
  --emerald:#1e8449;--emerald-dim:rgba(30,132,73,0.10);
  --parchment:#6b4c11;--text:#2c1a00;--text-2:#7a5a1a;--text-3:#b89040;
  --radius:12px;--radius-sm:7px;
  --shadow-royal:0 4px 24px rgba(184,134,11,0.10),0 1px 6px rgba(0,0,0,0.06);
  --shadow-gold:0 4px 20px rgba(184,134,11,0.18);
}
html,body,.stApp{background:#fdf8f0!important;font-family:'EB Garamond','Cormorant Garamond',serif!important;color:var(--text)!important;}
.stApp::before{content:'';position:fixed;inset:0;z-index:0;background-image:radial-gradient(ellipse 80% 50% at 50% 0%,rgba(184,134,11,0.06) 0%,transparent 60%),repeating-linear-gradient(45deg,transparent,transparent 34px,rgba(184,134,11,0.025) 34px,rgba(184,134,11,0.025) 35px),repeating-linear-gradient(-45deg,transparent,transparent 34px,rgba(184,134,11,0.025) 34px,rgba(184,134,11,0.025) 35px);pointer-events:none;}
.block-container{padding-top:0!important;max-width:1380px!important;position:relative;z-index:1;}
#MainMenu,footer,header{visibility:hidden;}
.card{background:linear-gradient(145deg,#fffdf8 0%,#fff8ec 100%);border:1px solid var(--gold-line);border-radius:var(--radius);padding:1.8rem 2rem;margin:.7rem 0;box-shadow:var(--shadow-royal);position:relative;transition:border-color .3s,box-shadow .3s;}
.card::before{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:60%;height:1px;background:linear-gradient(90deg,transparent,var(--gold-dim),transparent);}
.card:hover{border-color:var(--gold-border);box-shadow:var(--shadow-royal),0 0 30px rgba(201,168,76,0.08);}
.sec-head{display:flex;align-items:center;gap:.9rem;font-family:'Cinzel',serif;font-size:.68rem;font-weight:600;color:var(--gold);letter-spacing:3.5px;text-transform:uppercase;margin:2rem 0 1rem;}
.sec-head::before,.sec-head::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--gold-dim),transparent);}
.sec-head::before{background:linear-gradient(90deg,transparent,var(--gold-dim));}
.sec-head::after{background:linear-gradient(90deg,var(--gold-dim),transparent);}
.stTextInput input,.stNumberInput input,.stTextArea textarea,.stDateInput input{background:#fffdf8!important;color:var(--text)!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius-sm)!important;font-family:'EB Garamond',serif!important;font-size:1rem!important;transition:border-color .25s,box-shadow .25s!important;}
.stTextInput input:focus,.stNumberInput input:focus,.stTextArea textarea:focus{border-color:var(--gold)!important;box-shadow:0 0 0 3px rgba(201,168,76,0.12)!important;outline:none!important;}
div[data-baseweb="select"]>div{background:#fffdf8!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius-sm)!important;color:var(--text)!important;font-family:'EB Garamond',serif!important;}
label{color:var(--text-2)!important;font-family:'Cinzel',serif!important;font-size:.68rem!important;font-weight:600!important;letter-spacing:1.5px!important;text-transform:uppercase!important;}
.stButton>button{background:linear-gradient(135deg,#8a6e2f 0%,#c9a84c 40%,#e8c96a 60%,#c9a84c 80%,#8a6e2f 100%)!important;color:#07090f!important;border:1px solid rgba(232,201,106,0.5)!important;border-radius:var(--radius-sm)!important;padding:.75rem 1.6rem!important;font-family:'Cinzel',serif!important;font-size:.78rem!important;font-weight:700!important;letter-spacing:2.5px!important;text-transform:uppercase!important;width:100%!important;transition:all .3s ease!important;box-shadow:0 3px 16px rgba(201,168,76,0.3),inset 0 1px 0 rgba(255,255,255,0.2)!important;position:relative!important;overflow:hidden!important;}
.stButton>button:hover{background:linear-gradient(135deg,#9a7e3f 0%,#d9b85c 40%,#f8d97a 60%,#d9b85c 80%,#9a7e3f 100%)!important;transform:translateY(-2px)!important;box-shadow:0 6px 24px rgba(201,168,76,0.45),inset 0 1px 0 rgba(255,255,255,0.25)!important;}
.stButton>button:active{transform:translateY(0)!important;box-shadow:0 2px 8px rgba(201,168,76,0.3)!important;}
.stDownloadButton>button{background:linear-gradient(135deg,#0a2e18 0%,#1a6a38 40%,#2a8a50 60%,#1a6a38 80%,#0a2e18 100%)!important;color:#c9e8c8!important;border:1px solid rgba(74,222,128,0.3)!important;border-radius:var(--radius-sm)!important;padding:.75rem 1.6rem!important;font-family:'Cinzel',serif!important;font-size:.78rem!important;font-weight:700!important;letter-spacing:2.5px!important;text-transform:uppercase!important;width:100%!important;transition:all .3s ease!important;box-shadow:0 3px 16px rgba(30,132,73,0.3)!important;}
.stDownloadButton>button:hover{background:linear-gradient(135deg,#0d3a1e 0%,#1f7a42 40%,#32a060 60%,#1f7a42 80%,#0d3a1e 100%)!important;transform:translateY(-2px)!important;box-shadow:0 6px 24px rgba(30,132,73,0.45)!important;}
[data-testid="stFileUploader"]{background:#fffdf8!important;border:2px dashed var(--gold-line)!important;border-radius:var(--radius)!important;padding:1.6rem!important;transition:border-color .3s,box-shadow .3s!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--gold)!important;box-shadow:0 0 24px var(--gold-muted)!important;}
.diag-badge{display:inline-flex;align-items:center;gap:.5rem;font-family:'Cinzel',serif;font-size:1.3rem;font-weight:700;padding:.5rem 1.2rem;border-radius:4px;letter-spacing:1px;}
.diag-normal{background:var(--emerald-dim);color:#4ade80;border:1px solid rgba(74,222,128,0.35);box-shadow:0 0 20px rgba(74,222,128,0.1);}
.diag-parkinson{background:var(--crimson-dim);color:#f87171;border:1px solid rgba(248,113,113,0.35);box-shadow:0 0 20px rgba(248,113,113,0.1);}
.stat-tile{background:linear-gradient(145deg,#fffdf8,#fff6e8);border:1px solid var(--gold-line);border-radius:var(--radius);padding:1.1rem 1.3rem;text-align:center;box-shadow:var(--shadow-royal);position:relative;}
.stat-value{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#9a6e00;line-height:1.1;margin-bottom:.3rem;}
.stat-label{font-family:'Cinzel',serif;font-size:.58rem;color:var(--text-2);letter-spacing:2px;text-transform:uppercase;}
.risk-low{background:rgba(74,222,128,.1);color:#4ade80;border:1px solid rgba(74,222,128,.3);border-radius:4px;padding:.25rem 1rem;font-size:.75rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1px;}
.risk-moderate{background:rgba(251,191,36,.1);color:#fbbf24;border:1px solid rgba(251,191,36,.3);border-radius:4px;padding:.25rem 1rem;font-size:.75rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1px;}
.risk-high{background:rgba(248,113,113,.1);color:#f87171;border:1px solid rgba(248,113,113,.3);border-radius:4px;padding:.25rem 1rem;font-size:.75rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1px;}
.prob-row{margin:.6rem 0;}
.prob-label{font-family:'Cinzel',serif;font-size:.65rem;color:var(--text-2);margin-bottom:.3rem;display:flex;justify-content:space-between;letter-spacing:1px;}
.prob-track{background:rgba(255,255,255,.04);border-radius:3px;height:7px;overflow:hidden;border:1px solid rgba(201,168,76,0.1);}
.prob-fill-n{height:100%;border-radius:3px;background:linear-gradient(90deg,#065f46,#4ade80);transition:width .9s cubic-bezier(.22,1,.36,1);}
.prob-fill-p{height:100%;border-radius:3px;background:linear-gradient(90deg,#7f1d1d,#f87171);transition:width .9s cubic-bezier(.22,1,.36,1);}
.img-frame{border:1px solid var(--gold-line);border-radius:var(--radius);overflow:hidden;background:#f5f0e8;box-shadow:var(--shadow-royal);}
.img-caption{font-family:'Cinzel',serif;font-size:.6rem;color:#b89040;text-align:center;padding:.45rem 0 .25rem;letter-spacing:1.5px;text-transform:uppercase;background:rgba(245,240,232,0.8);}
.hm-legend{display:flex;align-items:center;gap:.6rem;font-family:'Cinzel',serif;font-size:.6rem;color:#b89040;margin-top:.6rem;letter-spacing:1px;}
.hm-bar{flex:1;height:6px;border-radius:3px;background:linear-gradient(90deg,#00008b,#0000ff,#00ff00,#ffff00,#ff0000);}
.stTabs [data-baseweb="tab-list"]{background:#fffdf8!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius)!important;padding:.35rem!important;gap:.25rem!important;}
.stTabs [data-baseweb="tab"]{font-family:'Cinzel',serif!important;font-size:.75rem!important;font-weight:600!important;color:#7a5a1a!important;border-radius:var(--radius-sm)!important;padding:.6rem 1.3rem!important;letter-spacing:1.5px!important;transition:all .25s!important;}
.stTabs [data-baseweb="tab"]:hover{color:#9a6e00!important;}
.stTabs [aria-selected="true"]{background:rgba(184,134,11,0.10)!important;color:#7a4f00!important;border:1px solid rgba(184,134,11,0.3)!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#fdf8f0,#fdf3e7)!important;border-right:1px solid var(--gold-line)!important;}
[data-testid="stSidebar"] *{font-family:'EB Garamond',serif!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-family:'Cinzel',serif!important;color:#9a6e00!important;letter-spacing:2px!important;}
[data-testid="stMetric"]{background:linear-gradient(145deg,#fffdf8,#fff6e8)!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius)!important;padding:1rem 1.2rem!important;}
[data-testid="stMetricValue"]{font-family:'Playfair Display',serif!important;font-size:1.6rem!important;font-weight:700!important;color:#9a6e00!important;}
[data-testid="stMetricLabel"]{color:#7a5a1a!important;font-family:'Cinzel',serif!important;font-size:.58rem!important;letter-spacing:1.5px!important;text-transform:uppercase!important;}
.stProgress>div>div>div{background:linear-gradient(90deg,var(--gold-dim),var(--gold-bright))!important;border-radius:3px!important;}
[data-testid="stDataFrame"]{border-radius:var(--radius)!important;border:1px solid var(--gold-line)!important;overflow:hidden!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#fdf8f0;}
::-webkit-scrollbar-thumb{background:#c9a84c;border-radius:10px;}
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,var(--gold-line),transparent)!important;margin:1.8rem 0!important;}
.stAlert{border-radius:var(--radius)!important;font-family:'EB Garamond',serif!important;}
.team-card{background:linear-gradient(145deg,#fffdf8,#fff6e8);border:1px solid var(--gold-line);border-radius:var(--radius);padding:1.4rem 1rem;text-align:center;transition:border-color .3s,transform .3s;}
.team-card:hover{border-color:var(--gold-border);transform:translateY(-4px);box-shadow:var(--shadow-gold);}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
logo_src  = get_logo_b64('logo.png')
logo_html = (
    f'<img src="{logo_src}" style="width:80px;height:80px;object-fit:contain;'
    f'border-radius:50%;border:2px solid rgba(184,134,11,0.5);'
    f'box-shadow:0 0 20px rgba(184,134,11,0.2);"/>'
    if logo_src else
    '<div style="width:80px;height:80px;background:linear-gradient(145deg,#fdf3dc,#fce8b8);'
    'border:2px solid rgba(184,134,11,0.5);border-radius:50%;display:flex;align-items:center;'
    'justify-content:center;font-size:2.2rem;box-shadow:0 0 20px rgba(184,134,11,0.2);">🧠</div>'
)

st.markdown(
    '<div style="background:linear-gradient(180deg,#fdf3dc 0%,#fdf8f0 60%,#fdf8f0 100%);'
    'border-bottom:1px solid rgba(184,134,11,0.22);padding:2.8rem 2rem 2.4rem;'
    'display:flex;flex-direction:column;align-items:center;gap:1rem;'
    'position:relative;overflow:hidden;">'
    '<div style="position:absolute;top:0;left:0;right:0;height:3px;'
    'background:linear-gradient(90deg,transparent,#c9a84c,#e8c96a,#c9a84c,transparent);"></div>'
    '<div style="position:absolute;top:14px;left:24px;font-size:.9rem;'
    'color:rgba(184,134,11,0.4);font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'
    '<div style="position:absolute;top:14px;right:24px;font-size:.9rem;'
    'color:rgba(184,134,11,0.4);font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'
    + logo_html +
    '<div style="font-family:Cinzel,serif;font-size:2.8rem;font-weight:900;color:#2c1a00;'
    'letter-spacing:8px;line-height:1;text-align:center;'
    'text-shadow:0 2px 12px rgba(184,134,11,0.15);">'
    'NEUROSCAN&nbsp;<span style="color:#b8860b;">AI</span></div>'
    '<div style="font-family:Cormorant Garamond,serif;font-size:1rem;font-style:italic;'
    'color:#9a7030;letter-spacing:4px;text-align:center;">'
    "Parkinson's Detection · Brain MRI · Deep Learning</div>"
    # ── Project title ──────────────────────────────────────────────────
    '<div style="background:linear-gradient(135deg,rgba(184,134,11,0.12),rgba(184,134,11,0.06));'
    'border:1px solid rgba(184,134,11,0.35);border-radius:8px;'
    'padding:.7rem 2rem;margin-top:.3rem;position:relative;overflow:hidden;">'
    '<div style="position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(232,201,106,0.08),transparent);"></div>'
    '<div style="font-family:Cinzel,serif;font-size:.95rem;font-weight:700;'
    'color:#7a4f00;letter-spacing:3px;text-align:center;text-transform:uppercase;'
    'text-shadow:0 1px 4px rgba(184,134,11,0.2);position:relative;z-index:1;">'
    'Parkinson\u2019s Disease Detection Using Deep Learning on Brain MRI'
    '</div></div>'
    '<div style="display:flex;align-items:center;gap:1rem;width:60%;max-width:400px;">'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,#c9a84c);"></div>'
    '<span style="color:#b8860b;font-size:.7rem;letter-spacing:4px;">✦</span>'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,#c9a84c,transparent);"></div>'
    '</div>'
    '<div style="display:flex;gap:1.2rem;flex-wrap:wrap;justify-content:center;">'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:3px;padding:.3rem 1rem;font-family:Cinzel,serif;font-size:.65rem;'
    'color:#8a5e00;letter-spacing:1.5px;">⚙ ResNet50+ViT / EffNet+ViT</span>'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:3px;padding:.3rem 1rem;font-family:Cinzel,serif;font-size:.65rem;'
    'color:#8a5e00;letter-spacing:1.5px;">◈ 100% 5-Fold CV Accuracy</span>'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:3px;padding:.3rem 1rem;font-family:Cinzel,serif;font-size:.65rem;'
    'color:#8a5e00;letter-spacing:1.5px;">✦ Grad-CAM XAI</span>'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:3px;padding:.3rem 1rem;font-family:Cinzel,serif;font-size:.65rem;'
    'color:#8a5e00;letter-spacing:1.5px;">⬡ Batch Analysis</span>'
    '</div>'
    '<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
    'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ('prediction_made',   False),
    ('patient_data',      {}),
    ('prediction_result', {}),
    ('batch_results',     []),
    ('last_model',        ''),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — info only, no model selector (moved to main page)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:.8rem 0 1.2rem;">'
        '<div style="font-family:Cinzel,serif;font-size:1.05rem;font-weight:700;'
        'color:#8a5e00;letter-spacing:3px;">⚕ NEUROSCAN AI</div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,'
        'rgba(184,134,11,0.4),transparent);margin:.6rem 0;"></div></div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Advanced hybrid deep-learning system for brain MRI analysis.\n\n"
        "Upload a scan and click **Analyze** — all 6 models run automatically."
    )
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,'
        'rgba(184,134,11,0.25),transparent);margin:.8rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#b8860b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.6rem;">⚙ Model Suite</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.92rem;color:#6b4c11;line-height:2.1;">'
        '⚜ ResNet50 (CNN)<br>'
        '⚜ EfficientNet-B4 (CNN)<br>'
        '⚜ ViT-B/16 (Transformer)<br>'
        '⚜ MiniSegNet (U-Net)<br>'
        '⚜ HybridNet RV (ResNet+ViT)<br>'
        '⚜ HybridNet EV (EffNet+ViT)<br>'
        '⚜ Ensemble (weighted avg)'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,'
        'rgba(184,134,11,0.25),transparent);margin:.8rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#b8860b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.6rem;">◈ Performance</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1: st.metric('CV Accuracy', '100%')
    with c2: st.metric('AUC', '1.000')
    c3, c4 = st.columns(2)
    with c3: st.metric('Precision', '100%')
    with c4: st.metric('Recall',    '98%')
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,'
        'rgba(184,134,11,0.25),transparent);margin:.8rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#b8860b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.6rem;">✦ Features</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.92rem;color:#6b4c11;line-height:2;">'
        '⚜ Single-scan analysis<br>⚜ Batch processing<br>'
        '⚜ Grad-CAM heatmap<br>⚜ Risk scoring<br>'
        '⚜ Gold PDF report<br>⚜ CSV export'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,'
        'rgba(184,134,11,0.25),transparent);margin:.8rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.warning("⚠️ For **research/academic** purposes only. Not for clinical diagnosis.")

# MODEL_PATH used for single-model fallback (batch, report etc.)
# All models are downloaded & run at analyze time
MODEL_PATH = list(MODEL_FILES.values())[0]   # default = HybridNet_RV


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_scan, tab_batch, tab_about = st.tabs([
    '🧠  Single MRI Analysis',
    '📦  Batch Analysis',
    '🏛  About',
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — SINGLE MRI
# ─────────────────────────────────────────────────────────────────────────────
with tab_scan:

    # ── Model cards row — shows all 6 models with status ─────────────────────
    st.markdown('<div class="sec-head">✦ All Models — Auto-Run on Analyze</div>',
                unsafe_allow_html=True)

    MODEL_INFO = {
        "HybridNet_RV (ResNet50 + ViT)":     {"icon":"🔀","short":"HybridNet RV", "acc":"98.4%","desc":"ResNet50 + ViT Cross-Attention"},
        "HybridNet_EV (EfficientNet + ViT)": {"icon":"🔀","short":"HybridNet EV", "acc":"99.2%","desc":"EfficientNet + ViT Cross-Attention"},
        "ResNet50":                           {"icon":"🧱","short":"ResNet50",     "acc":"100%", "desc":"CNN Backbone"},
        "EfficientNet-B4":                    {"icon":"⚡","short":"EffNet-B4",    "acc":"100%", "desc":"Compound-Scaled CNN"},
        "ViT-B16":                            {"icon":"👁","short":"ViT-B/16",     "acc":"99.9%","desc":"Vision Transformer"},
        "MiniSegNet":                         {"icon":"🗺","short":"MiniSegNet",   "acc":"97.6%","desc":"Lightweight U-Net"},
    }
    mc_cols = st.columns(6, gap="small")
    for ci, (mkey, minfo) in enumerate(MODEL_INFO.items()):
        with mc_cols[ci]:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.25);border-radius:10px;'
                f'padding:.8rem .5rem;text-align:center;'
                f'border-top:3px solid #c9a84c;'
                f'box-shadow:0 2px 10px rgba(184,134,11,0.08);">'
                f'<div style="font-size:1.5rem;margin-bottom:.3rem;">{minfo["icon"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.62rem;font-weight:700;'
                f'color:#7a4f00;letter-spacing:.5px;margin-bottom:.2rem;">{minfo["short"]}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.78rem;'
                f'color:#9a7030;margin-bottom:.3rem;">{minfo["desc"]}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:.9rem;'
                f'font-weight:700;color:#4CAF50;">{minfo["acc"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.5rem;color:#b89040;'
                f'letter-spacing:1px;margin-top:.2rem;">5-FOLD CV ACC</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,'
        'rgba(201,168,76,0.2),transparent);margin:1.2rem 0 .5rem;"></div>',
        unsafe_allow_html=True,
    )

    col_form, col_upload = st.columns([1, 1], gap='large')

    with col_form:
        st.markdown('<div class="sec-head">✦ Patient Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        patient_name = st.text_input('Full Name *', placeholder="Patient's full name")
        c1, c2 = st.columns(2)
        with c1: patient_age    = st.number_input('Age *', 0, 120, 45)
        with c2: patient_gender = st.selectbox('Gender *', ['Male', 'Female', 'Other'])
        c3, c4 = st.columns(2)
        with c3: patient_id = st.text_input('Patient ID *', placeholder='P-2024-0001')
        with c4: scan_date  = st.date_input('Scan Date *', value=datetime.now())
        referring_doctor = st.text_input('Referring Doctor', placeholder='Dr. Name (optional)')
        medical_history  = st.text_area('Medical History', placeholder='Relevant history (optional)', height=90)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_upload:
        st.markdown('<div class="sec-head">✦ MRI Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            'Upload Brain MRI Scan',
            type=['png', 'jpg', 'jpeg'],
            help='PNG / JPG / JPEG · any resolution',
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(image, caption='Uploaded MRI', use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            w, h = image.size
            st.markdown(
                f'<div style="font-family:Cinzel,monospace;font-size:.62rem;color:#9a7030;'
                f'margin-top:.5rem;text-align:center;letter-spacing:1px;">'
                f'{image.mode} · {w}×{h}px · {uploaded_file.size/1024:.1f} KB</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info('Upload a brain MRI scan to begin analysis.')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,'
        'rgba(201,168,76,0.2),transparent);margin:1.5rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sec-head">✦ Analysis & Results</div>', unsafe_allow_html=True)

    btn_col, report_col = st.columns([1, 1], gap='large')

    with btn_col:
        if st.button('⚕ Analyze MRI Scan'):
            if not patient_name.strip():
                st.error("Please enter the patient's full name.")
            elif not patient_id.strip():
                st.error('Please enter a Patient ID.')
            elif uploaded_file is None:
                st.error('Please upload a brain MRI scan.')
            else:

                with st.spinner('Running all 6 models + ensemble…'):
                    st.session_state.patient_data = {
                        'name':            patient_name.strip(),
                        'age':             patient_age,
                        'gender':          patient_gender,
                        'patient_id':      patient_id.strip(),
                        'scan_date':       scan_date.strftime('%Y-%m-%d'),
                        'doctor':          referring_doctor.strip() or '—',
                        'medical_history': medical_history.strip(),
                    }
                    try:
                        # ── Run every model and collect results ───────────────
                        all_model_results = {}
                        all_probs_list    = []
                        ensemble_weights  = [0.8, 0.9, 0.8, 0.6, 1.2, 1.2]
                        model_order       = list(MODEL_FILES.keys())

                        for idx_m, (mc, mpath) in enumerate(MODEL_FILES.items()):
                            download_model(mpath)
                            m, cls, mn, dev = load_model(mpath)
                            r = predict(m, dev, cls, mn, image)
                            all_model_results[mc] = r
                            all_probs_list.append(r['_probs_np'])

                        # ── Ensemble ──────────────────────────────────────────
                        import numpy as _np
                        ens_p     = _np.average(_np.stack(all_probs_list),
                                                axis=0, weights=ensemble_weights)
                        ens_idx   = int(ens_p.argmax())
                        ens_conf  = float(ens_p[ens_idx] * 100)
                        # Use classes from first model
                        first_cls = list(all_model_results.values())[0]
                        ens_cls   = first_cls['prediction'][:1]  # placeholder
                        _cls_list = cls  # last loaded classes
                        ens_label = _cls_list[ens_idx]
                        # Find parkinson index
                        _park_idx = next(
                            (i for i, c in enumerate(_cls_list) if 'parkinson' in c.lower()), 1)
                        ens_is_pk = (ens_idx == _park_idx)
                        ens_norm_p  = float(ens_p[1-_park_idx] * 100) if len(ens_p)>1 else 0.0
                        ens_park_p  = float(ens_p[_park_idx]   * 100)
                        ens_risk = ('High' if ens_conf >= 85 else 'Moderate') if ens_is_pk else 'Low'
                        ens_result = {
                            'prediction':     ens_label,
                            'class_idx':      ens_idx,
                            'is_parkinson':   ens_is_pk,
                            'confidence':     ens_conf,
                            'normal_prob':    ens_norm_p,
                            'parkinson_prob': ens_park_p,
                            'risk_level':     ens_risk,
                            'cam_overlay':    None,
                            'cam_heatmap':    None,
                            'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'image':          list(all_model_results.values())[0]['image'],
                            'model_name':     'Ensemble',
                        }

                        # Store everything in session
                        st.session_state.all_model_results  = all_model_results
                        st.session_state.ensemble_result    = ens_result
                        st.session_state.prediction_result  = ens_result   # default display
                        st.session_state.prediction_made    = True
                        st.success('All models analysed!')
                        st.balloons()
                        st.rerun()
                    except Exception as exc:
                        st.error(f'Error during analysis: {exc}')

    # ── RESULTS ──────────────────────────────────────────────────────────────
    if st.session_state.prediction_made:
        r         = st.session_state.prediction_result
        is_normal = not r.get('is_parkinson', r['class_idx'] == 0)
        diag_cls  = 'diag-normal' if is_normal else 'diag-parkinson'
        risk_cls  = f"risk-{r['risk_level'].lower()}"

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:Cinzel,serif;font-size:.72rem;font-weight:600;'
            'color:#8a5e00;letter-spacing:2.5px;text-transform:uppercase;'
            'text-align:center;margin-bottom:1.2rem;">◈ Diagnostic Summary ◈</div>',
            unsafe_allow_html=True,
        )
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Diagnosis</div>'
                f'<div style="margin-top:.5rem;"><span class="{diag_cls} diag-badge">'
                f'{"✦" if is_normal else "⚠"} {r["prediction"].upper()}'
                f'</span></div></div>',
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Confidence</div>'
                f'<div class="stat-value">{r["confidence"]:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with d3:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Risk Level</div>'
                f'<div style="margin-top:.6rem;">'
                f'<span class="{risk_cls}">{r["risk_level"]} Risk</span></div></div>',
                unsafe_allow_html=True,
            )
        with d4:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Model</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.85rem;'
                f'color:#a89060;margin-top:.35rem;line-height:1.4;">'
                f'{r["model_name"]}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>✦ Normal</span>'
            f'<span>{r["normal_prob"]:.1f}%</span></div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill-n" style="width:{r["normal_prob"]:.1f}%"></div>'
            f'</div></div>'
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>⚠ Parkinson\'s</span>'
            f'<span>{r["parkinson_prob"]:.1f}%</span></div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill-p" style="width:{r["parkinson_prob"]:.1f}%"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Grad-CAM
        # ── All models comparison ────────────────────────────────────────
        if st.session_state.get('all_model_results'):
            st.markdown('<div class="sec-head">✦ All Models Comparison</div>', unsafe_allow_html=True)
            amr = st.session_state.all_model_results
            ens = st.session_state.ensemble_result
            cols_hdr = st.columns(len(amr)+1)
            model_display_names = list(amr.keys()) + ['Ensemble']
            all_display = {**amr, 'Ensemble': ens}
            for ci, mname in enumerate(model_display_names):
                mr = all_display[mname]
                is_pk = mr.get('is_parkinson', False)
                badge_col = '#f87171' if is_pk else '#4ade80'
                badge_txt = 'PARKINSON' if is_pk else 'NORMAL'
                with cols_hdr[ci]:
                    st.markdown(
                        f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                        f'border:1px solid rgba(184,134,11,0.2);border-radius:10px;'
                        f'padding:.8rem .5rem;text-align:center;'
                        f'border-top:3px solid {badge_col};">'
                        f'<div style="font-family:Cinzel,serif;font-size:.58rem;'
                        f'color:#9a7030;letter-spacing:1px;margin-bottom:.4rem;">'
                        f'{mname}</div>'
                        f'<div style="font-family:Playfair Display,serif;font-size:1.1rem;'
                        f'font-weight:700;color:{badge_col};">{badge_txt}</div>'
                        f'<div style="font-family:Cinzel,serif;font-size:.65rem;'
                        f'color:#7a5a1a;margin-top:.3rem;">{mr["confidence"]:.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        if r['cam_overlay'] is not None:
            st.markdown(
                '<div class="sec-head">✦ Grad-CAM Explainability</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<p style="font-family:EB Garamond,serif;font-size:.95rem;color:#a89060;'
                'margin-bottom:1.2rem;line-height:1.7;">'
                "Grad-CAM highlights the brain regions that drove the model's decision. "
                '<strong style="color:#c9a84c;">Warmer colours (red/yellow)</strong> '
                'indicate higher neural attention.</p>',
                unsafe_allow_html=True,
            )
            ic1, ic2, ic3 = st.columns(3)
            for col, img_obj, cap in [
                (ic1, r['image'],       'Original MRI'),
                (ic2, r['cam_heatmap'], 'Attention Heatmap'),
                (ic3, r['cam_overlay'], 'Grad-CAM Overlay'),
            ]:
                with col:
                    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                    st.image(img_obj, use_column_width=True)
                    st.markdown(f'<div class="img-caption">{cap}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="hm-legend">'
                '<span>Low</span><div class="hm-bar"></div><span>High</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with report_col:
        if st.session_state.prediction_made:
            if st.button('📜 Generate PDF Report'):
                with st.spinner('Building royal report…'):
                    try:
                        # Attach all-model results to pass into PDF
                        _pdf_result = dict(st.session_state.prediction_result)
                        if st.session_state.get('all_model_results'):
                            _pdf_result['_all_model_results'] = {
                                k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                                for k, v in st.session_state.all_model_results.items()
                            }
                        pdf_bytes = build_pdf(
                            st.session_state.patient_data,
                            _pdf_result,
                        )
                        p     = st.session_state.patient_data
                        fname = f"NeuroScan_{p['patient_id']}_{datetime.now().strftime('%Y%m%d')}.pdf"
                        st.download_button(
                            '⬇ Download PDF Report',
                            data=pdf_bytes, file_name=fname, mime='application/pdf',
                        )
                        st.success('PDF ready for download!')
                    except Exception as exc:
                        st.error(f'PDF error: {exc}')
        else:
            st.info('Run an analysis first to generate a report.')


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — BATCH
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="sec-head">✦ Batch MRI Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:EB Garamond,serif;font-size:1rem;color:#a89060;'
        'margin-bottom:1.2rem;line-height:1.7;">'
        'Upload multiple brain MRI scans at once. Each is independently analysed '
        'and results are summarised with a downloadable CSV report.</p>',
        unsafe_allow_html=True,
    )

    batch_files = st.file_uploader(
        'Upload Multiple Brain MRI Scans',
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key='batch_uploader',
    )

    if batch_files:
        st.info(f'{len(batch_files)} file(s) ready for processing.')
        if st.button('⚕ Run Batch Analysis'):
            if st.session_state.get('last_model') != MODEL_PATH:
                st.cache_resource.clear()
                st.session_state['last_model'] = MODEL_PATH
            try:
                model, classes, mn, device = load_model(MODEL_PATH)
            except Exception as exc:
                st.error(f'Failed to load model: {exc}'); st.stop()

            batch_results = []
            prog   = st.progress(0)
            status = st.empty()
            for i, f in enumerate(batch_files):
                status.markdown(
                    f'<p style="font-family:Cinzel,serif;font-size:.75rem;color:#c9a84c;'
                    f'letter-spacing:1px;">Processing {f.name} ({i+1}/{len(batch_files)})…</p>',
                    unsafe_allow_html=True,
                )
                try:
                    res = predict(model, device, classes, mn, Image.open(f))
                    res['filename'] = f.name
                    batch_results.append(res)
                except Exception as exc:
                    st.warning(f'Skipped {f.name}: {exc}')
                prog.progress((i+1) / len(batch_files))

            st.session_state.batch_results = batch_results
            status.empty()
            st.success(f'Done! {len(batch_results)} scan(s) processed.')
            st.rerun()

    if st.session_state.batch_results:
        results  = st.session_state.batch_results
        n_total  = len(results)
        n_normal = sum(1 for r in results if r['class_idx'] == 0)
        n_park   = n_total - n_normal
        avg_conf = np.mean([r['confidence'] for r in results])

        st.markdown('<div class="sec-head">✦ Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric('Total Scans',    n_total)
        with m2: st.metric('Normal',         n_normal)
        with m3: st.metric("Parkinson's",    n_park)
        with m4: st.metric('Avg Confidence', f'{avg_conf:.1f}%')

        st.markdown('<div class="sec-head">✦ Distribution</div>', unsafe_allow_html=True)
        ch_col, tb_col = st.columns([1, 1], gap='large')

        with ch_col:
            fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor='#fdf8f0')
            ax.set_facecolor('#fdf8f0')
            if n_normal > 0 or n_park > 0:
                wedges, texts, autotexts = ax.pie(
                    [n_normal, n_park],
                    labels=['Normal', "Parkinson's"],
                    autopct='%1.0f%%',
                    colors=['#4ade80', '#f87171'],
                    startangle=90,
                    wedgeprops=dict(edgecolor='#0a0e1a', linewidth=2.5),
                    textprops=dict(color='#2c1a00', fontsize=10, fontfamily='EB Garamond'),
                )
                for at in autotexts:
                    at.set_color('#ffffff'); at.set_fontweight('bold')
            ax.set_title('Scan Distribution', color='#7a5a1a', fontsize=10, pad=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with tb_col:
            df = pd.DataFrame([{
                'File':          r['filename'],
                'Prediction':    r['prediction'],
                'Confidence':    f"{r['confidence']:.1f}%",
                'Normal %':      f"{r['normal_prob']:.1f}%",
                "Parkinson's %": f"{r['parkinson_prob']:.1f}%",
                'Risk':          r['risk_level'],
                'Model':         r['model_name'],
            } for r in results])
            st.dataframe(df, use_container_width=True, height=280)
            st.download_button(
                '⬇ Download CSV Report',
                data=df.to_csv(index=False),
                file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
            )

        st.markdown('<div class="sec-head">✦ Per-Image Results</div>', unsafe_allow_html=True)
        for r in results:
            is_n = not r.get('is_parkinson', r['class_idx'] == 0)
            col  = '#4ade80' if is_n else '#f87171'
            icon = '✦' if is_n else '⚠'
            rc1, rc2, rc3, rc4 = st.columns([1, 2, 1, 1])
            with rc1:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(r['image'], use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(
                    f'<p style="font-family:Cinzel,serif;font-size:.62rem;'
                    f'color:#9a7030;margin-bottom:.3rem;letter-spacing:1px;">'
                    f'{r["filename"]}</p>'
                    f'<p style="font-family:Playfair Display,serif;font-size:1.4rem;'
                    f'font-weight:700;color:{col};margin:0;">{icon} {r["prediction"]}</p>'
                    f'<p style="font-family:Cinzel,serif;font-size:.6rem;'
                    f'color:#9a7030;margin-top:.35rem;letter-spacing:1px;">'
                    f'{r["model_name"]} · {r["timestamp"]}</p>',
                    unsafe_allow_html=True,
                )
            with rc3:
                st.metric('Confidence', f"{r['confidence']:.1f}%")
                st.metric('Normal %',   f"{r['normal_prob']:.1f}%")
            with rc4:
                if r['cam_overlay'] is not None:
                    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                    st.image(r['cam_overlay'], use_column_width=True)
                    st.markdown('<div class="img-caption">Grad-CAM</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="height:1px;background:linear-gradient(90deg,transparent,'
                'rgba(201,168,76,0.15),transparent);margin:.8rem 0;"></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    college_logo = get_logo_b64('bvcr.jpg')
    clg_html = (
        f'<img src="{college_logo}" style="width:110px;height:110px;object-fit:contain;'
        f'margin-bottom:1.2rem;border:2px solid rgba(184,134,11,0.45);border-radius:10px;'
        f'box-shadow:0 6px 24px rgba(184,134,11,0.18);"/>'
        if college_logo else
        '<div style="font-size:4rem;margin-bottom:1.2rem;">🏛</div>'
    )

    # ── COLLEGE HERO BANNER ───────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(160deg,#fdf3dc 0%,#fdf8f0 50%,#fce8b0 100%);'
        f'border:1px solid rgba(184,134,11,0.28);border-radius:16px;'
        f'padding:3rem 2.5rem 2.5rem;margin-bottom:2rem;text-align:center;'
        f'position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,transparent,#c9a84c,#f0d070,#c9a84c,transparent);"></div>'
        f'<div style="position:absolute;inset:0;opacity:0.03;font-size:8rem;display:flex;'
        f'align-items:center;justify-content:center;color:#b8860b;font-family:Cinzel,serif;'
        f'pointer-events:none;letter-spacing:20px;">✦BVC✦</div>'
        + clg_html +
        f'<div style="font-family:Cinzel,serif;font-size:1.9rem;font-weight:900;color:#2c1a00;'
        f'letter-spacing:4px;margin-bottom:.5rem;text-shadow:0 2px 8px rgba(184,134,11,0.12);">'
        f'BVC College of Engineering</div>'
        f'<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:1.05rem;'
        f'color:#b8860b;letter-spacing:3px;margin-bottom:.8rem;">Palacharla, Andhra Pradesh</div>'
        f'<div style="display:flex;align-items:center;gap:1rem;justify-content:center;margin-bottom:1rem;">'
        f'<div style="flex:1;max-width:120px;height:1px;background:linear-gradient(90deg,transparent,#c9a84c);"></div>'
        f'<span style="color:#b8860b;font-size:.8rem;letter-spacing:5px;">✦</span>'
        f'<div style="flex:1;max-width:120px;height:1px;background:linear-gradient(90deg,#c9a84c,transparent);"></div>'
        f'</div>'
        f'<div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.2rem;">'
        f'<span style="background:rgba(184,134,11,0.1);border:1px solid rgba(184,134,11,0.3);'
        f'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        f'color:#7a4f00;letter-spacing:1.5px;font-weight:600;">🎓 Autonomous</span>'
        f'<span style="background:rgba(30,132,73,0.08);border:1px solid rgba(30,132,73,0.25);'
        f'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        f'color:#1a6a38;letter-spacing:1.5px;font-weight:600;">✓ NAAC A Grade</span>'
        f'<span style="background:rgba(0,100,180,0.07);border:1px solid rgba(0,100,180,0.2);'
        f'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        f'color:#004a90;letter-spacing:1.5px;font-weight:600;">⚙ AICTE Approved</span>'
        f'<span style="background:rgba(140,40,40,0.07);border:1px solid rgba(140,40,40,0.2);'
        f'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        f'color:#7a1a1a;letter-spacing:1.5px;font-weight:600;">📜 Affiliated to JNTUK</span>'
        f'</div>'
        f'<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a5a1a;line-height:1.8;'
        f'max-width:680px;margin:0 auto;">'
        f'BVC College of Engineering is a premier autonomous institution in Andhra Pradesh, '
        f'recognized for academic excellence and cutting-edge research. With state-of-the-art '
        f'laboratories, experienced faculty, and a legacy of producing distinguished engineers, '
        f'the institution stands as a beacon of quality technical education in the region.'
        f'</div>'
        f'<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
        f'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── INSTITUTE STATS ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">✦ Institute at a Glance</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, num, label, icon in [
        (s1, '20+',  'Years of Excellence', '🏆'),
        (s2, '3000+','Students Enrolled',   '👨‍🎓'),
        (s3, '150+', 'Faculty Members',     '👨‍🏫'),
        (s4, '10+',  'Departments',         '🏛'),
        (s5, '95%',  'Placement Rate',      '💼'),
    ]:
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
                f'padding:1.2rem .8rem;text-align:center;'
                f'box-shadow:0 2px 12px rgba(184,134,11,0.08);">'
                f'<div style="font-size:1.6rem;margin-bottom:.3rem;">{icon}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.6rem;'
                f'font-weight:700;color:#9a6e00;line-height:1;">{num}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:#9a7030;'
                f'letter-spacing:1px;text-transform:uppercase;margin-top:.3rem;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── PROJECT OVERVIEW + TECH STACK ────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:2.2rem;">✦ About the Project</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns([3, 2], gap='large')
    with pc1:
        st.markdown(
            '<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
            'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
            'padding:2rem;box-shadow:0 2px 16px rgba(184,134,11,0.08);">'
            '<div style="font-family:Cinzel,serif;font-size:.68rem;color:#b8860b;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:.8rem;">⚗ Project Overview</div>'
            '<div style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;'
            'color:#2c1a00;margin-bottom:.9rem;line-height:1.3;">'
            "NeuroScan AI — Early Parkinson's Detection via Deep Learning</div>"
            '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#5a3a0a;'
            'line-height:1.85;margin-bottom:1.2rem;">'
            'NeuroScan AI is a B.Tech final-year capstone project developed at the Department of ECE, '
            'BVC College of Engineering. It applies a hybrid '
            '<strong style="color:#8a5e00;">ResNet50 + Vision Transformer (ViT)</strong> and '
            '<strong style="color:#8a5e00;">EfficientNet-B4 + ViT</strong> cross-attention architecture '
            "to classify brain MRI scans and detect early signs of Parkinson's Disease with "
            'clinical-grade accuracy. The system integrates '
            '<strong style="color:#8a5e00;">Grad-CAM explainability</strong> '
            'to highlight regions of neural attention, bridging the gap between AI inference '
            'and medical interpretability. Validated with <strong style="color:#8a5e00;">5-fold '
            'cross-validation</strong> achieving 100% accuracy on ResNet50, EfficientNet-B4 and ViT-B16.'
            '</div>'
            '<div style="display:flex;gap:.7rem;flex-wrap:wrap;">'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">🧠 Deep Learning</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">🔬 Medical Imaging</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">📊 Explainable AI</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">⚡ Computer Vision</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">✦ 5-Fold CV Validated</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with pc2:
        st.markdown(
            '<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
            'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
            'padding:2rem;box-shadow:0 2px 16px rgba(184,134,11,0.08);">'
            '<div style="font-family:Cinzel,serif;font-size:.68rem;color:#b8860b;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:1rem;">⚙ Technology Stack</div>'
            '<div style="display:flex;flex-direction:column;gap:.7rem;">'

            '<div style="display:flex;align-items:center;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.3rem;">🤖</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">AI Framework</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'PyTorch · ResNet50 + ViT · EfficientNet+ViT</div></div></div>'

            '<div style="display:flex;align-items:center;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.3rem;">🔥</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Explainability</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'Gradient-weighted CAM (Grad-CAM)</div></div></div>'

            '<div style="display:flex;align-items:center;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.3rem;">📦</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Dataset</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'irfansheriff/parkinsons-brain-mri · 831 MRI scans</div></div></div>'

            '<div style="display:flex;align-items:center;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.3rem;">🖥</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">User Interface</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'Streamlit · Custom Royal CSS</div></div></div>'

            '<div style="display:flex;align-items:center;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.3rem;">📄</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Reporting</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'ReportLab PDF · Pandas · CSV Export</div></div></div>'

            '</div></div>',
            unsafe_allow_html=True,
        )

    # ── MODEL PERFORMANCE ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:2rem;">✦ Model Performance (5-Fold Cross-Validation)</div>', unsafe_allow_html=True)
    pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
    for col, val, lbl, color in [
        (pm1, '100%',   'ResNet50',      '#7a4f00'),
        (pm2, '100%',   'EfficientNet',  '#1a6a38'),
        (pm3, '99.88%', 'ViT-B16',       '#004a90'),
        (pm4, '98.4%',  'MiniSegNet',    '#7a1a1a'),
        (pm5, '98.4%',  'HybridNet RV',  '#5a1a7a'),
        (pm6, '99.2%',  'HybridNet EV',  '#1a5a7a'),
    ]:
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
                f'padding:1.2rem .8rem;text-align:center;">'
                f'<div style="font-family:Playfair Display,serif;font-size:1.5rem;'
                f'font-weight:900;color:{color};">{val}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:#9a7030;'
                f'letter-spacing:1px;text-transform:uppercase;margin-top:.3rem;">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── RESULT IMAGES FROM DRIVE ──────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:2rem;">✦ Training Results & Visualisations</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:EB Garamond,serif;font-size:1rem;color:#a89060;'
        'margin-bottom:1.2rem;line-height:1.7;">'
        'All result charts are loaded directly from Google Drive. '
        'These were generated during model training on the Parkinson\'s Brain MRI dataset.</p>',
        unsafe_allow_html=True,
    )

    # Drive direct-download URL helper
    def gdrive_img_url(file_id):
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # ── Replace these IDs with your actual file IDs from Drive ───────────────
    # To get a file ID: open the file in Drive → Share → Copy link
    # The ID is the long string between /d/ and /view in the URL
    # File order from Drive folder (alphabetical):
    # 1=confusion_matrices  2=eda_distribution  3=gradcam_all_models
    # 4=model_comparison    5=roc_curves        6=sample_grid   7=training_curves
    RESULT_IMAGES = {
        'Confusion Matrices':   '1qQgCIXUAvBFIqKOg31E2RKZqpMX3bfHn',
        'Class Distribution':   '1c10Rb-uQU8_Wtu59Q48P-E20IVfHDIwx',
        'GradCAM — All Models': '1uKot-ObwVs5VoqFXwxzWOUkEuV4pLr--',
        'Model Comparison':     '1pSapafZeP9r-gRs-dFsHOLgcJOddSFiz',
        'ROC Curves':           '1X78209LQ7PTwbu12lC8jWou9DqmF5RGp',
        'Sample Grid':          '1J6li-fQx28HqKuRclAlNHyUSzLOQ6Vvp',
        'Training Curves':      '13HIiDGEy8jEZgYis3vTzqctYtWlpKZ4N',
    }

    @st.cache_data(show_spinner=False)
    def fetch_drive_image(file_id):
        import requests
        # Try thumbnail URL first (bypasses virus-scan redirect for images)
        for url in [
            f"https://drive.google.com/thumbnail?id={file_id}&sz=w1200",
            f"https://lh3.googleusercontent.com/d/{file_id}",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
        ]:
            try:
                resp = requests.get(url, timeout=15, allow_redirects=True)
                ct   = resp.headers.get('Content-Type', '')
                if resp.status_code == 200 and 'image' in ct:
                    return resp.content
            except Exception:
                continue
        return None

    # Display in a 2-column grid
    img_items = list(RESULT_IMAGES.items())
    for i in range(0, len(img_items), 2):
        cols = st.columns(2, gap='large')
        for j, col in enumerate(cols):
            if i + j < len(img_items):
                name, fid = img_items[i + j]
                with col:
                    st.markdown(
                        f'<div class="card" style="padding:1rem;">'
                        f'<div style="font-family:Cinzel,serif;font-size:.65rem;'
                        f'color:#b8860b;letter-spacing:2px;text-transform:uppercase;'
                        f'margin-bottom:.8rem;text-align:center;">◈ {name}</div>',
                        unsafe_allow_html=True,
                    )
                    if 'REPLACE_WITH' in fid:
                        st.info(f'Add Drive file ID for: {name}')
                    else:
                        img_data = fetch_drive_image(fid)
                        if img_data:
                            st.image(img_data, use_column_width=True)
                        else:
                            st.warning(f'Could not load: {name}')
                    st.markdown('</div>', unsafe_allow_html=True)



    # ── TEAM ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;">✦ Project Team</div>', unsafe_allow_html=True)
    team = [
        {'roll': '236M5A0408', 'name': 'G Srinivasu',      'icon': '👨‍💻', 'role': 'AI Model & Backend'},
        {'roll': '226M1A0460', 'name': 'S Anusha Devi',    'icon': '👩‍💻', 'role': 'UI/UX & Frontend'},
        {'roll': '226M1A0473', 'name': 'V V Siva Vardhan', 'icon': '👨‍💻', 'role': 'Data & Testing'},
        {'roll': '236M5A0415', 'name': 'N L Sandeep',      'icon': '👨‍💻', 'role': 'Reports & Integration'},
    ]
    tcols = st.columns(4)
    for i, m in enumerate(team):
        with tcols[i]:
            st.markdown(
                f'<div class="team-card" style="position:relative;overflow:hidden;">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
                f'background:linear-gradient(90deg,transparent,#c9a84c,transparent);"></div>'
                f'<div style="font-size:2.6rem;margin-bottom:.6rem;">{m["icon"]}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.05rem;'
                f'font-weight:700;color:#2c1a00;margin-bottom:.3rem;">{m["name"]}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.88rem;'
                f'color:#8a5e00;margin-bottom:.4rem;font-style:italic;">{m["role"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;'
                f'color:#b89040;letter-spacing:1.5px;">{m["roll"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── GUIDANCE ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;">✦ Project Guidance</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3, gap='medium')
    for col, role_label, role_color, icon, name, tags in [
        (g1, 'Project Guide',       '#b8860b', '👨‍🏫',
         'Ms. N P U V S N Pavan Kumar, M.Tech',
         ['Assistant Professor', 'Dept. of ECE', 'Deputy CoE – III']),
        (g2, 'Project Coordinator', '#1a6a38', '📋',
         'Mr. K Anji Babu, M.Tech',
         ['Assistant Professor', 'Dept. of ECE']),
        (g3, 'Head of Department',  '#7a1a1a', '👨‍💼',
         'Dr. S A Vara Prasad, Ph.D, M.Tech',
         ['Professor & HOD · ECE', 'Chairman BoS', 'Anti-Ragging Committee']),
    ]:
        tag_html = ''.join(
            f'<span style="background:rgba(0,0,0,0.04);border:1px solid rgba(0,0,0,0.12);'
            f'border-radius:4px;padding:.2rem .75rem;font-family:Cinzel,serif;font-size:.6rem;'
            f'color:{role_color};letter-spacing:1px;font-weight:600;">{t}</span> '
            for t in tags
        )
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.2);border-radius:14px;'
                f'padding:2rem 1.6rem;text-align:center;position:relative;'
                f'box-shadow:0 3px 18px rgba(184,134,11,0.08);height:100%;">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
                f'background:linear-gradient(90deg,transparent,{role_color},transparent);'
                f'border-radius:14px 14px 0 0;"></div>'
                f'<div style="font-family:Cinzel,serif;font-size:.6rem;color:{role_color};'
                f'letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.9rem;'
                f'font-weight:700;">⭐ {role_label}</div>'
                f'<div style="font-size:2.5rem;margin-bottom:.7rem;">{icon}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1rem;'
                f'font-weight:700;color:#2c1a00;margin-bottom:1rem;line-height:1.4;">'
                f'{name}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center;">'
                f'{tag_html}</div></div>',
                unsafe_allow_html=True,
            )

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(192,57,43,0.05),'
        'rgba(192,57,43,0.02));border:1px solid rgba(192,57,43,0.2);'
        'border-left:4px solid #c0392b;border-radius:10px;padding:1.2rem 1.8rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#c0392b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">⚕ Medical Disclaimer</div>'
        '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a1a1a;line-height:1.7;">'
        'This project is developed <strong>for academic and research purposes only</strong>. '
        'The AI system is not a certified medical device and its outputs should never be used '
        'as a substitute for professional clinical diagnosis. '
        'Always consult a qualified neurologist for any medical decisions.'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ── ROYAL FOOTER ──────────────────────────────────────────────────────────────
st.markdown(
    '<div style="height:1px;background:linear-gradient(90deg,transparent,'
    'rgba(184,134,11,0.3),transparent);margin:2rem 0 0;"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="text-align:center;padding:1.6rem;'
    'background:linear-gradient(180deg,#fdf8f0,#fdf3dc);">'
    '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#9a7030;'
    'letter-spacing:3px;text-transform:uppercase;margin-bottom:.5rem;">'
    'Research &amp; Educational Use Only · Not for Clinical Diagnosis'
    '</div>'
    '<div style="display:flex;align-items:center;gap:1rem;justify-content:center;margin-bottom:.5rem;">'
    '<div style="flex:1;max-width:80px;height:1px;background:linear-gradient(90deg,transparent,'
    'rgba(201,168,76,0.2));"></div>'
    '<span style="color:rgba(201,168,76,0.3);font-size:.6rem;">✦</span>'
    '<div style="flex:1;max-width:80px;height:1px;background:linear-gradient(90deg,'
    'rgba(201,168,76,0.2),transparent);"></div>'
    '</div>'
    '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#9a7030;">'
    'NeuroScan AI · ResNet50+ViT · EfficientNet+ViT · Grad-CAM · PyTorch · Streamlit'
    '</div></div>',
    unsafe_allow_html=True,
)
