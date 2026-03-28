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
#  CONFIG  — 3 models: HybridNet_RV · HybridNet_EV · ViT-B16
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DRIVE_IDS = {
    "hybridnet_rv_best.pth": "1D_HT-DCMvmqTLVxnjG3y814Q9qH05sZs",
    "hybridnet_ev_best.pth": "1uXL90WvSm7akZbgpYZCinGAzLUhwhl6i",
    "vit-b16_best.pth":      "1D_Kit-vFC8PBPYFk3kG4ZVBcmNzVkJBw",
}

MODEL_FILES = {
    "HybridNet_RV (ResNet50 + ViT)":     "hybridnet_rv_best.pth",
    "HybridNet_EV (EfficientNet + ViT)": "hybridnet_ev_best.pth",
    "ViT-B16  —  Vision Transformer":    "vit-b16_best.pth",
}

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES — HybridNet_RV · HybridNet_EV · ViT-B16
# ══════════════════════════════════════════════════════════════════════════════

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
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ck = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    if isinstance(ck, dict) and 'model_state_dict' in ck:
        state_dict = ck['model_state_dict']
        nc         = ck.get('config', {}).get('num_classes', 2)
        mn         = ck.get('model_name', 'ResNet50')
        bt         = ck.get('backbone_type', None)
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
        classes = ['Normal', "Parkinson\'s Disease"]
    else:
        st.error("Unrecognised checkpoint format.")
        st.stop()

    if mn in ('HybridNet_RV', 'HybridNet_EV'):
        backbone = bt or ('resnet50' if 'RV' in mn else 'efficientnet_b4')
        m = HybridNet(nc, backbone_type=backbone)
    elif mn == 'ViT-B16':
        m = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=nc)
    else:
        st.error(f"Unsupported model: {mn}. This app uses HybridNet_RV, HybridNet_EV and ViT-B16 only.")
        st.stop()

    try:
        m.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
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
    try:
        if mn in ('HybridNet_RV', 'HybridNet_EV'):
            return model.get_gradcam_layer()
    except Exception:
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
            overlay  = None
            heatmap  = None

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
        '_probs_np':      probs_np,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
def build_pdf(patient, result):
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=letter,
                               leftMargin=0.75*inch, rightMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
    story  = []
    styles = getSampleStyleSheet()

    GOLD        = colors.HexColor('#b8860b')
    GOLD_DARK   = colors.HexColor('#7a4f00')
    GOLD_LIGHT  = colors.HexColor('#fdf3dc')
    PARCHMENT   = colors.HexColor('#fdf8f0')
    DEEP        = colors.HexColor('#2c1a00')
    MIDGOLD     = colors.HexColor('#c9a84c')
    CRIMSON     = colors.HexColor('#c0392b')
    EMERALD     = colors.HexColor('#1a6a38')
    GREY_TEXT   = colors.HexColor('#7a5a1a')

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

    def gold_line():
        t = Table([['', '', '']], colWidths=[0.5*inch, 6.5*inch, 0.5*inch])
        t.setStyle(TableStyle([
            ('LINEABOVE',  (1,0),(1,0), 1, MIDGOLD),
            ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0),(-1,-1), 0),
            ('BOTTOMPADDING',(0,0),(-1,-1), 4),
        ]))
        return t

    def gold_tbl(rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND',     (0,0),(-1,0), GOLD_DARK),
            ('TEXTCOLOR',      (0,0),(-1,0), colors.white),
            ('FONTNAME',       (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',       (0,0),(-1,0), 10),
            ('BOTTOMPADDING',  (0,0),(-1,0), 8),
            ('TOPPADDING',     (0,0),(-1,0), 8),
            ('ROWBACKGROUNDS', (0,1),(-1,-1), [GOLD_LIGHT, PARCHMENT]),
            ('GRID',           (0,0),(-1,-1), 0.4, MIDGOLD),
            ('FONTNAME',       (0,1),(-1,-1), 'Helvetica'),
            ('FONTSIZE',       (0,1),(-1,-1), 10),
            ('TEXTCOLOR',      (0,1),(-1,-1), DEEP),
            ('TOPPADDING',     (0,1),(-1,-1), 6),
            ('BOTTOMPADDING',  (0,1),(-1,-1), 6),
            ('LEFTPADDING',    (0,0),(-1,-1), 10),
            ('FONTNAME',       (0,1),(0,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR',      (0,1),(0,-1), GOLD_DARK),
        ]))
        return t

    story.append(Paragraph('NeuroScan AI', title_s))
    story.append(Paragraph("Parkinson's Disease MRI Analysis Report", subtitle_s))
    story.append(Paragraph(
        'PARKINSONS DISEASE DETECTION USING DEEP LEARNING ON BRAIN MRI',
        project_s))
    story.append(gold_line())
    story.append(Spacer(1, 4))

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
        f"NeuroScan AI | Parkinson's Disease Detection | BVC College of Engineering, Rajahmundry | "
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

# ── Session state MUST be initialised before any widget ──────────────────────
for _k, _v in [
    ('prediction_made',      False),
    ('patient_data',         {}),
    ('prediction_result',    {}),
    ('batch_results',        []),
    ('last_model',           ''),
    ('show_project_info',    False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
#  ROYAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,900;1,400;1,600&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Cinzel:wght@400;500;600;700;900&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');
:root {
  --gold:#b8860b; --gold-bright:#9a6e00; --gold-dim:#c9a84c;
  --gold-muted:rgba(184,134,11,0.10); --gold-glow:rgba(184,134,11,0.25);
  --gold-line:rgba(184,134,11,0.20); --gold-border:rgba(184,134,11,0.40);
  --crimson:#c0392b; --crimson-dim:rgba(192,57,43,0.10);
  --emerald:#1e8449; --emerald-dim:rgba(30,132,73,0.10);
  --text:#2c1a00; --text-2:#7a5a1a; --text-3:#b89040;
  --radius:12px; --radius-sm:7px;
  --shadow-royal:0 4px 24px rgba(184,134,11,0.10),0 1px 6px rgba(0,0,0,0.06);
  --shadow-gold:0 4px 20px rgba(184,134,11,0.18);
}

/* ── Base ── */
html,body,.stApp{background:#fdf8f0!important;font-family:'EB Garamond','Cormorant Garamond',serif!important;color:var(--text)!important;}
.stApp::before{content:'';position:fixed;inset:0;z-index:0;
  background-image:radial-gradient(ellipse 80% 50% at 50% 0%,rgba(184,134,11,0.06) 0%,transparent 60%),
  repeating-linear-gradient(45deg,transparent,transparent 34px,rgba(184,134,11,0.025) 34px,rgba(184,134,11,0.025) 35px),
  repeating-linear-gradient(-45deg,transparent,transparent 34px,rgba(184,134,11,0.025) 34px,rgba(184,134,11,0.025) 35px);
  pointer-events:none;}
.block-container{padding-top:0!important;max-width:1380px!important;position:relative;z-index:1;}
#MainMenu,footer,header{visibility:hidden;}

/* ── Cards ── */
.card{background:linear-gradient(145deg,#fffdf8 0%,#fff8ec 100%);border:1px solid var(--gold-line);border-radius:var(--radius);padding:1.8rem 2rem;margin:.7rem 0;box-shadow:var(--shadow-royal);position:relative;transition:border-color .3s,box-shadow .3s;}
.card::before{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:60%;height:1px;background:linear-gradient(90deg,transparent,var(--gold-dim),transparent);}
.card:hover{border-color:var(--gold-border);box-shadow:var(--shadow-royal),0 0 30px rgba(201,168,76,0.08);}

/* ── Section headings ── */
.sec-head{display:flex;align-items:center;gap:.9rem;font-family:'Cinzel',serif;font-size:.68rem;font-weight:600;color:var(--gold);letter-spacing:3.5px;text-transform:uppercase;margin:2rem 0 1rem;}
.sec-head::before{content:'';flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--gold-dim));}
.sec-head::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--gold-dim),transparent);}

/* ── Inputs ── */
.stTextInput input,.stNumberInput input,.stTextArea textarea,.stDateInput input{background:#fffdf8!important;color:var(--text)!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius-sm)!important;font-family:'EB Garamond',serif!important;font-size:1rem!important;transition:border-color .25s,box-shadow .25s!important;}
.stTextInput input:focus,.stNumberInput input:focus,.stTextArea textarea:focus{border-color:var(--gold)!important;box-shadow:0 0 0 3px rgba(201,168,76,0.12)!important;outline:none!important;}
div[data-baseweb="select"]>div{background:#fffdf8!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius-sm)!important;color:var(--text)!important;font-family:'EB Garamond',serif!important;}
label{color:var(--text-2)!important;font-family:'Cinzel',serif!important;font-size:.68rem!important;font-weight:600!important;letter-spacing:1.5px!important;text-transform:uppercase!important;}

/* ── Buttons ── */
.stButton>button{background:linear-gradient(135deg,#8a6e2f 0%,#c9a84c 40%,#e8c96a 60%,#c9a84c 80%,#8a6e2f 100%)!important;color:#07090f!important;border:1px solid rgba(232,201,106,0.5)!important;border-radius:var(--radius-sm)!important;padding:.75rem 1.6rem!important;font-family:'Cinzel',serif!important;font-size:.78rem!important;font-weight:700!important;letter-spacing:2.5px!important;text-transform:uppercase!important;width:100%!important;transition:all .3s ease!important;box-shadow:0 3px 16px rgba(201,168,76,0.3),inset 0 1px 0 rgba(255,255,255,0.2)!important;}
.stButton>button:hover{background:linear-gradient(135deg,#9a7e3f 0%,#d9b85c 40%,#f8d97a 60%,#d9b85c 80%,#9a7e3f 100%)!important;transform:translateY(-2px)!important;box-shadow:0 6px 24px rgba(201,168,76,0.45)!important;}
.stButton>button:active{transform:translateY(0)!important;}
.stDownloadButton>button{background:linear-gradient(135deg,#0a2e18 0%,#1a6a38 40%,#2a8a50 60%,#1a6a38 80%,#0a2e18 100%)!important;color:#c9e8c8!important;border:1px solid rgba(74,222,128,0.3)!important;border-radius:var(--radius-sm)!important;padding:.75rem 1.6rem!important;font-family:'Cinzel',serif!important;font-size:.78rem!important;font-weight:700!important;letter-spacing:2.5px!important;text-transform:uppercase!important;width:100%!important;transition:all .3s ease!important;box-shadow:0 3px 16px rgba(30,132,73,0.3)!important;}
.stDownloadButton>button:hover{background:linear-gradient(135deg,#0d3a1e 0%,#1f7a42 40%,#32a060 60%,#1f7a42 80%,#0d3a1e 100%)!important;transform:translateY(-2px)!important;}

/* ── File uploader ── */
[data-testid="stFileUploader"]{background:#fffdf8!important;border:2px dashed var(--gold-line)!important;border-radius:var(--radius)!important;padding:1.6rem!important;transition:border-color .3s,box-shadow .3s!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--gold)!important;box-shadow:0 0 24px var(--gold-muted)!important;}

/* ── Diagnosis badges ── */
.diag-badge{display:inline-flex;align-items:center;gap:.5rem;font-family:'Cinzel',serif;font-size:1.3rem;font-weight:700;padding:.5rem 1.2rem;border-radius:4px;letter-spacing:1px;}
.diag-normal{background:var(--emerald-dim);color:#4ade80;border:1px solid rgba(74,222,128,0.35);}
.diag-parkinson{background:var(--crimson-dim);color:#f87171;border:1px solid rgba(248,113,113,0.35);}
.stat-tile{background:linear-gradient(145deg,#fffdf8,#fff6e8);border:1px solid var(--gold-line);border-radius:var(--radius);padding:1.1rem 1.3rem;text-align:center;box-shadow:var(--shadow-royal);}
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
.img-caption{font-family:'Cinzel',serif;font-size:.6rem;color:#b89040;text-align:center;padding:.45rem 0 .25rem;letter-spacing:1.5px;text-transform:uppercase;}
.hm-legend{display:flex;align-items:center;gap:.6rem;font-family:'Cinzel',serif;font-size:.6rem;color:#b89040;margin-top:.6rem;letter-spacing:1px;}
.hm-bar{flex:1;height:6px;border-radius:3px;background:linear-gradient(90deg,#00008b,#0000ff,#00ff00,#ffff00,#ff0000);}

/* ── TABS — fixed alignment & style ── */
.stTabs{margin-top:.5rem;}
.stTabs [data-baseweb="tab-list"]{
  background:linear-gradient(135deg,#fdf3dc,#fffdf8)!important;
  border:1.5px solid rgba(184,134,11,0.3)!important;
  border-radius:12px!important;
  padding:.4rem .5rem!important;
  gap:.3rem!important;
  justify-content:center!important;
  align-items:center!important;
  overflow:visible!important;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Cinzel',serif!important;
  font-size:.72rem!important;
  font-weight:700!important;
  color:#7a5a1a!important;
  border-radius:8px!important;
  padding:.65rem 1.6rem!important;
  letter-spacing:1.5px!important;
  transition:all .25s!important;
  white-space:nowrap!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  gap:.4rem!important;
  min-height:2.4rem!important;
}
.stTabs [data-baseweb="tab"]:hover{
  color:#7a4f00!important;
  background:rgba(184,134,11,0.08)!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,rgba(184,134,11,0.18),rgba(184,134,11,0.10))!important;
  color:#5a3000!important;
  border:1px solid rgba(184,134,11,0.4)!important;
  box-shadow:0 2px 8px rgba(184,134,11,0.15)!important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"]{display:none!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.2rem!important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#fdf8f0,#fdf3e7)!important;border-right:1.5px solid var(--gold-line)!important;}
[data-testid="stSidebar"] *{font-family:'EB Garamond',serif!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-family:'Cinzel',serif!important;color:#9a6e00!important;letter-spacing:2px!important;}

/* ── Sidebar toggle button — gold circle, no emoji tricks ── */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
  background:linear-gradient(135deg,#8a6e2f,#c9a84c,#e8c96a,#c9a84c,#8a6e2f)!important;
  border:2px solid rgba(232,201,106,0.9)!important;
  border-radius:12px!important;
  width:44px!important; height:44px!important;
  min-width:44px!important; min-height:44px!important;
  box-shadow:0 4px 16px rgba(184,134,11,0.45),inset 0 1px 0 rgba(255,255,255,0.25)!important;
  transition:all .3s ease!important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover{
  transform:scale(1.1)!important;
  box-shadow:0 6px 24px rgba(184,134,11,0.65)!important;
}
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="collapsedControl"] button svg{
  fill:#3a2000!important;
  stroke:#3a2000!important;
  width:20px!important;
  height:20px!important;
}

/* ── Sidebar metrics + radio ── */
[data-testid="stMetric"]{background:linear-gradient(145deg,#fffdf8,#fff6e8)!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius)!important;padding:1rem 1.2rem!important;}
[data-testid="stMetricValue"]{font-family:'Playfair Display',serif!important;font-size:1.6rem!important;font-weight:700!important;color:#9a6e00!important;}
[data-testid="stMetricLabel"]{color:#7a5a1a!important;font-family:'Cinzel',serif!important;font-size:.58rem!important;letter-spacing:1.5px!important;text-transform:uppercase!important;}
.stProgress>div>div>div{background:linear-gradient(90deg,var(--gold-dim),var(--gold-bright))!important;border-radius:3px!important;}
[data-testid="stDataFrame"]{border-radius:var(--radius)!important;border:1px solid var(--gold-line)!important;overflow:hidden!important;}
.stRadio>label{font-family:'Cinzel',serif!important;font-size:.72rem!important;font-weight:600!important;color:#7a4f00!important;letter-spacing:1px!important;}
.stRadio [data-baseweb="radio"] div[role="radio"]{border-color:var(--gold-dim)!important;}
.stRadio [data-baseweb="radio"] div[aria-checked="true"]{background:var(--gold-dim)!important;border-color:var(--gold)!important;}

/* ── Misc ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#fdf8f0;}
::-webkit-scrollbar-thumb{background:#c9a84c;border-radius:10px;}
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,var(--gold-line),transparent)!important;margin:1.8rem 0!important;}
.stAlert{border-radius:var(--radius)!important;font-family:'EB Garamond',serif!important;}
.team-card{background:linear-gradient(145deg,#fffdf8,#fff6e8);border:1px solid var(--gold-line);border-radius:var(--radius);padding:1.4rem 1rem;text-align:center;transition:border-color .3s,transform .3s;}
.team-card:hover{border-color:var(--gold-border);transform:translateY(-4px);box-shadow:var(--shadow-gold);}
.model-selected{background:linear-gradient(135deg,rgba(184,134,11,0.15),rgba(184,134,11,0.08));border:1px solid rgba(184,134,11,0.4);border-left:4px solid #c9a84c;border-radius:8px;padding:.6rem .9rem;margin:.3rem 0;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ABOUT PROJECT  — defined at module level (not conditionally)
# ══════════════════════════════════════════════════════════════════════════════
@st.dialog("📋 About the Project", width="large")
def _show_project_dialog():
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.68rem;color:#b8860b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.5rem;">'
        '⚗ B.Tech Final Year Project · 2025-26 · Dept. of ECE · BVC College of Engineering</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:Playfair Display,serif;font-size:1.35rem;font-weight:700;'
        'color:#2c1a00;margin-bottom:.9rem;line-height:1.35;">'
        "Parkinson's Disease Detection Using Deep Learning on Brain MRI"
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#5a3a0a;'
        'line-height:1.85;margin-bottom:1.1rem;">'
        'This project applies <strong style="color:#7a4f00;">Hybrid Deep Learning</strong> to classify '
        'brain MRI scans for early Parkinson\'s detection. Six models are trained on '
        '<em>irfansheriff/parkinsons-brain-mri-dataset</em> (831 scans — 610 Normal, 221 Parkinson): '
        '<strong style="color:#7a4f00;">ResNet50</strong>, <strong style="color:#7a4f00;">EfficientNet-B4</strong>, '
        '<strong style="color:#7a4f00;">ViT-B/16</strong>, <strong style="color:#7a4f00;">MiniSegNet</strong>, '
        '<strong style="color:#7a4f00;">HybridNet_RV</strong> (ResNet50+ViT Cross-Attention), '
        'and <strong style="color:#7a4f00;">HybridNet_EV</strong> (EfficientNet-B4+ViT Cross-Attention).'
        '</div>',
        unsafe_allow_html=True,
    )
    _d1, _d2 = st.columns(2, gap='medium')
    with _d1:
        for _t, _body in [
            ('📦 Dataset', '831 scans (Kaggle) · 610 Normal · 221 Parkinson<br>Split: 581 Train · 125 Val · 125 Test<br>WeightedRandomSampler for class balance'),
            ('⚙ Training Config', 'SEED=42 · IMG=224×224 · BS=16 · EPOCHS=30<br>MixUp (α=0.3) · Label Smoothing (ε=0.1)<br>CosineAnnealingLR · Early Stopping (p=7)<br>AMP mixed precision · Grad clip=1.0'),
        ]:
            st.markdown(
                f'<div style="background:rgba(184,134,11,0.06);border:1px solid rgba(184,134,11,0.18);'
                f'border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.7rem;">'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:#b8860b;'
                f'letter-spacing:2px;text-transform:uppercase;margin-bottom:.5rem;">{_t}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.92rem;color:#5a3a0a;line-height:1.7;">{_body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with _d2:
        for _t, _body in [
            ('📊 Test Results', 'ResNet50: <b>100%</b> · EfficientNet: <b>99.2%</b><br>ViT-B/16: <b>100%</b> · MiniSegNet: <b>97.6%</b><br>HybridNet_RV: <b>96%</b> · HybridNet_EV: <b>100%</b><br>Ensemble: <b style="color:#1a6a38;">100%</b>'),
            ('🖥 Tech Stack', 'PyTorch · timm · torchvision · scikit-learn<br>OpenCV · Seaborn · Streamlit · ReportLab<br>Google Colab · Tesla T4 GPU (15.6 GB VRAM)'),
        ]:
            st.markdown(
                f'<div style="background:rgba(184,134,11,0.06);border:1px solid rgba(184,134,11,0.18);'
                f'border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.7rem;">'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:#b8860b;'
                f'letter-spacing:2px;text-transform:uppercase;margin-bottom:.5rem;">{_t}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.92rem;color:#5a3a0a;line-height:1.7;">{_body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div style="background:rgba(30,132,73,0.07);border:1px solid rgba(30,132,73,0.22);'
        'border-left:4px solid #1a6a38;border-radius:8px;padding:.9rem 1.2rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.58rem;color:#1a6a38;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.35rem;">🏆 Key Innovation</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.95rem;color:#0a2e18;line-height:1.75;">'
        '<strong>Cross-Attention Fusion</strong> — bridges CNN spatial feature maps with ViT global '
        'patch tokens via multi-head cross-attention, combining local texture (CNN) and global context '
        '(ViT) for superior Parkinson\'s detection from brain MRI.'
        '</div></div>',
        unsafe_allow_html=True,
    )


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
    'border-bottom:1px solid rgba(184,134,11,0.22);padding:2.8rem 2rem 2.2rem;'
    'display:flex;flex-direction:column;align-items:center;gap:.9rem;'
    'position:relative;overflow:hidden;">'
    # Top gold bar
    '<div style="position:absolute;top:0;left:0;right:0;height:3px;'
    'background:linear-gradient(90deg,transparent,#c9a84c,#e8c96a,#c9a84c,transparent);"></div>'
    # Corner ornaments
    '<div style="position:absolute;top:14px;left:24px;font-size:.9rem;'
    'color:rgba(184,134,11,0.35);font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'
    '<div style="position:absolute;top:14px;right:24px;font-size:.9rem;'
    'color:rgba(184,134,11,0.35);font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'
    + logo_html +
    # Main title
    '<div style="font-family:Cinzel,serif;font-size:2.8rem;font-weight:900;color:#2c1a00;'
    'letter-spacing:8px;line-height:1;text-align:center;'
    'text-shadow:0 2px 12px rgba(184,134,11,0.15);">'
    'NEUROSCAN&nbsp;<span style="color:#b8860b;">AI</span></div>'
    # Subtitle
    '<div style="font-family:Cormorant Garamond,serif;font-size:1rem;font-style:italic;'
    'color:#9a7030;letter-spacing:4px;text-align:center;">'
    "Parkinson\u2019s Detection \u00b7 Brain MRI \u00b7 Deep Learning</div>"
    # Project title pill
    '<div style="background:linear-gradient(135deg,rgba(184,134,11,0.12),rgba(184,134,11,0.06));'
    'border:1px solid rgba(184,134,11,0.35);border-radius:8px;'
    'padding:.65rem 2rem;margin-top:.1rem;position:relative;overflow:hidden;">'
    '<div style="position:absolute;inset:0;background:linear-gradient(90deg,transparent,'
    'rgba(232,201,106,0.08),transparent);"></div>'
    '<div style="font-family:Cinzel,serif;font-size:.9rem;font-weight:700;'
    'color:#7a4f00;letter-spacing:2.5px;text-align:center;text-transform:uppercase;'
    'position:relative;z-index:1;">'
    "Parkinson\u2019s Disease Detection Using Deep Learning on Brain MRI"
    '</div></div>'
    # Divider
    '<div style="display:flex;align-items:center;gap:1rem;width:60%;max-width:380px;">'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,#c9a84c);"></div>'
    '<span style="color:#b8860b;font-size:.7rem;">✦</span>'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,#c9a84c,transparent);"></div>'
    '</div>'
    # Feature badges
    '<div style="display:flex;gap:.8rem;flex-wrap:wrap;justify-content:center;">'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:20px;padding:.28rem .9rem;font-family:Cinzel,serif;font-size:.62rem;'
    'color:#8a5e00;letter-spacing:1px;">⚙ ResNet50+ViT / EffNet+ViT</span>'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:20px;padding:.28rem .9rem;font-family:Cinzel,serif;font-size:.62rem;'
    'color:#8a5e00;letter-spacing:1px;">◈ 100% Ensemble Accuracy</span>'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:20px;padding:.28rem .9rem;font-family:Cinzel,serif;font-size:.62rem;'
    'color:#8a5e00;letter-spacing:1px;">✦ Grad-CAM XAI</span>'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:20px;padding:.28rem .9rem;font-family:Cinzel,serif;font-size:.62rem;'
    'color:#8a5e00;letter-spacing:1px;">⬡ Batch Analysis</span>'
    '</div>'
    # Guidance line
    '<div style="display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;justify-content:center;">'
    '<div style="height:1px;width:60px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.4));"></div>'
    '<div style="background:linear-gradient(135deg,rgba(184,134,11,0.10),rgba(184,134,11,0.05));'
    'border:1px solid rgba(184,134,11,0.28);border-radius:20px;padding:.3rem 1.2rem;">'
    '<span style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:.88rem;color:#9a7030;">'
    'Under the Esteemed Guidance of&nbsp;</span>'
    '<span style="font-family:Cinzel,serif;font-size:.85rem;font-weight:700;color:#7a4f00;">'
    'Mr. N P U V S N Pavan Kumar</span>'
    '<span style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:.82rem;color:#b8860b;">'
    '&nbsp;· Asst. Professor, ECE</span>'
    '</div>'
    '<div style="height:1px;width:60px;background:linear-gradient(90deg,rgba(184,134,11,0.4),transparent);"></div>'
    '</div>'
    # Bottom bar
    '<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
    'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── About Project button — centred below hero ─────────────────────────────────
_b1, _b2, _b3 = st.columns([4, 2, 4])
with _b2:
    if st.button('📋 About Project', key='hero_about_btn'):
        _show_project_dialog()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown(
        '<div style="background:linear-gradient(135deg,#fdf3dc,#fdf8f0);'
        'border-bottom:2px solid rgba(184,134,11,0.3);'
        'padding:1.2rem 1rem 1rem;margin:-1rem -1rem .8rem;text-align:center;">'
        '<div style="font-size:1.8rem;margin-bottom:.3rem;">🧠</div>'
        '<div style="font-family:Cinzel,serif;font-size:1rem;font-weight:900;'
        'color:#7a4f00;letter-spacing:4px;">⚕ NEUROSCAN AI</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.78rem;font-style:italic;'
        'color:#9a7030;margin-top:.2rem;">Model Control Panel</div>'
        '<div style="height:2px;background:linear-gradient(90deg,transparent,#c9a84c,transparent);'
        'margin:.6rem 0 0;"></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.62rem;color:#b8860b;'
        'letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.5rem;">⚙ Select Model</div>',
        unsafe_allow_html=True,
    )

    MODEL_DISPLAY = {
        "🔀 HybridNet RV  —  ResNet50 + ViT":      "HybridNet_RV (ResNet50 + ViT)",
        "🔀 HybridNet EV  —  EffNet + ViT":         "HybridNet_EV (EfficientNet + ViT)",
        "👁 ViT-B/16  —  Vision Transformer":        "ViT-B16  —  Vision Transformer",
        "✦ All 3 Models  —  Ensemble (recommended)": "__ALL__",
    }

    selected_display = st.radio(
        "model_radio",
        list(MODEL_DISPLAY.keys()),
        index=0,
        label_visibility="collapsed",
    )
    model_choice_key = MODEL_DISPLAY[selected_display]

    selected_acc = {
        "HybridNet_RV (ResNet50 + ViT)":    "98.4%",
        "HybridNet_EV (EfficientNet + ViT)":"99.2%",
        "ViT-B16  —  Vision Transformer":   "99.9%",
        "__ALL__":                           "99.8%",
    }
    acc        = selected_acc.get(model_choice_key, "—")
    mode_label = "All 3 Models + Ensemble" if model_choice_key == "__ALL__" else model_choice_key
    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(184,134,11,0.12),'
        f'rgba(184,134,11,0.06));border:1px solid rgba(184,134,11,0.4);'
        f'border-left:4px solid #c9a84c;border-radius:8px;padding:.6rem .9rem;margin:.5rem 0 .8rem;">'
        f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:#9a7030;'
        f'letter-spacing:1px;text-transform:uppercase;">Selected</div>'
        f'<div style="font-family:Playfair Display,serif;font-size:.92rem;'
        f'font-weight:700;color:#7a4f00;">{mode_label}</div>'
        f'<div style="font-family:Cinzel,serif;font-size:.6rem;color:#4CAF50;'
        f'margin-top:.2rem;">CV Accuracy: {acc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),transparent);margin:.5rem 0;"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.62rem;color:#b8860b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">◈ Performance</div>',
        unsafe_allow_html=True,
    )
    _c1, _c2 = st.columns(2)
    with _c1: st.metric('CV Acc',    '100%')
    with _c2: st.metric('AUC',       '1.000')
    _c3, _c4 = st.columns(2)
    with _c3: st.metric('Precision', '100%')
    with _c4: st.metric('Recall',    '98%')

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),transparent);margin:.5rem 0;"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.62rem;color:#b8860b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">✦ Features</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.88rem;color:#6b4c11;line-height:2;">'
        '⚜ Single-scan analysis<br>'
        '⚜ Batch processing<br>'
        '⚜ Grad-CAM heatmap<br>'
        '⚜ Risk scoring<br>'
        '⚜ Gold PDF report<br>'
        '⚜ CSV export'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),transparent);margin:.5rem 0;"></div>', unsafe_allow_html=True)
    st.warning("⚠️ Research/academic use only.")
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.55rem;color:#9a7030;'
        'letter-spacing:1px;text-align:center;margin-top:.4rem;">'
        '💡 One model loaded at a time — memory safe'
        '</div>',
        unsafe_allow_html=True,
    )

MODEL_PATH     = MODEL_FILES.get(model_choice_key, list(MODEL_FILES.values())[0])
RUN_ALL_MODELS = (model_choice_key == "__ALL__")


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_scan, tab_batch, tab_about = st.tabs([
    '🧠  Single MRI Analysis',
    '📦  Batch Analysis',
    '👥  About Us',
])

with tab_scan:

    _run_mode = "All 3 Models + Ensemble" if RUN_ALL_MODELS else model_choice_key
    st.markdown(
        f'<div class="sec-head">✦ Mode: {_run_mode}</div>',
        unsafe_allow_html=True
    )
    _MODEL_INFO = {
        "HybridNet_RV (ResNet50 + ViT)":     {"icon":"🔀","short":"HybridNet RV","acc":"98.4%","desc":"ResNet50 + ViT"},
        "HybridNet_EV (EfficientNet + ViT)": {"icon":"🔀","short":"HybridNet EV","acc":"99.2%","desc":"EfficientNet + ViT"},
        "ViT-B16  —  Vision Transformer":    {"icon":"👁","short":"ViT-B/16",    "acc":"99.9%","desc":"Pure Transformer"},
    }
    _mc_cols = st.columns(3, gap="large")
    for _ci, (_mkey, _minfo) in enumerate(_MODEL_INFO.items()):
        _is_sel     = RUN_ALL_MODELS or (model_choice_key == _mkey)
        _top_bdr    = "3px solid #c9a84c" if _is_sel else "3px solid rgba(184,134,11,0.1)"
        _bg_card    = "linear-gradient(145deg,#fdf3dc,#fceedd)" if _is_sel else "linear-gradient(145deg,#fffdf8,#fff6e8)"
        _shadow_c   = "0 4px 16px rgba(184,134,11,0.22)" if _is_sel else "0 2px 8px rgba(184,134,11,0.06)"
        _tick       = " ✦" if _is_sel else ""
        with _mc_cols[_ci]:
            st.markdown(
                f'<div style="background:{_bg_card};border:1px solid rgba(184,134,11,0.3);'
                f'border-radius:10px;padding:.8rem .4rem;text-align:center;'
                f'border-top:{_top_bdr};box-shadow:{_shadow_c};">'
                f'<div style="font-size:1.4rem;margin-bottom:.25rem;">{_minfo["icon"]}{_tick}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.6rem;font-weight:700;'
                f'color:#7a4f00;letter-spacing:.5px;margin-bottom:.15rem;">{_minfo["short"]}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.74rem;'
                f'color:#9a7030;margin-bottom:.2rem;">{_minfo["desc"]}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:.88rem;'
                f'font-weight:700;color:#4CAF50;">{_minfo["acc"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.48rem;color:#b89040;'
                f'letter-spacing:1px;margin-top:.2rem;">5-FOLD CV</div>'
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
                _spinner_txt = 'Running all 6 models + ensemble…' if RUN_ALL_MODELS else f'Running {model_choice_key}…'
                with st.spinner(_spinner_txt):
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
                        all_model_results = {}
                        all_probs_list    = []
                        ensemble_weights  = [1.2, 1.2, 0.8]
                        _cls_list  = None
                        _first_img = None

                        if RUN_ALL_MODELS:
                            _models_to_run = list(MODEL_FILES.items())
                        else:
                            _single_path = MODEL_FILES.get(model_choice_key,
                                           list(MODEL_FILES.values())[0])
                            _models_to_run = [(model_choice_key, _single_path)]

                        _prog = st.progress(0)
                        _stat = st.empty()

                        for _idx_m, (_mc, _mpath) in enumerate(_models_to_run):
                            _stat.markdown(
                                f'<p style="font-family:Cinzel,serif;font-size:.75rem;'
                                f'color:#c9a84c;">Running {_mc} ({_idx_m+1}/'
                                f'{len(MODEL_FILES)})…</p>',
                                unsafe_allow_html=True
                            )
                            download_model(_mpath)
                            _m, _cls, _mn, _dev = load_model(_mpath)
                            _cls_list = _cls
                            _r = predict(_m, _dev, _cls, _mn, image)
                            if _first_img is None:
                                _first_img = _r['image']
                            all_model_results[_mc] = _r
                            all_probs_list.append(_r['_probs_np'])
                            del _m
                            import gc as _gc
                            _gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            _prog.progress((_idx_m + 1) / max(len(_models_to_run), 1))

                        _stat.empty()
                        _prog.empty()

                        if RUN_ALL_MODELS and len(all_probs_list) == 3:
                            _ens_weights = ensemble_weights
                        else:
                            _ens_weights = [1.0] * len(all_probs_list)
                        _ens_p    = np.average(np.stack(all_probs_list),
                                               axis=0, weights=_ens_weights)
                        _ens_idx  = int(_ens_p.argmax())
                        _ens_conf = float(_ens_p[_ens_idx] * 100)
                        _park_idx = next(
                            (i for i, c in enumerate(_cls_list) if 'parkinson' in c.lower()), 1)
                        _norm_idx = 1 - _park_idx if len(_cls_list) > 1 else 0
                        _ens_is_pk   = (_ens_idx == _park_idx)
                        _ens_label   = _cls_list[_ens_idx]
                        _ens_norm_p  = float(_ens_p[_norm_idx] * 100)
                        _ens_park_p  = float(_ens_p[_park_idx] * 100)
                        _ens_risk    = ('High' if _ens_conf >= 85 else 'Moderate') if _ens_is_pk else 'Low'

                        ens_result = {
                            'prediction':   _ens_label,
                            'class_idx':    _ens_idx,
                            'is_parkinson': _ens_is_pk,
                            'confidence':   _ens_conf,
                            'normal_prob':  _ens_norm_p,
                            'parkinson_prob': _ens_park_p,
                            'risk_level':   _ens_risk,
                            'cam_overlay':  None,
                            'cam_heatmap':  None,
                            'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'image':        _first_img,
                            'model_name':   'Ensemble (6 models)' if RUN_ALL_MODELS else model_choice_key,
                        }

                        _lite_results = {}
                        for _k, _v in all_model_results.items():
                            _lite_results[_k] = {
                                kk: vv for kk, vv in _v.items()
                                if kk not in ('_probs_np',)
                            }

                        st.session_state.all_model_results = _lite_results
                        st.session_state.ensemble_result   = ens_result
                        st.session_state.prediction_result = ens_result
                        st.session_state.prediction_made   = True
                        _done_msg = 'All 3 models analysed!' if RUN_ALL_MODELS else f'{model_choice_key} analysis complete!'
                        st.success(_done_msg)
                        st.balloons()
                        st.rerun()
                    except Exception as _exc:
                        st.error(f'Error during analysis: {_exc}')

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

            del model
            import gc as _gc; _gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

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

with tab_about:
    college_logo = get_logo_b64('bvcr.jpg')
    if college_logo:
        clg_html = (
            '<img src="' + college_logo + '" style="width:110px;height:110px;object-fit:contain;'
            'margin-bottom:1.2rem;border:2px solid rgba(184,134,11,0.45);border-radius:10px;'
            'box-shadow:0 6px 24px rgba(184,134,11,0.18);"/>'
        )
    else:
        clg_html = '<div style="font-size:4rem;margin-bottom:1.2rem;">🏛</div>'

    st.markdown(
        '<div style="background:linear-gradient(160deg,#fdf3dc 0%,#fdf8f0 50%,#fce8b0 100%);'
        'border:1px solid rgba(184,134,11,0.28);border-radius:16px;'
        'padding:3rem 2.5rem 2.5rem;margin-bottom:2rem;text-align:center;'
        'position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        'background:linear-gradient(90deg,transparent,#c9a84c,#f0d070,#c9a84c,transparent);"></div>'
        + clg_html +
        '<div style="font-family:Cinzel,serif;font-size:1.9rem;font-weight:900;color:#2c1a00;'
        'letter-spacing:4px;margin-bottom:.4rem;">BVC College of Engineering</div>'
        '<div style="font-family:Cormorant Garamond,serif;font-style:italic;'
        'font-size:1.05rem;color:#b8860b;letter-spacing:3px;margin-bottom:.3rem;">'
        'Rajahmundry, Andhra Pradesh</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#9a7030;margin-bottom:1rem;">'
        '<a href="https://bvcr.edu.in" target="_blank" '
        'style="color:#b8860b;text-decoration:none;border-bottom:1px solid rgba(184,134,11,0.4);">'
        'bvcr.edu.in</a></div>'
        '<div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.2rem;">'
        '<span style="background:rgba(184,134,11,0.1);border:1px solid rgba(184,134,11,0.3);'
        'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        'color:#7a4f00;letter-spacing:1.5px;font-weight:600;">🎓 Autonomous</span>'
        '<span style="background:rgba(30,132,73,0.08);border:1px solid rgba(30,132,73,0.25);'
        'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        'color:#1a6a38;letter-spacing:1.5px;font-weight:600;">✓ NAAC A Grade</span>'
        '<span style="background:rgba(0,100,180,0.07);border:1px solid rgba(0,100,180,0.2);'
        'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        'color:#004a90;letter-spacing:1.5px;font-weight:600;">⚙ AICTE Approved</span>'
        '<span style="background:rgba(140,40,40,0.07);border:1px solid rgba(140,40,40,0.2);'
        'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        'color:#7a1a1a;letter-spacing:1.5px;font-weight:600;">📜 Affiliated to JNTUK</span>'
        '</div>'
        '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a5a1a;'
        'line-height:1.8;max-width:700px;margin:0 auto;">'
        'BVC College of Engineering, Rajahmundry is a premier autonomous institution '
        'in Andhra Pradesh permanently affiliated to JNTUK. '
        'Accredited by NAAC with A Grade and approved by AICTE, recognized for academic '
        'excellence, state-of-the-art infrastructure, and outstanding placements.'
        '</div>'
        '<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
        'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    #  ABOUT THE PROJECT  — based on training notebook (FinaleAll.ipynb)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head" style="margin-top:1rem;">✦ About the Project</div>', unsafe_allow_html=True)
    _pc1, _pc2 = st.columns([3, 2], gap='large')
    with _pc1:
        st.markdown(
            '<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
            'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
            'padding:2rem;box-shadow:0 2px 16px rgba(184,134,11,0.08);">'
            '<div style="font-family:Cinzel,serif;font-size:.68rem;color:#b8860b;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:.8rem;">⚗ Project Overview</div>'
            '<div style="font-family:Playfair Display,serif;font-size:1.35rem;font-weight:700;'
            'color:#2c1a00;margin-bottom:.9rem;line-height:1.3;">'
            "Parkinson's Disease Detection Using Deep Learning on Brain MRI</div>"
            '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#5a3a0a;'
            'line-height:1.9;margin-bottom:1.2rem;">'
            'This B.Tech Final Year Project (2025-26) at the Dept. of ECE, '
            'BVC College of Engineering, Rajahmundry applies '
            '<strong style="color:#8a5e00;">Hybrid Deep Learning</strong> architectures '
            "to classify brain MRI scans for early detection of Parkinson's Disease. "
            'The system uses the <em>irfansheriff/parkinsons-brain-mri-dataset</em> '
            '(831 total scans: 610 Normal · 221 Parkinson) sourced from Kaggle, '
            'with all corrupt files verified and removed. '
            'Three architectures are trained and validated: '
            '<strong style="color:#8a5e00;">HybridNet_RV</strong> (ResNet50 + ViT with Cross-Attention Fusion), '
            '<strong style="color:#8a5e00;">HybridNet_EV</strong> (EfficientNet-B4 + ViT with Cross-Attention Fusion), '
            'and a standalone <strong style="color:#8a5e00;">ViT-B/16</strong> Vision Transformer. '
            'A <strong style="color:#8a5e00;">MiniSegNet</strong> lightweight CNN baseline is also included. '
            'All models are trained with <strong style="color:#8a5e00;">MixUp augmentation</strong>, '
            '<strong style="color:#8a5e00;">Label Smoothing CE loss</strong>, '
            '<strong style="color:#8a5e00;">Cosine Annealing LR</strong>, and '
            '<strong style="color:#8a5e00;">Early Stopping</strong> on a T4 GPU (Google Colab). '
            'Final evaluation uses a <strong style="color:#8a5e00;">Weighted Ensemble</strong> '
            '(weights: ResNet50 0.8 · EfficientNet 0.9 · ViT 0.8 · MiniSegNet 0.6 · HybridNet_RV 1.2 · HybridNet_EV 1.2) '
            'achieving <strong style="color:#8a5e00;">100% ensemble accuracy</strong> on the held-out test set.'
            '</div>'
            '<div style="display:flex;gap:.7rem;flex-wrap:wrap;">'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">🧠 Hybrid Deep Learning</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">🔬 Brain MRI Analysis</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">📊 Grad-CAM XAI</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">✦ 70/15/15 Train/Val/Test Split</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">⚡ Cross-Attention Fusion</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">🔀 MixUp + Label Smoothing</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with _pc2:
        st.markdown(
            '<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
            'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
            'padding:2rem;box-shadow:0 2px 16px rgba(184,134,11,0.08);">'
            '<div style="font-family:Cinzel,serif;font-size:.68rem;color:#b8860b;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:1rem;">📋 Project Specs</div>'
            '<div style="display:flex;flex-direction:column;gap:.75rem;">'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">📦</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Dataset</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'irfansheriff/parkinsons-brain-mri (Kaggle)<br>'
            '831 scans · Normal: 610 · Parkinson: 221<br>'
            'Split: 581 Train · 125 Val · 125 Test</div></div></div>'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">🤖</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Models Trained</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'ResNet50 · EfficientNet-B4 · ViT-B/16<br>'
            'MiniSegNet · HybridNet_RV · HybridNet_EV<br>'
            '+ Weighted Ensemble (6 models)</div></div></div>'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">⚙</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Training Config</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'SEED=42 · IMG_SIZE=224 · BS=16 · EPOCHS=30<br>'
            'MixUp (α=0.3) · Label Smoothing (ε=0.1)<br>'
            'WeightedRandomSampler · Early Stopping (patience=7)</div></div></div>'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">🔥</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Explainability</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'Gradient-weighted CAM (Grad-CAM)<br>'
            'Per-model spatial attention heatmaps<br>'
            'ViT: N/A (no spatial conv maps)</div></div></div>'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">🖥</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Tech Stack</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'PyTorch · timm · torchvision · scikit-learn<br>'
            'Streamlit · ReportLab · OpenCV · Seaborn<br>'
            'Google Colab · Tesla T4 GPU (15.6 GB VRAM)</div></div></div>'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">🎓</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Academic Year</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'B.Tech Final Year 2025-26<br>Dept. of ECE · BVC College of Engineering</div></div></div>'

            '</div></div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  MODEL PERFORMANCE  — exact figures from notebook output
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head" style="margin-top:2rem;">✦ Model Performance (Test Set)</div>', unsafe_allow_html=True)

    # Parameters table from notebook
    _param_data = [
        ("ResNet50",       "24.6M", "23.1M", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"),
        ("EfficientNet-B4","17.6M", "13.9M", "0.9920", "0.9896", "0.9946", "0.9848", "0.9997"),
        ("ViT-B/16",       "85.8M", "28.4M", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"),
        ("MiniSegNet",     "0.3M",  "0.3M",  "0.9760", "0.9682", "0.9342", "0.9728", "0.9960"),
        ("HybridNet_RV",   "113M",  "32.1M", "0.9600", "0.9508", "0.9342", "0.9728", "0.9987"),
        ("HybridNet_EV",   "105M",  "44.3M", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"),
        ("Ensemble",       "—",     "—",     "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"),
    ]

    _th_style = ('font-family:Cinzel,serif;font-size:.62rem;color:#fff;'
                 'letter-spacing:1px;text-transform:uppercase;padding:.5rem .6rem;'
                 'background:#7a4f00;text-align:center;')
    _td_style = ('font-family:EB Garamond,serif;font-size:.9rem;color:#2c1a00;'
                 'padding:.45rem .6rem;text-align:center;border-bottom:1px solid rgba(184,134,11,0.12);')
    _td_l_style = ('font-family:Cinzel,serif;font-size:.62rem;color:#7a4f00;font-weight:700;'
                   'padding:.45rem .8rem;text-align:left;border-bottom:1px solid rgba(184,134,11,0.12);'
                   'background:rgba(184,134,11,0.04);')

    _tbl_html = (
        '<div style="overflow-x:auto;margin-top:.5rem;">'
        '<table style="width:100%;border-collapse:collapse;border:1px solid rgba(184,134,11,0.25);'
        'border-radius:10px;overflow:hidden;background:#fffdf8;">'
        '<thead><tr>'
        f'<th style="{_th_style}text-align:left;">Model</th>'
        f'<th style="{_th_style}">Total Params</th>'
        f'<th style="{_th_style}">Trainable</th>'
        f'<th style="{_th_style}">Accuracy</th>'
        f'<th style="{_th_style}">F1-Macro</th>'
        f'<th style="{_th_style}">Precision</th>'
        f'<th style="{_th_style}">Recall</th>'
        f'<th style="{_th_style}">AUC</th>'
        '</tr></thead><tbody>'
    )
    for row in _param_data:
        _acc_color = '#1a6a38' if float(row[3]) >= 0.999 else ('#b8860b' if float(row[3]) >= 0.98 else '#c0392b')
        _tbl_html += (
            f'<tr>'
            f'<td style="{_td_l_style}">{row[0]}</td>'
            f'<td style="{_td_style}">{row[1]}</td>'
            f'<td style="{_td_style}">{row[2]}</td>'
            f'<td style="{_td_style}font-weight:700;color:{_acc_color};">{row[3]}</td>'
            f'<td style="{_td_style}">{row[4]}</td>'
            f'<td style="{_td_style}">{row[5]}</td>'
            f'<td style="{_td_style}">{row[6]}</td>'
            f'<td style="{_td_style}">{row[7]}</td>'
            f'</tr>'
        )
    _tbl_html += '</tbody></table></div>'
    st.markdown(_tbl_html, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TRAINING DETAILS  — from notebook
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head" style="margin-top:2rem;">✦ Training Pipeline Details</div>', unsafe_allow_html=True)
    _tp1, _tp2, _tp3 = st.columns(3, gap='medium')

    for _col, _icon, _title, _color, _items in [
        (_tp1, '📐', 'Data Augmentation', '#b8860b', [
            'Resize to 256×256 → RandomCrop 224',
            'RandomHorizontalFlip (p=0.5)',
            'RandomVerticalFlip (p=0.2)',
            'RandomRotation (±20°)',
            'RandomAffine (translate 0.1, scale 0.85–1.15)',
            'ColorJitter (brightness/contrast 0.4, saturation 0.2)',
            'GaussianBlur (kernel=3, σ=0.1–2.0)',
            'RandomErasing (p=0.2, scale 0.02–0.10)',
            'MixUp (α=0.3, applied 60% of batches)',
        ]),
        (_tp2, '⚙', 'Optimiser & Schedule', '#1a6a38', [
            'Optimiser: AdamW',
            'ResNet50 / EffNet: LR = 1e-4',
            'ViT-B/16: LR = 5e-5',
            'MiniSegNet: LR = 8e-5',
            'HybridNet RV/EV: LR = 5e-5',
            'Scheduler: CosineAnnealingLR',
            'AMP (mixed precision) enabled',
            'Gradient clip norm = 1.0',
            'Early stopping patience = 7',
        ]),
        (_tp3, '📊', 'Evaluation Protocol', '#004a90', [
            'Stratified 70 / 15 / 15 split',
            'Train: 581 · Val: 125 · Test: 125',
            'WeightedRandomSampler (class balance)',
            'Label Smoothing CE (ε=0.1)',
            'Metrics: Acc · F1 · Precision · Recall · AUC',
            'Normalised Confusion Matrices',
            'ROC curves per model',
            'Grad-CAM on test samples',
            'Results saved to Google Drive',
        ]),
    ]:
        with _col:
            _li = ''.join(
                f'<li style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;'
                f'padding:.18rem 0;line-height:1.5;">{it}</li>'
                for it in _items
            )
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
                f'padding:1.6rem 1.4rem;box-shadow:0 2px 12px rgba(184,134,11,0.07);'
                f'border-top:3px solid {_color};">'
                f'<div style="font-size:1.4rem;margin-bottom:.5rem;">{_icon}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.65rem;color:{_color};'
                f'letter-spacing:2px;text-transform:uppercase;margin-bottom:.8rem;font-weight:700;">'
                f'{_title}</div>'
                f'<ul style="padding-left:1.1rem;margin:0;">{_li}</ul>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  ENSEMBLE WEIGHTS  — from notebook
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head" style="margin-top:2rem;">✦ Ensemble Configuration</div>', unsafe_allow_html=True)
    _ew_data = [
        ("ResNet50",        "0.8", "Standalone CNN backbone"),
        ("EfficientNet-B4", "0.9", "Standalone efficient backbone"),
        ("ViT-B/16",        "0.8", "Pure Vision Transformer"),
        ("MiniSegNet",      "0.6", "Lightweight CNN baseline"),
        ("HybridNet_RV",    "1.2", "ResNet50 + ViT Cross-Attention"),
        ("HybridNet_EV",    "1.2", "EfficientNet-B4 + ViT Cross-Attention"),
    ]
    _ew_cols = st.columns(6, gap='small')
    for _ci, (_name, _w, _desc) in enumerate(_ew_data):
        _wf = float(_w)
        _wc = '#1a6a38' if _wf >= 1.2 else ('#b8860b' if _wf >= 0.8 else '#c0392b')
        with _ew_cols[_ci]:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.2);border-radius:10px;'
                f'padding:1rem .6rem;text-align:center;'
                f'border-top:3px solid {_wc};">'
                f'<div style="font-family:Playfair Display,serif;font-size:1.5rem;'
                f'font-weight:900;color:{_wc};">{_w}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:#7a4f00;'
                f'letter-spacing:.5px;margin:.3rem 0;font-weight:700;">{_name}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.78rem;'
                f'color:#9a7030;line-height:1.4;">{_desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div style="margin-top:.8rem;padding:.8rem 1.2rem;'
        'background:rgba(184,134,11,0.06);border:1px solid rgba(184,134,11,0.2);'
        'border-radius:8px;font-family:EB Garamond,serif;font-size:.92rem;color:#5a3a0a;">'
        '⚗ <strong>Ensemble formula:</strong> weighted average of softmax probabilities across all 6 models, '
        'with higher weights assigned to the HybridNet variants (RV & EV) that combine both CNN spatial '
        'features and ViT global attention via Cross-Attention Fusion. Total weight sum = 5.5.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    #  TRAINING RESULTS IMAGES  — from Google Drive
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head" style="margin-top:2rem;">✦ Training Results</div>', unsafe_allow_html=True)
    RESULT_IMAGES = {
        'Confusion Matrices':   '1qQgCIXUAvBFIqKOg31E2RKZqpMX3bfHn',
        'Class Distribution':   '1c10Rb-uQU8_Wtu59Q48P-E20IVfHDIwx',
        'GradCAM All Models':   '1uKot-ObwVs5VoqFXwxzWOUkEuV4pLr--',
        'Model Comparison':     '1pSapafZeP9r-gRs-dFsHOLgcJOddSFiz',
        'ROC Curves':           '1X78209LQ7PTwbu12lC8jWou9DqmF5RGp',
        'Sample Grid':          '1J6li-fQx28HqKuRclAlNHyUSzLOQ6Vvp',
        'Training Curves':      '13HIiDGEy8jEZgYis3vTzqctYtWlpKZ4N',
    }

    @st.cache_data(show_spinner=False)
    def fetch_drive_image(file_id):
        import requests
        for url in [
            f"https://drive.google.com/thumbnail?id={file_id}&sz=w1200",
            f"https://lh3.googleusercontent.com/d/{file_id}",
        ]:
            try:
                resp = requests.get(url, timeout=15, allow_redirects=True)
                if resp.status_code == 200 and 'image' in resp.headers.get('Content-Type', ''):
                    return resp.content
            except Exception:
                continue
        return None

    _img_items = list(RESULT_IMAGES.items())
    for _i in range(0, len(_img_items), 2):
        _ri_cols = st.columns(2, gap='large')
        for _j, _col in enumerate(_ri_cols):
            if _i + _j < len(_img_items):
                _name, _fid = _img_items[_i + _j]
                with _col:
                    st.markdown(
                        f'<div class="card" style="padding:1rem;">'
                        f'<div style="font-family:Cinzel,serif;font-size:.65rem;'
                        f'color:#b8860b;letter-spacing:2px;text-transform:uppercase;'
                        f'margin-bottom:.8rem;text-align:center;">◈ {_name}</div>',
                        unsafe_allow_html=True,
                    )
                    _img_data = fetch_drive_image(_fid)
                    if _img_data:
                        st.image(_img_data, use_column_width=True)
                    else:
                        st.warning(f'Could not load: {_name}')
                    st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  PROJECT TEAM
    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════
    #  WHO ARE WE  — unique immersive team section
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;">✦ Who We Are</div>', unsafe_allow_html=True)

    # Guidance banner — full width, prominent
    st.markdown(
        '<div style="background:linear-gradient(135deg,#fdf3dc 0%,#fceedd 50%,#fdf3dc 100%);'
        'border:1.5px solid rgba(184,134,11,0.4);border-radius:16px;'
        'padding:1.6rem 2.5rem;margin-bottom:1.8rem;position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        'background:linear-gradient(90deg,transparent,#c9a84c,#e8c96a,#c9a84c,transparent);"></div>'
        '<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
        'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
        '<div style="display:flex;align-items:center;gap:2rem;flex-wrap:wrap;">'
        '<div style="font-size:3rem;">🎓</div>'
        '<div style="flex:1;">'
        '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#b8860b;'
        'letter-spacing:3px;text-transform:uppercase;margin-bottom:.3rem;">'
        '⭐ Under the Esteemed Guidance of</div>'
        '<div style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;'
        'color:#2c1a00;line-height:1.2;margin-bottom:.25rem;">'
        'Mr. N P U V S N Pavan Kumar, M.Tech</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.95rem;font-style:italic;color:#8a5e00;">'
        'Assistant Professor · Dept. of ECE · Deputy CoE III · BVC College of Engineering, Rajahmundry'
        '</div></div>'
        '<div style="background:linear-gradient(135deg,#7a4f00,#b8860b);'
        'border-radius:10px;padding:.8rem 1.4rem;text-align:center;flex-shrink:0;">'
        '<div style="font-family:Cinzel,serif;font-size:.6rem;color:rgba(255,255,255,0.7);'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.2rem;">Project Guide</div>'
        '<div style="font-size:1.6rem;">🏆</div>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    # Team members — unique card per person with role-themed gradient
    _team_data = [
        {
            'roll':  '236M5A0408',
            'name':  'G Srinivasu',
            'role':  'AI Model Architect & Backend Developer',
            'icon':  '🤖',
            'color1':'#1a3a2a', 'color2':'#2d6a4f',
            'accent':'#4ade80',
            'tags':  ['PyTorch', 'Model Training', 'Backend API'],
            'desc':  'Designed and trained all 6 deep learning models including the HybridNet Cross-Attention Fusion architecture. Led backend integration and model optimization on Google Colab T4 GPU.',
        },
        {
            'roll':  '226M1A0460',
            'name':  'S Anusha Devi',
            'role':  'UI/UX Designer & Frontend Developer',
            'icon':  '🎨',
            'color1':'#2a1a3a', 'color2':'#6a2d8f',
            'accent':'#c084fc',
            'tags':  ['Streamlit', 'CSS Design', 'User Experience'],
            'desc':  'Crafted the entire royal gold aesthetic UI including the hero banner, interactive tabs, and PDF report layout. Ensured a seamless, elegant user experience across all features.',
        },
        {
            'roll':  '226M1A0473',
            'name':  'V V Siva Vardhan',
            'role':  'Data Engineer & Quality Analyst',
            'icon':  '📊',
            'color1':'#1a2a3a', 'color2':'#2d4f8a',
            'accent':'#60a5fa',
            'tags':  ['Data Pipeline', 'Augmentation', 'Testing'],
            'desc':  'Managed the full data pipeline including Kaggle dataset ingestion, image verification, augmentation strategy design (MixUp, ColorJitter, RandomErasing), and model evaluation testing.',
        },
        {
            'roll':  '236M5A0415',
            'name':  'N L Sandeep',
            'role':  'Documentation & Integration Lead',
            'icon':  '📝',
            'color1':'#3a2a1a', 'color2':'#8a4f2d',
            'accent':'#fb923c',
            'tags':  ['PDF Reports', 'Integration', 'Documentation'],
            'desc':  'Built the automated gold-themed PDF report generation system using ReportLab. Handled system integration, ensemble logic documentation, and project report writing.',
        },
    ]

    # Row 1 — first two members
    _tc1, _tc2 = st.columns(2, gap='large')
    for _col, _m in zip([_tc1, _tc2], _team_data[:2]):
        with _col:
            _tags_html = ''.join(
                f'<span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);'
                f'border-radius:20px;padding:.2rem .75rem;font-family:Cinzel,serif;font-size:.56rem;'
                f'color:rgba(255,255,255,0.85);letter-spacing:.8px;">{t}</span>'
                for t in _m['tags']
            )
            st.markdown(
                f'<div style="background:linear-gradient(145deg,{_m["color1"]},{_m["color2"]});'
                f'border:1px solid rgba(255,255,255,0.1);border-radius:16px;'
                f'padding:1.8rem 1.6rem;position:relative;overflow:hidden;'
                f'box-shadow:0 8px 32px rgba(0,0,0,0.25);transition:transform .3s;">'
                # Decorative glow
                f'<div style="position:absolute;top:-30px;right:-30px;width:100px;height:100px;'
                f'background:radial-gradient(circle,{_m["accent"]}22,transparent 70%);'
                f'border-radius:50%;"></div>'
                # Top accent line
                f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
                f'background:linear-gradient(90deg,transparent,{_m["accent"]},transparent);"></div>'
                # Icon + roll
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">'
                f'<div style="width:56px;height:56px;background:rgba(255,255,255,0.08);'
                f'border:1px solid rgba(255,255,255,0.15);border-radius:14px;'
                f'display:flex;align-items:center;justify-content:center;font-size:1.8rem;">'
                f'{_m["icon"]}</div>'
                f'<div style="background:rgba(255,255,255,0.08);border-radius:6px;'
                f'padding:.2rem .6rem;font-family:Cinzel,serif;font-size:.55rem;'
                f'color:{_m["accent"]};letter-spacing:1px;">{_m["roll"]}</div>'
                f'</div>'
                # Name
                f'<div style="font-family:Playfair Display,serif;font-size:1.25rem;font-weight:700;'
                f'color:#ffffff;margin-bottom:.2rem;line-height:1.2;">{_m["name"]}</div>'
                # Role
                f'<div style="font-family:Cinzel,serif;font-size:.62rem;color:{_m["accent"]};'
                f'letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.8rem;">{_m["role"]}</div>'
                # Desc
                f'<div style="font-family:EB Garamond,serif;font-size:.9rem;'
                f'color:rgba(255,255,255,0.75);line-height:1.7;margin-bottom:1rem;">{_m["desc"]}</div>'
                # Tags
                f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;">{_tags_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:.8rem;"></div>', unsafe_allow_html=True)

    # Row 2 — next two members
    _tc3, _tc4 = st.columns(2, gap='large')
    for _col, _m in zip([_tc3, _tc4], _team_data[2:]):
        with _col:
            _tags_html = ''.join(
                f'<span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);'
                f'border-radius:20px;padding:.2rem .75rem;font-family:Cinzel,serif;font-size:.56rem;'
                f'color:rgba(255,255,255,0.85);letter-spacing:.8px;">{t}</span>'
                for t in _m['tags']
            )
            st.markdown(
                f'<div style="background:linear-gradient(145deg,{_m["color1"]},{_m["color2"]});'
                f'border:1px solid rgba(255,255,255,0.1);border-radius:16px;'
                f'padding:1.8rem 1.6rem;position:relative;overflow:hidden;'
                f'box-shadow:0 8px 32px rgba(0,0,0,0.25);">'
                f'<div style="position:absolute;top:-30px;right:-30px;width:100px;height:100px;'
                f'background:radial-gradient(circle,{_m["accent"]}22,transparent 70%);border-radius:50%;"></div>'
                f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
                f'background:linear-gradient(90deg,transparent,{_m["accent"]},transparent);"></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">'
                f'<div style="width:56px;height:56px;background:rgba(255,255,255,0.08);'
                f'border:1px solid rgba(255,255,255,0.15);border-radius:14px;'
                f'display:flex;align-items:center;justify-content:center;font-size:1.8rem;">'
                f'{_m["icon"]}</div>'
                f'<div style="background:rgba(255,255,255,0.08);border-radius:6px;'
                f'padding:.2rem .6rem;font-family:Cinzel,serif;font-size:.55rem;'
                f'color:{_m["accent"]};letter-spacing:1px;">{_m["roll"]}</div>'
                f'</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.25rem;font-weight:700;'
                f'color:#ffffff;margin-bottom:.2rem;line-height:1.2;">{_m["name"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.62rem;color:{_m["accent"]};'
                f'letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.8rem;">{_m["role"]}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.9rem;'
                f'color:rgba(255,255,255,0.75);line-height:1.7;margin-bottom:1rem;">{_m["desc"]}</div>'
                f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;">{_tags_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  PROJECT GUIDANCE — redesigned
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;">✦ Project Guidance</div>', unsafe_allow_html=True)
    _g1, _g2, _g3 = st.columns(3, gap='medium')
    _guide_data = [
        (_g1, 'Project Guide',       '👨\u200d🏫', '#b8860b', '#fdf3dc',
         'Mr. N P U V S N Pavan Kumar, M.Tech',
         ['Assistant Professor', 'Dept. of ECE', 'Deputy CoE III'],
         'Primary guide who mentored the team throughout model design, training strategy, and deployment of the NeuroScan AI system.'),
        (_g2, 'Project Coordinator', '📋', '#1a6a38', '#f0fdf4',
         'Mr. K Anji Babu, M.Tech',
         ['Assistant Professor', 'Dept. of ECE'],
         'Coordinated project milestones, review sessions, and facilitated academic compliance throughout the project lifecycle.'),
        (_g3, 'Head of Department',  '👨\u200d💼', '#7a1a1a', '#fff1f1',
         'Dr. S A Vara Prasad, Ph.D, M.Tech',
         ['Professor & HOD, ECE', 'Chairman BoS', 'Anti-Ragging Committee'],
         'Provided departmental leadership and institutional support, enabling access to resources for this B.Tech Final Year Project.'),
    ]
    for _col, _rl, _icon, _rc, _bg, _name, _tags, _desc in _guide_data:
        _th = ''.join(
            f'<span style="background:rgba(0,0,0,0.05);border:1px solid rgba(0,0,0,0.1);'
            f'border-radius:4px;padding:.2rem .7rem;font-family:Cinzel,serif;font-size:.58rem;'
            f'color:{_rc};letter-spacing:.8px;font-weight:600;">{t}</span> '
            for t in _tags
        )
        with _col:
            st.markdown(
                f'<div style="background:{_bg};border:1px solid rgba(0,0,0,0.08);'
                f'border-radius:14px;padding:1.8rem 1.4rem;text-align:center;'
                f'position:relative;box-shadow:0 3px 18px rgba(0,0,0,0.07);">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:4px;'
                f'background:linear-gradient(90deg,transparent,{_rc},transparent);'
                f'border-radius:14px 14px 0 0;"></div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:{_rc};'
                f'letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.8rem;font-weight:700;">'
                f'⭐ {_rl}</div>'
                f'<div style="width:64px;height:64px;background:linear-gradient(135deg,{_rc}22,{_rc}11);'
                f'border:2px solid {_rc}44;border-radius:50%;display:flex;align-items:center;'
                f'justify-content:center;font-size:1.8rem;margin:0 auto .8rem;">{_icon}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1rem;font-weight:700;'
                f'color:#2c1a00;margin-bottom:.6rem;line-height:1.4;">{_name}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:.3rem;justify-content:center;margin-bottom:.8rem;">{_th}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.88rem;color:#5a3a0a;'
                f'line-height:1.65;font-style:italic;">{_desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(192,57,43,0.05),'
        'rgba(192,57,43,0.02));border:1px solid rgba(192,57,43,0.2);'
        'border-left:4px solid #c0392b;border-radius:10px;padding:1.2rem 1.8rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#c0392b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">⚕ Medical Disclaimer</div>'
        '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a1a1a;line-height:1.7;">'
        'This project is developed <strong>for academic and research purposes only</strong>. '
        "It is a B.Tech Final Year Project at BVC College of Engineering, Rajahmundry "
        '(<a href="https://bvcr.edu.in" target="_blank" style="color:#c0392b;">bvcr.edu.in</a>). '
        "The AI system is not a certified medical device and must not replace professional clinical diagnosis. "
        'Always consult a qualified neurologist.'
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
