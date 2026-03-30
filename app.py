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
import gc
import gdown
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import timm

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — SINGLE MODEL: HybridNet_EV only
# ══════════════════════════════════════════════════════════════════════════════

MODEL_FILE_ID   = "1uXL90WvSm7akZbgpYZCinGAzLUhwhl6i"
MODEL_FILENAME  = "hybridnet_ev_best.pth"
MODEL_LABEL     = "HybridNet_EV (EfficientNet-B4 + ViT)"

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE — HybridNet_EV only
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
        x    = self.n1(cnn_t + a)
        return self.n2(x + self.ffn(x))


class HybridNet_EV(nn.Module):
    """EfficientNet-B4 + ViT Cross-Attention Fusion"""
    def __init__(self, nc=2, d_model=512, n_heads=8, dropout=0.2):
        super().__init__()
        self.cnn      = timm.create_model('efficientnet_b4', pretrained=False, features_only=True)
        cnn_ch        = self.cnn.feature_info[-1]['num_chs']
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

    def forward(self, x):
        cf    = self.cnn(x)[-1]
        ct    = self.cnn_proj(cf.flatten(2).transpose(1, 2))
        vt    = self.vit_proj(self.vit.forward_features(x)[:, 1:, :])
        fused = self.fusion(ct, vt)
        return self.head(self.gap(fused.transpose(1, 2)).squeeze(-1))

    def get_gradcam_layer(self):
        return self.cnn.blocks[-1][-1].conv_pwl


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_model(path: str) -> None:
    if not os.path.exists(path):
        with st.spinner(f"Downloading model — please wait…"):
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
                    path, quiet=False,
                )
                st.success("Model downloaded successfully.")
            except Exception as exc:
                st.error(f"Download failed: {exc}")
                st.info("Ensure the Google Drive file is shared as 'Anyone with the link'.")
                st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL  — cached, CPU-first to conserve memory
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    device = torch.device("cpu")   # CPU to avoid OOM on free-tier
    try:
        ck = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    if isinstance(ck, dict) and 'model_state_dict' in ck:
        state_dict = ck['model_state_dict']
        nc         = ck.get('config', {}).get('num_classes', 2)
        raw_cls    = ck.get('classes', ['normal', 'parkinson'])
    elif isinstance(ck, dict):
        state_dict = ck
        nc         = 2
        raw_cls    = ['normal', 'parkinson']
    else:
        st.error("Unrecognised checkpoint format.")
        st.stop()

    cls_map = {
        'normal':               'Normal',
        'healthy':              'Normal',
        'parkinson':            "Parkinson's Disease",
        'parkinsons':           "Parkinson's Disease",
        "parkinson's disease":  "Parkinson's Disease",
    }
    classes = [cls_map.get(c.lower(), c.title()) for c in raw_cls]

    m = HybridNet_EV(nc=nc)
    try:
        m.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        st.error(f"Weight mismatch: {e}")
        st.stop()

    m.eval()
    return m.to(device), classes, device


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
        t = img_tensor.clone().to(device).requires_grad_(True)
        with torch.enable_grad():
            out = self.model(t)
            self.model.zero_grad()
            out[0, class_idx].backward(retain_graph=True)
        if self._grads is None or self._acts is None:
            return np.zeros((224, 224))
        weights = self._grads.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, (224, 224), mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().detach().numpy()
        lo, hi  = cam.min(), cam.max()
        return (cam - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros((224, 224))


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


def predict(model, device, classes, pil_image):
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
    if parkinson_idx is None: parkinson_idx = 1

    normal_prob    = float(probs_np[normal_idx]    * 100)
    parkinson_prob = float(probs_np[parkinson_idx] * 100)
    is_parkinson   = (class_idx == parkinson_idx)
    risk = ('High' if confidence_pct >= 85 else 'Moderate') if is_parkinson else 'Low'

    overlay = heatmap = None
    try:
        gc_layer = model.get_gradcam_layer()
        gcamp    = GradCAM(model, gc_layer)
        fresh    = TRANSFORM(img_rgb).unsqueeze(0)
        cam_map  = gcamp.generate(fresh, class_idx, device)
        if cam_map.max() > 0:
            overlay, heatmap = apply_colormap(img_rgb, cam_map)
    except Exception:
        pass

    # Free memory
    del img_tensor
    gc.collect()

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
        'model_name':     MODEL_LABEL,
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

    GOLD       = colors.HexColor('#b8860b')
    GOLD_DARK  = colors.HexColor('#7a4f00')
    GOLD_LIGHT = colors.HexColor('#fdf3dc')
    PARCHMENT  = colors.HexColor('#fdf8f0')
    DEEP       = colors.HexColor('#2c1a00')
    MIDGOLD    = colors.HexColor('#c9a84c')
    CRIMSON    = colors.HexColor('#c0392b')
    EMERALD    = colors.HexColor('#1a6a38')
    GREY_TEXT  = colors.HexColor('#7a5a1a')

    title_s    = ParagraphStyle('T',   parent=styles['Heading1'],  fontSize=22, textColor=GOLD_DARK, spaceAfter=2,  alignment=TA_CENTER, fontName='Helvetica-Bold')
    subtitle_s = ParagraphStyle('Sub', parent=styles['Normal'],    fontSize=9,  textColor=GOLD,      spaceAfter=4,  alignment=TA_CENTER, fontName='Helvetica')
    project_s  = ParagraphStyle('Prj', parent=styles['Normal'],    fontSize=10, textColor=DEEP,      spaceAfter=16, alignment=TA_CENTER, fontName='Helvetica-Bold')
    head_s     = ParagraphStyle('H',   parent=styles['Heading2'],  fontSize=12, textColor=GOLD_DARK, spaceAfter=6, spaceBefore=12, fontName='Helvetica-Bold')
    body_s     = ParagraphStyle('B',   parent=styles['Normal'],    fontSize=10, textColor=DEEP,      leading=14)
    warn_s     = ParagraphStyle('W',   parent=styles['Normal'],    fontSize=9,  textColor=colors.HexColor('#7f1d1d'), leading=13, backColor=colors.HexColor('#fff1f1'), borderPad=6)
    foot_s     = ParagraphStyle('F',   parent=styles['Normal'],    fontSize=8,  textColor=GREY_TEXT, alignment=TA_CENTER)

    def gold_line():
        t = Table([['', '', '']], colWidths=[0.5*inch, 6.5*inch, 0.5*inch])
        t.setStyle(TableStyle([
            ('LINEABOVE',      (1,0),(1,0), 1, MIDGOLD),
            ('ALIGN',          (0,0),(-1,-1), 'CENTER'),
            ('TOPPADDING',     (0,0),(-1,-1), 0),
            ('BOTTOMPADDING',  (0,0),(-1,-1), 4),
        ]))
        return t

    def gold_tbl(rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,0), GOLD_DARK),
            ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
            ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0),(-1,0), 10),
            ('BOTTOMPADDING', (0,0),(-1,0), 8),
            ('TOPPADDING',    (0,0),(-1,0), 8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1), [GOLD_LIGHT, PARCHMENT]),
            ('GRID',          (0,0),(-1,-1), 0.4, MIDGOLD),
            ('FONTNAME',      (0,1),(-1,-1), 'Helvetica'),
            ('FONTSIZE',      (0,1),(-1,-1), 10),
            ('TEXTCOLOR',     (0,1),(-1,-1), DEEP),
            ('TOPPADDING',    (0,1),(-1,-1), 6),
            ('BOTTOMPADDING', (0,1),(-1,-1), 6),
            ('LEFTPADDING',   (0,0),(-1,-1), 10),
            ('FONTNAME',      (0,1),(0,-1),  'Helvetica-Bold'),
            ('TEXTCOLOR',     (0,1),(0,-1),  GOLD_DARK),
        ]))
        return t

    story.append(Paragraph('NeuroScan AI', title_s))
    story.append(Paragraph("Parkinson's Disease MRI Analysis Report", subtitle_s))
    story.append(Paragraph('PARKINSONS DISEASE DETECTION USING DEEP LEARNING ON BRAIN MRI', project_s))
    story.append(gold_line())
    story.append(Spacer(1, 4))

    is_pk      = result.get('is_parkinson', False)
    diag_color = CRIMSON if is_pk else EMERALD
    verdict    = Table([[result['prediction'].upper()]], colWidths=[7*inch])
    verdict.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), diag_color),
        ('TEXTCOLOR',     (0,0),(-1,-1), colors.white),
        ('FONTNAME',      (0,0),(-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 16),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0),(-1,-1), 10),
        ('BOTTOMPADDING', (0,0),(-1,-1), 10),
    ]))
    story.append(verdict)
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
        ['Metric',                  'Value'],
        ['Diagnosis',               result['prediction']],
        ['Confidence Score',        f"{result['confidence']:.2f}%"],
        ['Normal Probability',      f"{result['normal_prob']:.2f}%"],
        ["Parkinson's Probability", f"{result['parkinson_prob']:.2f}%"],
        ['Risk Level',              result['risk_level']],
        ['Analysis Time',           result['timestamp']],
        ['AI Model',                result['model_name']],
    ], [2.8*inch, 4.2*inch]))

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
        cap_txt = 'Left: Original MRI  |  Right: Grad-CAM Overlay'
    else:
        img_tbl = Table([[RLImage(img_buf, width=2.8*inch, height=2.8*inch)]], colWidths=[3.2*inch])
        cap_txt = 'Original MRI Scan'

    img_tbl.setStyle(TableStyle([
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
        ('BOX',        (0,0),(-1,-1), 1.5, MIDGOLD),
        ('BACKGROUND', (0,0),(-1,-1), PARCHMENT),
    ]))
    story.append(img_tbl)
    cap_s = ParagraphStyle('C', parent=styles['Normal'], fontSize=8, textColor=GOLD, alignment=TA_CENTER, spaceBefore=4)
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
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroScan AI — Parkinson's MRI Analysis",
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Session state
for _k, _v in [
    ('prediction_made',   False),
    ('patient_data',      {}),
    ('prediction_result', {}),
    ('batch_results',     []),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
#  ENHANCED ROYAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400&family=Cinzel:wght@400;600;700;900&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

:root {
  --gold:#b8860b; --gold-b:#9a6e00; --gold-d:#c9a84c;
  --gold-m:rgba(184,134,11,0.10); --gold-g:rgba(184,134,11,0.28);
  --gold-l:rgba(184,134,11,0.18); --gold-bdr:rgba(184,134,11,0.45);
  --crimson:#c0392b; --crimson-d:rgba(192,57,43,0.10);
  --emerald:#1e8449; --emerald-d:rgba(30,132,73,0.10);
  --text:#2c1a00; --text2:#7a5a1a; --text3:#b89040;
  --r:12px; --rs:7px;
  --sw:0 6px 32px rgba(184,134,11,0.13),0 1px 8px rgba(0,0,0,0.07);
  --sg:0 6px 28px rgba(184,134,11,0.22);
  --parchment:#fdf8f0; --cream:#fffdf8; --warm:#fdf3dc;
}

html,body,.stApp{background:var(--parchment)!important;font-family:'EB Garamond','Cormorant Garamond',serif!important;color:var(--text)!important;}

/* Decorative background pattern */
.stApp::before{content:'';position:fixed;inset:0;z-index:0;
  background-image:
    radial-gradient(ellipse 90% 55% at 50% 0%,rgba(184,134,11,0.07) 0%,transparent 65%),
    radial-gradient(ellipse 60% 40% at 100% 100%,rgba(184,134,11,0.04) 0%,transparent 60%),
    repeating-linear-gradient(45deg,transparent,transparent 34px,rgba(184,134,11,0.022) 34px,rgba(184,134,11,0.022) 35px),
    repeating-linear-gradient(-45deg,transparent,transparent 34px,rgba(184,134,11,0.022) 34px,rgba(184,134,11,0.022) 35px);
  pointer-events:none;}

.block-container{padding-top:0!important;max-width:1400px!important;position:relative;z-index:1;}
#MainMenu,footer,header{visibility:hidden;}

/* ── Enhanced Cards with double-border effect ── */
.card{
  background:linear-gradient(150deg,var(--cream) 0%,#fff8ec 100%);
  border:1px solid var(--gold-bdr);
  border-radius:var(--r);
  padding:1.8rem 2rem;
  margin:.7rem 0;
  box-shadow:var(--sw);
  position:relative;
  transition:border-color .3s,box-shadow .3s,transform .2s;
  outline:3px solid rgba(184,134,11,0.07);
  outline-offset:3px;
}
.card::before{
  content:'';position:absolute;top:0;left:10%;right:10%;height:2px;
  background:linear-gradient(90deg,transparent,var(--gold-d),rgba(232,201,106,0.9),var(--gold-d),transparent);
  border-radius:0 0 4px 4px;
}
.card::after{
  content:'';position:absolute;bottom:0;left:10%;right:10%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);
}
.card:hover{
  border-color:rgba(184,134,11,0.7);
  box-shadow:var(--sw),0 0 40px rgba(201,168,76,0.10);
  transform:translateY(-1px);
  outline-color:rgba(184,134,11,0.14);
}

/* ── Section headings ── */
.sec-head{
  display:flex;align-items:center;gap:.9rem;
  font-family:'Cinzel',serif;font-size:.68rem;font-weight:600;
  color:var(--gold);letter-spacing:3.5px;text-transform:uppercase;
  margin:2rem 0 1rem;
}
.sec-head::before{content:'';flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--gold-d));}
.sec-head::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--gold-d),transparent);}

/* ── Inputs ── */
.stTextInput input,.stNumberInput input,.stTextArea textarea,.stDateInput input{
  background:var(--cream)!important;color:var(--text)!important;
  border:1.5px solid var(--gold-l)!important;
  border-radius:var(--rs)!important;
  font-family:'EB Garamond',serif!important;font-size:1rem!important;
  transition:border-color .25s,box-shadow .25s!important;
  box-shadow:inset 0 1px 4px rgba(184,134,11,0.06)!important;
}
.stTextInput input:focus,.stNumberInput input:focus,.stTextArea textarea:focus{
  border-color:var(--gold)!important;
  box-shadow:0 0 0 3px rgba(201,168,76,0.15),inset 0 1px 4px rgba(184,134,11,0.06)!important;
  outline:none!important;
}
div[data-baseweb="select"]>div{
  background:var(--cream)!important;
  border:1.5px solid var(--gold-l)!important;
  border-radius:var(--rs)!important;
  color:var(--text)!important;
  font-family:'EB Garamond',serif!important;
}
label{
  color:var(--text2)!important;font-family:'Cinzel',serif!important;
  font-size:.68rem!important;font-weight:600!important;
  letter-spacing:1.5px!important;text-transform:uppercase!important;
}

/* ── Buttons ── */
.stButton>button{
  background:linear-gradient(135deg,#6a4e1a 0%,#a8822a 25%,#c9a84c 45%,#e8c96a 55%,#c9a84c 75%,#8a6020 100%)!important;
  color:#1a0a00!important;
  border:1.5px solid rgba(232,201,106,0.6)!important;
  border-radius:var(--rs)!important;
  padding:.75rem 1.6rem!important;
  font-family:'Cinzel',serif!important;font-size:.78rem!important;
  font-weight:700!important;letter-spacing:2.5px!important;
  text-transform:uppercase!important;width:100%!important;
  transition:all .3s ease!important;
  box-shadow:0 3px 18px rgba(201,168,76,0.35),inset 0 1px 0 rgba(255,255,255,0.25),inset 0 -1px 0 rgba(0,0,0,0.10)!important;
  position:relative;overflow:hidden;
}
.stButton>button:hover{
  background:linear-gradient(135deg,#7a5e2a 0%,#b8922e 25%,#d9b85c 45%,#f8d97a 55%,#d9b85c 75%,#9a7030 100%)!important;
  transform:translateY(-2px)!important;
  box-shadow:0 7px 28px rgba(201,168,76,0.50),inset 0 1px 0 rgba(255,255,255,0.3)!important;
}
.stButton>button:active{transform:translateY(0)!important;}

.stDownloadButton>button{
  background:linear-gradient(135deg,#072010 0%,#0f5028 35%,#1a7a3a 55%,#0f5028 80%,#072010 100%)!important;
  color:#b0e8c0!important;
  border:1.5px solid rgba(74,222,128,0.35)!important;
  border-radius:var(--rs)!important;
  padding:.75rem 1.6rem!important;
  font-family:'Cinzel',serif!important;font-size:.78rem!important;
  font-weight:700!important;letter-spacing:2.5px!important;
  text-transform:uppercase!important;width:100%!important;
  transition:all .3s ease!important;
  box-shadow:0 3px 18px rgba(30,132,73,0.35),inset 0 1px 0 rgba(255,255,255,0.10)!important;
}
.stDownloadButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 7px 28px rgba(30,132,73,0.45)!important;}

/* ── File uploader ── */
[data-testid="stFileUploader"]{
  background:var(--cream)!important;
  border:2px dashed var(--gold-bdr)!important;
  border-radius:var(--r)!important;
  padding:1.6rem!important;
  transition:border-color .3s,box-shadow .3s!important;
  box-shadow:inset 0 2px 8px rgba(184,134,11,0.05)!important;
}
[data-testid="stFileUploader"]:hover{
  border-color:var(--gold)!important;
  box-shadow:0 0 28px var(--gold-m),inset 0 2px 8px rgba(184,134,11,0.05)!important;
}

/* ── Diagnosis badges ── */
.diag-badge{display:inline-flex;align-items:center;gap:.5rem;font-family:'Cinzel',serif;font-size:1.3rem;font-weight:700;padding:.5rem 1.2rem;border-radius:4px;letter-spacing:1px;}
.diag-normal{background:var(--emerald-d);color:#3dbb6e;border:1.5px solid rgba(74,222,128,0.4);}
.diag-parkinson{background:var(--crimson-d);color:#e05555;border:1.5px solid rgba(248,113,113,0.4);}

/* ── Stat tiles with ornate borders ── */
.stat-tile{
  background:linear-gradient(150deg,var(--cream),#fff6e8);
  border:1px solid var(--gold-bdr);
  border-radius:var(--r);
  padding:1.1rem 1.3rem;
  text-align:center;
  box-shadow:var(--sw);
  position:relative;
  outline:2px solid rgba(184,134,11,0.06);
  outline-offset:3px;
  transition:box-shadow .3s,transform .2s;
}
.stat-tile:hover{box-shadow:var(--sg);transform:translateY(-2px);}
.stat-tile::before{
  content:'';position:absolute;top:0;left:20%;right:20%;height:2px;
  background:linear-gradient(90deg,transparent,var(--gold-d),transparent);
}
.stat-value{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#8a5e00;line-height:1.1;margin-bottom:.3rem;}
.stat-label{font-family:'Cinzel',serif;font-size:.58rem;color:var(--text2);letter-spacing:2px;text-transform:uppercase;}

.risk-low{background:rgba(74,222,128,.1);color:#3dbb6e;border:1.5px solid rgba(74,222,128,.35);border-radius:4px;padding:.25rem 1rem;font-size:.75rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1px;}
.risk-moderate{background:rgba(251,191,36,.1);color:#d4a200;border:1.5px solid rgba(251,191,36,.35);border-radius:4px;padding:.25rem 1rem;font-size:.75rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1px;}
.risk-high{background:rgba(248,113,113,.1);color:#e05555;border:1.5px solid rgba(248,113,113,.35);border-radius:4px;padding:.25rem 1rem;font-size:.75rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1px;}

/* ── Probability bars ── */
.prob-row{margin:.6rem 0;}
.prob-label{font-family:'Cinzel',serif;font-size:.65rem;color:var(--text2);margin-bottom:.3rem;display:flex;justify-content:space-between;letter-spacing:1px;}
.prob-track{background:rgba(0,0,0,0.04);border-radius:4px;height:8px;overflow:hidden;border:1px solid rgba(184,134,11,0.12);box-shadow:inset 0 1px 3px rgba(0,0,0,0.05);}
.prob-fill-n{height:100%;border-radius:4px;background:linear-gradient(90deg,#0a4f30,#3dbb6e);transition:width .9s cubic-bezier(.22,1,.36,1);}
.prob-fill-p{height:100%;border-radius:4px;background:linear-gradient(90deg,#7f1010,#e05555);transition:width .9s cubic-bezier(.22,1,.36,1);}

/* ── Image frames with corner ornaments ── */
.img-frame{
  border:1.5px solid var(--gold-bdr);
  border-radius:var(--r);
  overflow:hidden;
  background:#f5f0e8;
  box-shadow:var(--sw);
  position:relative;
}
.img-caption{font-family:'Cinzel',serif;font-size:.6rem;color:var(--text3);text-align:center;padding:.45rem 0 .25rem;letter-spacing:1.5px;text-transform:uppercase;}
.hm-legend{display:flex;align-items:center;gap:.6rem;font-family:'Cinzel',serif;font-size:.6rem;color:var(--text3);margin-top:.6rem;letter-spacing:1px;}
.hm-bar{flex:1;height:6px;border-radius:3px;background:linear-gradient(90deg,#00008b,#0000ff,#00ff00,#ffff00,#ff0000);}

/* ── TABS ── */
.stTabs{margin-top:.5rem;}
.stTabs [data-baseweb="tab-list"]{
  background:linear-gradient(135deg,var(--warm),var(--cream))!important;
  border:1.5px solid rgba(184,134,11,0.35)!important;
  border-radius:12px!important;
  padding:.4rem .5rem!important;
  gap:.3rem!important;
  justify-content:center!important;
  align-items:center!important;
  box-shadow:0 2px 12px rgba(184,134,11,0.08)!important;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Cinzel',serif!important;font-size:.72rem!important;
  font-weight:700!important;color:#7a5a1a!important;
  border-radius:8px!important;padding:.65rem 1.6rem!important;
  letter-spacing:1.5px!important;transition:all .25s!important;
  white-space:nowrap!important;
}
.stTabs [data-baseweb="tab"]:hover{color:#5a3000!important;background:rgba(184,134,11,0.10)!important;}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,rgba(184,134,11,0.22),rgba(184,134,11,0.12))!important;
  color:#3a1800!important;
  border:1.5px solid rgba(184,134,11,0.45)!important;
  box-shadow:0 2px 10px rgba(184,134,11,0.18),inset 0 1px 0 rgba(255,255,255,0.5)!important;
}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.2rem!important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--warm),#fdf3e7)!important;
  border-right:2px solid rgba(184,134,11,0.3)!important;
  box-shadow:4px 0 24px rgba(184,134,11,0.08)!important;
}
[data-testid="stSidebar"] *{font-family:'EB Garamond',serif!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-family:'Cinzel',serif!important;color:#9a6e00!important;letter-spacing:2px!important;}

/* Sidebar toggle button */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button,
button[kind="header"]{
  background:linear-gradient(135deg,#8a6e2f,#c9a84c,#e8c96a,#c9a84c,#8a6e2f)!important;
  border:2px solid rgba(232,201,106,0.9)!important;
  border-radius:12px!important;
  width:44px!important;height:44px!important;
  min-width:44px!important;min-height:44px!important;
  box-shadow:0 4px 16px rgba(184,134,11,0.45),inset 0 1px 0 rgba(255,255,255,0.25)!important;
  transition:all .3s ease!important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover,
button[kind="header"]:hover{transform:scale(1.1)!important;box-shadow:0 6px 24px rgba(184,134,11,0.65)!important;}
[data-testid="stSidebarCollapsedControl"] button svg,
[data-testid="collapsedControl"] button svg,
[data-testid="stSidebarCollapseButton"] button svg,
button[kind="header"] svg{fill:#3a2000!important;stroke:#3a2000!important;width:20px!important;height:20px!important;}

/* ── Sidebar metrics ── */
[data-testid="stMetric"]{
  background:linear-gradient(145deg,var(--cream),#fff6e8)!important;
  border:1px solid var(--gold-bdr)!important;
  border-radius:var(--r)!important;
  padding:1rem 1.2rem!important;
  box-shadow:0 2px 12px rgba(184,134,11,0.08)!important;
  outline:2px solid rgba(184,134,11,0.05)!important;
  outline-offset:2px!important;
}
[data-testid="stMetricValue"]{font-family:'Playfair Display',serif!important;font-size:1.6rem!important;font-weight:700!important;color:#9a6e00!important;}
[data-testid="stMetricLabel"]{color:#7a5a1a!important;font-family:'Cinzel',serif!important;font-size:.58rem!important;letter-spacing:1.5px!important;text-transform:uppercase!important;}
.stProgress>div>div>div{background:linear-gradient(90deg,var(--gold-d),var(--gold-b))!important;border-radius:3px!important;}
[data-testid="stDataFrame"]{border-radius:var(--r)!important;border:1.5px solid var(--gold-bdr)!important;overflow:hidden!important;box-shadow:var(--sw)!important;}
.stRadio [data-baseweb="radio"] div[role="radio"]{border-color:var(--gold-d)!important;}
.stRadio [data-baseweb="radio"] div[aria-checked="true"]{background:var(--gold-d)!important;border-color:var(--gold)!important;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--parchment);}
::-webkit-scrollbar-thumb{background:#c9a84c;border-radius:10px;}

/* ── Divider ── */
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,var(--gold-l),transparent)!important;margin:1.8rem 0!important;}
.stAlert{border-radius:var(--r)!important;font-family:'EB Garamond',serif!important;}

/* ── Team cards ── */
.team-card{background:linear-gradient(145deg,var(--cream),#fff6e8);border:1px solid var(--gold-bdr);border-radius:var(--r);padding:1.4rem 1rem;text-align:center;transition:border-color .3s,transform .3s;outline:2px solid rgba(184,134,11,0.04);outline-offset:3px;}
.team-card:hover{border-color:rgba(184,134,11,0.7);transform:translateY(-4px);box-shadow:var(--sg);}

/* ── Ornate section divider ── */
.ornate-div{display:flex;align-items:center;gap:1rem;margin:1.5rem 0;}
.ornate-div::before,.ornate-div::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--gold-d));}
.ornate-div::after{background:linear-gradient(90deg,var(--gold-d),transparent);}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ABOUT DIALOG
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
        '</div>', unsafe_allow_html=True,
    )
    _d1, _d2 = st.columns(2, gap='medium')
    for _col, _items in [
        (_d1, [
            ('📦 Dataset', '831 scans (Kaggle) · 610 Normal · 221 Parkinson<br>Split: 581 Train · 125 Val · 125 Test<br>WeightedRandomSampler for class balance'),
            ('⚙ Training Config', 'SEED=42 · IMG=224×224 · BS=16 · EPOCHS=30<br>MixUp (α=0.3) · Label Smoothing (ε=0.1)<br>CosineAnnealingLR · Early Stopping (p=7)<br>AMP mixed precision · Grad clip=1.0'),
        ]),
        (_d2, [
            ('📊 Test Results', 'HybridNet_EV: <b style="color:#1a6a38;">100%</b> accuracy<br>F1: 1.000 · Precision: 1.000 · Recall: 1.000<br>AUC: 1.0000 · 5-fold cross-validation'),
            ('🖥 Tech Stack', 'PyTorch · timm (EfficientNet-B4 + ViT-B/16)<br>Cross-Attention Fusion · Streamlit · ReportLab<br>Google Colab · Tesla T4 GPU (15.6 GB VRAM)'),
        ]),
    ]:
        with _col:
            for _t, _body in _items:
                st.markdown(
                    f'<div style="background:rgba(184,134,11,0.06);border:1px solid rgba(184,134,11,0.2);'
                    f'border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.7rem;">'
                    f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:#b8860b;'
                    f'letter-spacing:2px;text-transform:uppercase;margin-bottom:.5rem;">{_t}</div>'
                    f'<div style="font-family:EB Garamond,serif;font-size:.92rem;color:#5a3a0a;line-height:1.7;">{_body}</div>'
                    f'</div>', unsafe_allow_html=True,
                )
    st.markdown(
        '<div style="background:rgba(30,132,73,0.07);border:1px solid rgba(30,132,73,0.22);'
        'border-left:4px solid #1a6a38;border-radius:8px;padding:.9rem 1.2rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.58rem;color:#1a6a38;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.35rem;">🏆 Key Innovation</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.95rem;color:#0a2e18;line-height:1.75;">'
        '<strong>Cross-Attention Fusion</strong> — EfficientNet-B4 spatial feature maps fused with ViT global '
        'patch tokens via multi-head cross-attention, combining local texture and global context '
        'for superior Parkinson\'s detection. Achieves 100% test accuracy with full Grad-CAM explainability.'
        '</div></div>', unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
logo_src  = get_logo_b64('logo.png')
logo_html = (
    f'<img src="{logo_src}" style="width:82px;height:82px;object-fit:contain;'
    f'border-radius:50%;border:2.5px solid rgba(184,134,11,0.6);'
    f'box-shadow:0 0 28px rgba(184,134,11,0.25),0 0 0 5px rgba(184,134,11,0.08);"/>'
    if logo_src else
    '<div style="width:82px;height:82px;background:linear-gradient(145deg,#fdf3dc,#fce8b8);'
    'border:2.5px solid rgba(184,134,11,0.6);border-radius:50%;display:flex;align-items:center;'
    'justify-content:center;font-size:2.2rem;box-shadow:0 0 28px rgba(184,134,11,0.25),'
    '0 0 0 5px rgba(184,134,11,0.08);">🧠</div>'
)

st.markdown(
    '<div style="background:linear-gradient(175deg,#fdf3dc 0%,#fdf8f0 55%,#fdf8f0 100%);'
    'border-bottom:2px solid rgba(184,134,11,0.28);padding:2.8rem 2rem 2.2rem;'
    'display:flex;flex-direction:column;align-items:center;gap:.9rem;'
    'position:relative;overflow:hidden;">'
    # Top double bar
    '<div style="position:absolute;top:0;left:0;right:0;height:4px;'
    'background:linear-gradient(90deg,transparent,#c9a84c,#f0d070,#e8c96a,#c9a84c,transparent);"></div>'
    '<div style="position:absolute;top:5px;left:0;right:0;height:1px;'
    'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
    # Corner ornaments
    '<div style="position:absolute;top:18px;left:24px;font-size:.9rem;color:rgba(184,134,11,0.4);font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'
    '<div style="position:absolute;top:18px;right:24px;font-size:.9rem;color:rgba(184,134,11,0.4);font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'
    + logo_html +
    '<div style="font-family:Cinzel,serif;font-size:2.8rem;font-weight:900;color:#2c1a00;'
    'letter-spacing:8px;line-height:1;text-align:center;'
    'text-shadow:0 2px 14px rgba(184,134,11,0.18);">'
    'NEUROSCAN&nbsp;<span style="color:#b8860b;">AI</span></div>'
    '<div style="font-family:Cormorant Garamond,serif;font-size:1rem;font-style:italic;'
    'color:#9a7030;letter-spacing:4px;text-align:center;">'
    "Parkinson\u2019s Detection \u00b7 Brain MRI \u00b7 Deep Learning</div>"
    # Model badge — updated for single model
    '<div style="background:linear-gradient(135deg,rgba(184,134,11,0.14),rgba(184,134,11,0.07));'
    'border:1.5px solid rgba(184,134,11,0.4);border-radius:8px;'
    'padding:.65rem 2rem;margin-top:.1rem;position:relative;overflow:hidden;">'
    '<div style="font-family:Cinzel,serif;font-size:.9rem;font-weight:700;'
    'color:#7a4f00;letter-spacing:2.5px;text-align:center;text-transform:uppercase;">'
    "HybridNet\u2009EV \u00b7 EfficientNet-B4 + ViT Cross-Attention \u00b7 100% Accuracy"
    '</div></div>'
    '<div style="display:flex;align-items:center;gap:1rem;width:60%;max-width:380px;">'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,#c9a84c);"></div>'
    '<span style="color:#b8860b;font-size:.7rem;">✦</span>'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,#c9a84c,transparent);"></div>'
    '</div>'
    '<div style="display:flex;gap:.8rem;flex-wrap:wrap;justify-content:center;">'
    '<span style="background:rgba(184,134,11,0.09);border:1.5px solid rgba(184,134,11,0.28);'
    'border-radius:20px;padding:.28rem .9rem;font-family:Cinzel,serif;font-size:.62rem;color:#7a5000;letter-spacing:1px;">⚙ EfficientNet-B4 + ViT-B/16</span>'
    '<span style="background:rgba(184,134,11,0.09);border:1.5px solid rgba(184,134,11,0.28);'
    'border-radius:20px;padding:.28rem .9rem;font-family:Cinzel,serif;font-size:.62rem;color:#7a5000;letter-spacing:1px;">◈ 100% Test Accuracy</span>'
    '<span style="background:rgba(184,134,11,0.09);border:1.5px solid rgba(184,134,11,0.28);'
    'border-radius:20px;padding:.28rem .9rem;font-family:Cinzel,serif;font-size:.62rem;color:#7a5000;letter-spacing:1px;">✦ Grad-CAM XAI</span>'
    '<span style="background:rgba(184,134,11,0.09);border:1.5px solid rgba(184,134,11,0.28);'
    'border-radius:20px;padding:.28rem .9rem;font-family:Cinzel,serif;font-size:.62rem;color:#7a5000;letter-spacing:1px;">⬡ Batch Analysis</span>'
    '</div>'
    '<div style="display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;justify-content:center;">'
    '<div style="height:1px;width:60px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.4));"></div>'
    '<div style="background:linear-gradient(135deg,rgba(184,134,11,0.10),rgba(184,134,11,0.05));'
    'border:1.5px solid rgba(184,134,11,0.3);border-radius:20px;padding:.3rem 1.2rem;">'
    '<span style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:.88rem;color:#9a7030;">'
    'Under the Esteemed Guidance of&nbsp;</span>'
    '<span style="font-family:Cinzel,serif;font-size:.85rem;font-weight:700;color:#7a4f00;">Mr. N P U V S N Pavan Kumar</span>'
    '<span style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:.82rem;color:#b8860b;">&nbsp;· Asst. Professor, ECE</span>'
    '</div>'
    '<div style="height:1px;width:60px;background:linear-gradient(90deg,rgba(184,134,11,0.4),transparent);"></div>'
    '</div>'
    '<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
    'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
    '</div>',
    unsafe_allow_html=True,
)

_b1, _b2, _b3 = st.columns([4, 2, 4])
with _b2:
    if st.button('📋 About Project', key='hero_about_btn'):
        _show_project_dialog()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — simplified (single model info)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="background:linear-gradient(135deg,#fdf3dc,#fdf8f0);'
        'border-bottom:2px solid rgba(184,134,11,0.3);border-right:0;'
        'padding:1.2rem 1rem 1rem;margin:-1rem -1rem .8rem;text-align:center;">'
        '<div style="font-size:1.8rem;margin-bottom:.3rem;">🧠</div>'
        '<div style="font-family:Cinzel,serif;font-size:1rem;font-weight:900;color:#7a4f00;letter-spacing:4px;">⚕ NEUROSCAN AI</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.78rem;font-style:italic;color:#9a7030;margin-top:.2rem;">Model Control Panel</div>'
        '<div style="height:2px;background:linear-gradient(90deg,transparent,#c9a84c,transparent);margin:.6rem 0 0;"></div></div>',
        unsafe_allow_html=True,
    )

    # Active model display
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(30,132,73,0.10),rgba(30,132,73,0.05));'
        'border:1.5px solid rgba(30,132,73,0.35);border-left:4px solid #1a6a38;'
        'border-radius:10px;padding:.8rem 1rem;margin:.5rem 0 .8rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.55rem;color:#1a6a38;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.3rem;">✦ Active Model</div>'
        '<div style="font-family:Playfair Display,serif;font-size:1rem;font-weight:700;color:#0a3a20;">HybridNet EV</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.85rem;color:#2a6a40;margin-top:.1rem;">EfficientNet-B4 + ViT-B/16</div>'
        '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#4CAF50;margin-top:.3rem;letter-spacing:1px;">✓ Test Accuracy: 100% &nbsp;|&nbsp; AUC: 1.000</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),transparent);margin:.5rem 0;"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.62rem;color:#b8860b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">◈ Performance</div>',
        unsafe_allow_html=True,
    )
    _c1, _c2 = st.columns(2)
    with _c1: st.metric('Accuracy', '100%')
    with _c2: st.metric('AUC',      '1.000')
    _c3, _c4 = st.columns(2)
    with _c3: st.metric('F1-Score', '1.000')
    with _c4: st.metric('Recall',   '100%')

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),transparent);margin:.5rem 0;"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.62rem;color:#b8860b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">⚗ Architecture</div>'
        '<div style="background:rgba(184,134,11,0.05);border:1px solid rgba(184,134,11,0.15);'
        'border-radius:8px;padding:.7rem .9rem;font-family:EB Garamond,serif;font-size:.88rem;color:#5a3a0a;line-height:1.9;">'
        '🔹 EfficientNet-B4 backbone<br>'
        '🔹 ViT-B/16 transformer<br>'
        '🔹 Cross-Attention Fusion<br>'
        '🔹 105M total parameters<br>'
        '🔹 44.3M trainable params'
        '</div>',
        unsafe_allow_html=True,
    )

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
    st.warning("⚠️ Research/academic use only. Not for clinical diagnosis.")
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.55rem;color:#9a7030;'
        'letter-spacing:1px;text-align:center;margin-top:.4rem;">'
        '💡 Single model · CPU-optimised · Memory safe'
        '</div>', unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_scan, tab_batch, tab_about = st.tabs([
    '🧠  Single MRI Analysis',
    '📦  Batch Analysis',
    '👥  About Us',
])

# ─── TAB 1: Single Scan ───────────────────────────────────────────────────────
with tab_scan:
    st.markdown(
        '<div class="sec-head">✦ HybridNet EV — EfficientNet-B4 + ViT Cross-Attention</div>',
        unsafe_allow_html=True,
    )

    # Model info card
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(30,132,73,0.07),rgba(184,134,11,0.05));'
        'border:1.5px solid rgba(184,134,11,0.3);border-radius:12px;'
        'padding:1rem 1.5rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:1.5rem;'
        'box-shadow:0 2px 14px rgba(184,134,11,0.08);">'
        '<div style="font-size:2.2rem;">🔀</div>'
        '<div style="flex:1;">'
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#b8860b;letter-spacing:2px;text-transform:uppercase;margin-bottom:.2rem;">Active Model</div>'
        '<div style="font-family:Playfair Display,serif;font-size:1.1rem;font-weight:700;color:#2c1a00;">HybridNet_EV &nbsp;—&nbsp; EfficientNet-B4 + ViT Cross-Attention Fusion</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#6b4c11;margin-top:.2rem;">105M params · 44.3M trainable · Grad-CAM enabled · CPU-optimised inference</div>'
        '</div>'
        '<div style="text-align:center;background:rgba(30,132,73,0.10);border:1px solid rgba(30,132,73,0.3);'
        'border-radius:8px;padding:.5rem .9rem;flex-shrink:0;">'
        '<div style="font-family:Playfair Display,serif;font-size:1.5rem;font-weight:900;color:#1a6a38;">100%</div>'
        '<div style="font-family:Cinzel,serif;font-size:.55rem;color:#2a8a50;letter-spacing:1px;">TEST ACC</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.2),transparent);margin:.5rem 0 1rem;"></div>', unsafe_allow_html=True)

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

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.2),transparent);margin:1.5rem 0;"></div>', unsafe_allow_html=True)
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
                with st.spinner('Running HybridNet_EV analysis…'):
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
                        download_model(MODEL_FILENAME)
                        model, classes, device = load_model_cached(MODEL_FILENAME)
                        result = predict(model, device, classes, image)
                        st.session_state.prediction_result = result
                        st.session_state.prediction_made   = True
                        st.success('Analysis complete!')
                        st.balloons()
                        st.rerun()
                    except Exception as exc:
                        st.error(f'Error during analysis: {exc}')

    if st.session_state.prediction_made:
        r         = st.session_state.prediction_result
        is_normal = not r.get('is_parkinson', False)
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
                f'</span></div></div>', unsafe_allow_html=True,
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
                f'<div style="font-family:EB Garamond,serif;font-size:.82rem;'
                f'color:#a89060;margin-top:.35rem;line-height:1.4;">HybridNet EV<br>'
                f'<span style="font-size:.75rem;color:#c9a84c;">EfficientNet-B4 + ViT</span></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>✦ Normal</span><span>{r["normal_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-n" style="width:{r["normal_prob"]:.1f}%"></div></div></div>'
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>⚠ Parkinson\'s</span><span>{r["parkinson_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-p" style="width:{r["parkinson_prob"]:.1f}%"></div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if r.get('cam_overlay') is not None:
            st.markdown('<div class="sec-head">✦ Grad-CAM Explainability</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<p style="font-family:EB Garamond,serif;font-size:.95rem;color:#a89060;'
                'margin-bottom:1.2rem;line-height:1.7;">'
                "Grad-CAM highlights brain regions driving the model's decision. "
                '<strong style="color:#c9a84c;">Warmer colours (red/yellow)</strong> indicate higher neural attention.</p>',
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
                '<div class="hm-legend"><span>Low</span><div class="hm-bar"></div><span>High</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with report_col:
        if st.session_state.prediction_made:
            if st.button('📜 Generate PDF Report'):
                with st.spinner('Building report…'):
                    try:
                        pdf_bytes = build_pdf(
                            st.session_state.patient_data,
                            st.session_state.prediction_result,
                        )
                        p     = st.session_state.patient_data
                        fname = f"NeuroScan_{p['patient_id']}_{datetime.now().strftime('%Y%m%d')}.pdf"
                        st.download_button('⬇ Download PDF Report', data=pdf_bytes, file_name=fname, mime='application/pdf')
                        st.success('PDF ready!')
                    except Exception as exc:
                        st.error(f'PDF error: {exc}')
        else:
            st.info('Run an analysis first to generate a report.')


# ─── TAB 2: Batch ─────────────────────────────────────────────────────────────
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
            try:
                download_model(MODEL_FILENAME)
                model, classes, device = load_model_cached(MODEL_FILENAME)
            except Exception as exc:
                st.error(f'Failed to load model: {exc}')
                st.stop()

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
                    res = predict(model, device, classes, Image.open(f))
                    res['filename'] = f.name
                    batch_results.append(res)
                except Exception as exc:
                    st.warning(f'Skipped {f.name}: {exc}')
                prog.progress((i+1) / len(batch_files))
                gc.collect()

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
                    colors=['#3dbb6e', '#e05555'],
                    startangle=90,
                    wedgeprops=dict(edgecolor='#2c1a00', linewidth=2),
                    textprops=dict(color='#2c1a00', fontsize=10),
                )
                for at in autotexts:
                    at.set_color('#ffffff'); at.set_fontweight('bold')
            ax.set_title('Scan Distribution', color='#7a5a1a', fontsize=10, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            gc.collect()

        with tb_col:
            df = pd.DataFrame([{
                'File':          r['filename'],
                'Prediction':    r['prediction'],
                'Confidence':    f"{r['confidence']:.1f}%",
                'Normal %':      f"{r['normal_prob']:.1f}%",
                "Parkinson's %": f"{r['parkinson_prob']:.1f}%",
                'Risk':          r['risk_level'],
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
            is_n = not r.get('is_parkinson', False)
            col  = '#3dbb6e' if is_n else '#e05555'
            icon = '✦' if is_n else '⚠'
            rc1, rc2, rc3, rc4 = st.columns([1, 2, 1, 1])
            with rc1:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(r['image'], use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(
                    f'<p style="font-family:Cinzel,serif;font-size:.62rem;color:#9a7030;margin-bottom:.3rem;letter-spacing:1px;">{r["filename"]}</p>'
                    f'<p style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;color:{col};margin:0;">{icon} {r["prediction"]}</p>'
                    f'<p style="font-family:Cinzel,serif;font-size:.6rem;color:#9a7030;margin-top:.35rem;letter-spacing:1px;">{r["model_name"]} · {r["timestamp"]}</p>',
                    unsafe_allow_html=True,
                )
            with rc3:
                st.metric('Confidence', f"{r['confidence']:.1f}%")
                st.metric('Normal %',   f"{r['normal_prob']:.1f}%")
            with rc4:
                if r.get('cam_overlay') is not None:
                    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                    st.image(r['cam_overlay'], use_column_width=True)
                    st.markdown('<div class="img-caption">Grad-CAM</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.15),transparent);margin:.8rem 0;"></div>',
                unsafe_allow_html=True,
            )


# ─── TAB 3: About ─────────────────────────────────────────────────────────────
with tab_about:
    college_logo = get_logo_b64('bvcr.jpg')
    clg_html = (
        '<img src="' + college_logo + '" style="width:110px;height:110px;object-fit:contain;'
        'margin-bottom:1.2rem;border:2.5px solid rgba(184,134,11,0.5);border-radius:10px;'
        'box-shadow:0 6px 28px rgba(184,134,11,0.2);"/>'
        if college_logo else
        '<div style="font-size:4rem;margin-bottom:1.2rem;">🏛</div>'
    )

    st.markdown(
        '<div style="background:linear-gradient(160deg,#fdf3dc 0%,#fdf8f0 50%,#fce8b0 100%);'
        'border:1.5px solid rgba(184,134,11,0.32);border-radius:16px;'
        'padding:3rem 2.5rem 2.5rem;margin-bottom:2rem;text-align:center;'
        'position:relative;overflow:hidden;outline:3px solid rgba(184,134,11,0.06);outline-offset:4px;">'
        '<div style="position:absolute;top:0;left:0;right:0;height:4px;'
        'background:linear-gradient(90deg,transparent,#c9a84c,#f0d070,#c9a84c,transparent);"></div>'
        '<div style="position:absolute;top:5px;left:0;right:0;height:1px;'
        'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
        + clg_html +
        '<div style="font-family:Cinzel,serif;font-size:1.9rem;font-weight:900;color:#2c1a00;letter-spacing:4px;margin-bottom:.4rem;">BVC College of Engineering</div>'
        '<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:1.05rem;color:#b8860b;letter-spacing:3px;margin-bottom:.3rem;">Rajahmundry, Andhra Pradesh</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#9a7030;margin-bottom:1rem;">'
        '<a href="https://bvcr.edu.in" target="_blank" style="color:#b8860b;text-decoration:none;border-bottom:1px solid rgba(184,134,11,0.4);">bvcr.edu.in</a></div>'
        '<div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.2rem;">'
        '<span style="background:rgba(184,134,11,0.1);border:1.5px solid rgba(184,134,11,0.35);border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;color:#7a4f00;letter-spacing:1.5px;font-weight:600;">🎓 Autonomous</span>'
        '<span style="background:rgba(30,132,73,0.08);border:1.5px solid rgba(30,132,73,0.3);border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;color:#1a6a38;letter-spacing:1.5px;font-weight:600;">✓ NAAC A Grade</span>'
        '<span style="background:rgba(0,100,180,0.07);border:1.5px solid rgba(0,100,180,0.22);border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;color:#004a90;letter-spacing:1.5px;font-weight:600;">⚙ AICTE Approved</span>'
        '<span style="background:rgba(140,40,40,0.07);border:1.5px solid rgba(140,40,40,0.22);border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;color:#7a1a1a;letter-spacing:1.5px;font-weight:600;">📜 Affiliated to JNTUK</span>'
        '</div>'
        '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a5a1a;line-height:1.8;max-width:700px;margin:0 auto;">'
        'BVC College of Engineering, Rajahmundry is a premier autonomous institution in Andhra Pradesh '
        'permanently affiliated to JNTUK. Accredited by NAAC with A Grade and approved by AICTE.'
        '</div>'
        '<div style="position:absolute;bottom:0;left:0;right:0;height:2px;'
        'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # About Project
    st.markdown('<div class="sec-head" style="margin-top:1rem;">✦ About the Project</div>', unsafe_allow_html=True)
    _pc1, _pc2 = st.columns([3, 2], gap='large')
    with _pc1:
        st.markdown(
            '<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
            'border:1.5px solid rgba(184,134,11,0.25);border-radius:12px;'
            'padding:2rem;box-shadow:0 2px 18px rgba(184,134,11,0.09);'
            'outline:3px solid rgba(184,134,11,0.05);outline-offset:3px;">'
            '<div style="font-family:Cinzel,serif;font-size:.68rem;color:#b8860b;letter-spacing:2px;text-transform:uppercase;margin-bottom:.8rem;">⚗ Project Overview</div>'
            '<div style="font-family:Playfair Display,serif;font-size:1.35rem;font-weight:700;color:#2c1a00;margin-bottom:.9rem;line-height:1.3;">'
            "Parkinson's Disease Detection Using Deep Learning on Brain MRI</div>"
            '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#5a3a0a;line-height:1.9;margin-bottom:1.2rem;">'
            'This B.Tech Final Year Project (2025-26) at the Dept. of ECE, BVC College of Engineering, Rajahmundry '
            'applies <strong style="color:#8a5e00;">Hybrid Deep Learning</strong> to classify brain MRI scans for early '
            "detection of Parkinson's Disease. The deployed model is "
            '<strong style="color:#8a5e00;">HybridNet_EV</strong> — a novel architecture fusing '
            '<strong style="color:#8a5e00;">EfficientNet-B4</strong> spatial features with '
            '<strong style="color:#8a5e00;">ViT-B/16</strong> global patch tokens via '
            '<strong style="color:#8a5e00;">Multi-Head Cross-Attention Fusion</strong>. '
            'Trained on the <em>irfansheriff/parkinsons-brain-mri-dataset</em> (831 scans: 610 Normal, 221 Parkinson) '
            'achieving <strong style="color:#1a6a38;">100% test accuracy</strong> with full Grad-CAM explainability.'
            '</div>'
            '<div style="display:flex;gap:.7rem;flex-wrap:wrap;">'
            '<span style="background:rgba(184,134,11,0.09);border:1.5px solid rgba(184,134,11,0.28);border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;color:#7a4f00;letter-spacing:1px;">🔀 EfficientNet-B4 + ViT</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1.5px solid rgba(184,134,11,0.28);border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;color:#7a4f00;letter-spacing:1px;">⚡ Cross-Attention Fusion</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1.5px solid rgba(184,134,11,0.28);border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;color:#7a4f00;letter-spacing:1px;">📊 Grad-CAM XAI</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1.5px solid rgba(184,134,11,0.28);border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;color:#7a4f00;letter-spacing:1px;">🔬 Brain MRI Analysis</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with _pc2:
        st.markdown(
            '<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
            'border:1.5px solid rgba(184,134,11,0.25);border-radius:12px;'
            'padding:2rem;box-shadow:0 2px 18px rgba(184,134,11,0.09);">'
            '<div style="font-family:Cinzel,serif;font-size:.68rem;color:#b8860b;letter-spacing:2px;text-transform:uppercase;margin-bottom:1rem;">📋 Project Specs</div>'
            '<div style="display:flex;flex-direction:column;gap:.75rem;">'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">📦</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;letter-spacing:1px;font-weight:700;">Dataset</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">831 scans (Kaggle) · 610 Normal · 221 PD<br>581 Train · 125 Val · 125 Test</div></div></div>'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">🤖</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;letter-spacing:1px;font-weight:700;">Deployed Model</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">HybridNet_EV<br>EfficientNet-B4 + ViT-B/16<br>Cross-Attention Fusion</div></div></div>'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">⚙</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;letter-spacing:1px;font-weight:700;">Training Config</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">LR=5e-5 · BS=16 · EPOCHS=30<br>MixUp(α=0.3) · AdamW · CosineAnnealingLR</div></div></div>'

            '<div style="display:flex;align-items:flex-start;gap:.9rem;padding:.7rem;background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.1rem;flex-shrink:0;">🏆</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;letter-spacing:1px;font-weight:700;">Results</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">Accuracy: 100% · F1: 1.000<br>AUC: 1.0000 · Recall: 100%</div></div></div>'

            '</div></div>',
            unsafe_allow_html=True,
        )

    # Team section
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;">✦ Who We Are</div>', unsafe_allow_html=True)

    # Guide banner
    st.markdown(
        '<div style="background:linear-gradient(135deg,#fdf3dc 0%,#fceedd 50%,#fdf3dc 100%);'
        'border:2px solid rgba(184,134,11,0.45);border-radius:16px;'
        'padding:1.6rem 2.5rem;margin-bottom:1.8rem;position:relative;overflow:hidden;'
        'outline:3px solid rgba(184,134,11,0.07);outline-offset:4px;">'
        '<div style="position:absolute;top:0;left:0;right:0;height:4px;'
        'background:linear-gradient(90deg,transparent,#c9a84c,#e8c96a,#c9a84c,transparent);"></div>'
        '<div style="display:flex;align-items:center;gap:2rem;flex-wrap:wrap;">'
        '<div style="font-size:3rem;">🎓</div>'
        '<div style="flex:1;">'
        '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#b8860b;letter-spacing:3px;text-transform:uppercase;margin-bottom:.3rem;">⭐ Under the Esteemed Guidance of</div>'
        '<div style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;color:#2c1a00;line-height:1.2;margin-bottom:.25rem;">Mr. N P U V S N Pavan Kumar, M.Tech</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.95rem;font-style:italic;color:#8a5e00;">Assistant Professor · Dept. of ECE · Deputy CoE III · BVC College of Engineering, Rajahmundry</div>'
        '</div>'
        '<div style="background:linear-gradient(135deg,#7a4f00,#b8860b);border-radius:10px;padding:.8rem 1.4rem;text-align:center;flex-shrink:0;">'
        '<div style="font-family:Cinzel,serif;font-size:.6rem;color:rgba(255,255,255,0.7);letter-spacing:2px;text-transform:uppercase;margin-bottom:.2rem;">Project Guide</div>'
        '<div style="font-size:1.6rem;">🏆</div>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    # Team members
    _team = [
        {'roll':'236M5A0408','name':'G Srinivasu',      'role':'AI Model Architect & Backend',  'icon':'🤖','c1':'#1a3a2a','c2':'#2d6a4f','acc':'#4ade80','tags':['PyTorch','Model Training','Backend'],'desc':'Designed and trained HybridNet_EV including the Cross-Attention Fusion architecture. Led backend integration and model optimization.'},
        {'roll':'226M1A0460','name':'S Anusha Devi',    'role':'UI/UX Designer & Frontend',     'icon':'🎨','c1':'#2a1a3a','c2':'#6a2d8f','acc':'#c084fc','tags':['Streamlit','CSS Design','UX'],      'desc':'Crafted the royal gold aesthetic UI including hero banner, interactive tabs, and PDF report layout.'},
        {'roll':'226M1A0473','name':'V V Siva Vardhan', 'role':'Data Engineer & QA',            'icon':'📊','c1':'#1a2a3a','c2':'#2d4f8a','acc':'#60a5fa','tags':['Data Pipeline','Augmentation','Testing'],'desc':'Managed the full data pipeline including augmentation strategy (MixUp, ColorJitter) and model evaluation.'},
        {'roll':'236M5A0415','name':'N L Sandeep',      'role':'Documentation & Integration',   'icon':'📝','c1':'#3a2a1a','c2':'#8a4f2d','acc':'#fb923c','tags':['PDF Reports','Integration','Docs'], 'desc':'Built the gold PDF report system using ReportLab and handled system integration documentation.'},
    ]

    _tc1, _tc2 = st.columns(2, gap='large')
    for _col, _m in zip([_tc1, _tc2], _team[:2]):
        with _col:
            _tags = ''.join(f'<span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:20px;padding:.2rem .75rem;font-family:Cinzel,serif;font-size:.56rem;color:rgba(255,255,255,0.85);letter-spacing:.8px;">{t}</span>' for t in _m['tags'])
            st.markdown(
                f'<div style="background:linear-gradient(145deg,{_m["c1"]},{_m["c2"]});'
                f'border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:1.8rem 1.6rem;'
                f'position:relative;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.25);">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,{_m["acc"]},transparent);"></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">'
                f'<div style="width:56px;height:56px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.8rem;">{_m["icon"]}</div>'
                f'<div style="background:rgba(255,255,255,0.08);border-radius:6px;padding:.2rem .6rem;font-family:Cinzel,serif;font-size:.55rem;color:{_m["acc"]};letter-spacing:1px;">{_m["roll"]}</div>'
                f'</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.25rem;font-weight:700;color:#ffffff;margin-bottom:.2rem;">{_m["name"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.62rem;color:{_m["acc"]};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.8rem;">{_m["role"]}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.9rem;color:rgba(255,255,255,0.75);line-height:1.7;margin-bottom:1rem;">{_m["desc"]}</div>'
                f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;">{_tags}</div>'
                f'</div>', unsafe_allow_html=True,
            )

    st.markdown('<div style="height:.8rem;"></div>', unsafe_allow_html=True)
    _tc3, _tc4 = st.columns(2, gap='large')
    for _col, _m in zip([_tc3, _tc4], _team[2:]):
        with _col:
            _tags = ''.join(f'<span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:20px;padding:.2rem .75rem;font-family:Cinzel,serif;font-size:.56rem;color:rgba(255,255,255,0.85);letter-spacing:.8px;">{t}</span>' for t in _m['tags'])
            st.markdown(
                f'<div style="background:linear-gradient(145deg,{_m["c1"]},{_m["c2"]});'
                f'border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:1.8rem 1.6rem;'
                f'position:relative;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.25);">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,{_m["acc"]},transparent);"></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">'
                f'<div style="width:56px;height:56px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.8rem;">{_m["icon"]}</div>'
                f'<div style="background:rgba(255,255,255,0.08);border-radius:6px;padding:.2rem .6rem;font-family:Cinzel,serif;font-size:.55rem;color:{_m["acc"]};letter-spacing:1px;">{_m["roll"]}</div>'
                f'</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.25rem;font-weight:700;color:#ffffff;margin-bottom:.2rem;">{_m["name"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.62rem;color:{_m["acc"]};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.8rem;">{_m["role"]}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.9rem;color:rgba(255,255,255,0.75);line-height:1.7;margin-bottom:1rem;">{_m["desc"]}</div>'
                f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;">{_tags}</div>'
                f'</div>', unsafe_allow_html=True,
            )

    # Faculty guidance
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;">✦ Project Guidance</div>', unsafe_allow_html=True)
    _g1, _g2, _g3 = st.columns(3, gap='medium')
    _guides = [
        (_g1,'Project Guide','👨‍🏫','#b8860b','#fdf3dc','Mr. N P U V S N Pavan Kumar, M.Tech',['Assistant Professor','Dept. of ECE','Deputy CoE III'],'Primary guide who mentored the team throughout model design, training strategy, and deployment.'),
        (_g2,'Project Coordinator','📋','#1a6a38','#f0fdf4','Mr. K Anji Babu, M.Tech',['Assistant Professor','Dept. of ECE'],'Coordinated project milestones, review sessions, and facilitated academic compliance.'),
        (_g3,'Head of Department','👨‍💼','#7a1a1a','#fff1f1','Dr. S A Vara Prasad, Ph.D, M.Tech',['Professor & HOD, ECE','Chairman BoS'],'Provided departmental leadership and institutional support for this B.Tech Final Year Project.'),
    ]
    for _col, _rl, _icon, _rc, _bg, _name, _tags, _desc in _guides:
        _th = ''.join(f'<span style="background:rgba(0,0,0,0.05);border:1px solid rgba(0,0,0,0.1);border-radius:4px;padding:.2rem .7rem;font-family:Cinzel,serif;font-size:.58rem;color:{_rc};letter-spacing:.8px;font-weight:600;">{t}</span> ' for t in _tags)
        with _col:
            st.markdown(
                f'<div style="background:{_bg};border:1.5px solid rgba(0,0,0,0.08);border-radius:14px;padding:1.8rem 1.4rem;text-align:center;position:relative;box-shadow:0 3px 20px rgba(0,0,0,0.07);outline:2px solid rgba(0,0,0,0.03);outline-offset:3px;">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,transparent,{_rc},transparent);border-radius:14px 14px 0 0;"></div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:{_rc};letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.8rem;font-weight:700;">⭐ {_rl}</div>'
                f'<div style="width:64px;height:64px;background:linear-gradient(135deg,{_rc}22,{_rc}11);border:2px solid {_rc}44;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.8rem;margin:0 auto .8rem;">{_icon}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1rem;font-weight:700;color:#2c1a00;margin-bottom:.6rem;line-height:1.4;">{_name}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:.3rem;justify-content:center;margin-bottom:.8rem;">{_th}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.88rem;color:#5a3a0a;line-height:1.65;font-style:italic;">{_desc}</div>'
                f'</div>', unsafe_allow_html=True,
            )

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(192,57,43,0.05),rgba(192,57,43,0.02));'
        'border:1.5px solid rgba(192,57,43,0.25);border-left:4px solid #c0392b;'
        'border-radius:10px;padding:1.2rem 1.8rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#c0392b;letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">⚕ Medical Disclaimer</div>'
        '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a1a1a;line-height:1.7;">'
        'This project is developed <strong>for academic and research purposes only</strong>. '
        'It is a B.Tech Final Year Project at BVC College of Engineering, Rajahmundry. '
        "The AI system is not a certified medical device and must not replace professional clinical diagnosis. "
        'Always consult a qualified neurologist.'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ── ROYAL FOOTER ──────────────────────────────────────────────────────────────
st.markdown(
    '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);margin:2rem 0 0;"></div>'
    '<div style="text-align:center;padding:1.6rem;background:linear-gradient(180deg,#fdf8f0,#fdf3dc);">'
    '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#9a7030;letter-spacing:3px;text-transform:uppercase;margin-bottom:.5rem;">'
    'Research &amp; Educational Use Only · Not for Clinical Diagnosis'
    '</div>'
    '<div style="display:flex;align-items:center;gap:1rem;justify-content:center;margin-bottom:.5rem;">'
    '<div style="flex:1;max-width:80px;height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.2));"></div>'
    '<span style="color:rgba(201,168,76,0.4);font-size:.6rem;">✦</span>'
    '<div style="flex:1;max-width:80px;height:1px;background:linear-gradient(90deg,rgba(201,168,76,0.2),transparent);"></div>'
    '</div>'
    '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#9a7030;">'
    'NeuroScan AI · HybridNet_EV · EfficientNet-B4 + ViT · Grad-CAM · PyTorch · Streamlit'
    '</div></div>',
    unsafe_allow_html=True,
)
