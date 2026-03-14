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
import torchvision.models as models
import torchvision.models as tv_models
import base64
import os
import gdown
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import timm

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — point to your Drive .pth file
# ══════════════════════════════════════════════════════════════════════════════
GDRIVE_FILE_ID = "1p2uIwGMGI06iPyuHYeqUAw2EtBN53vvq"   # ← your existing ID
MODEL_PATH     = "hybridnet_rv_best.pth"                 # ← change to whichever .pth you want


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES  (must match training notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared building block ────────────────────────────────────────────────────
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
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.25), nn.Linear(64, nc)
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


# ── HybridNet (ResNet50+ViT  or  EfficientNet-B4+ViT) ────────────────────────
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
            nn.Dropout(dropout/2), nn.Linear(256, nc)
        )

    def _cnn_feat(self, x):
        if self.backbone_type == 'resnet50':
            return self.cnn(x)
        return self.cnn(x)[-1]

    def forward(self, x):
        cf  = self._cnn_feat(x)
        ct  = self.cnn_proj(cf.flatten(2).transpose(1, 2))
        vt  = self.vit_proj(self.vit.forward_features(x)[:, 1:, :])
        fused = self.fusion(ct, vt)
        return self.head(self.gap(fused.transpose(1, 2)).squeeze(-1))

    def get_gradcam_layer(self):
        if self.backbone_type == 'resnet50':
            return self.cnn[-1][-1].conv3
        return self.cnn.blocks[-1][-1].conv_pwl


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_model() -> None:
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model — first time only, please wait…"):
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                    MODEL_PATH, quiet=False,
                )
                st.success("Model downloaded!")
            except Exception as exc:
                st.error(f"Download failed: {exc}")
                st.info("Make sure the Google Drive file is shared as 'Anyone with the link'.")
                st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL  (auto-detects architecture from checkpoint)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck     = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    nc      = ck["config"]["num_classes"]
    mn      = ck["model_name"]
    bt      = ck.get("backbone_type", None)
    classes = ck.get("classes", [str(i) for i in range(nc)])

    if mn == "ResNet50":
        m   = tv_models.resnet50(weights=None)
        inf = m.fc.in_features
        m.fc = nn.Sequential(
            nn.BatchNorm1d(inf), nn.Dropout(0.45),
            nn.Linear(inf, 512), nn.ReLU(True),
            nn.BatchNorm1d(512), nn.Dropout(0.30),
            nn.Linear(512, nc)
        )
    elif mn == "EfficientNet-B4":
        m = timm.create_model("efficientnet_b4", pretrained=False, num_classes=nc)
    elif mn == "ViT-B16":
        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=nc)
    elif mn == "MiniSegNet":
        m = MiniSegNet(nc)
    elif mn in ("HybridNet_RV", "HybridNet_EV"):
        backbone = bt or ("resnet50" if "RV" in mn else "efficientnet_b4")
        m = HybridNet(nc, backbone_type=backbone)
    else:
        st.error(f"Unknown model name in checkpoint: {mn}")
        st.stop()

    m.load_state_dict(ck["model_state_dict"])
    return m.to(device).eval(), classes, mn, device


# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._acts = self._grads = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0]))

    def generate(self, img_tensor, class_idx, device):
        img_tensor = img_tensor.clone().to(device).requires_grad_(True)
        with torch.enable_grad():
            out = self.model(img_tensor)
            self.model.zero_grad()
            out[0, class_idx].backward(retain_graph=True)

        if self._grads is None or self._acts is None:
            return np.zeros((224, 224))

        weights = F.relu(self._grads).mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self._acts).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, (224, 224), mode="bicubic", align_corners=False)
        cam     = cam.squeeze().cpu().detach().numpy()
        lo, hi  = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            return np.zeros((224, 224))
        return np.power(cam, 1.8)


def get_gradcam_layer(model, mn):
    if mn == "ResNet50":
        return model.layer4[-1].conv3
    elif mn == "EfficientNet-B4":
        return model.conv_head
    elif mn == "MiniSegNet":
        return model.enc3[1].net[0]
    elif mn in ("HybridNet_RV", "HybridNet_EV"):
        return model.get_gradcam_layer()
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
    img_rgb    = pil_image.convert("RGB")
    img_tensor = TRANSFORM(img_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out        = model(img_tensor)
        probs      = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        class_idx  = pred.item()

    # GradCAM
    gc_layer = get_gradcam_layer(model, mn)
    overlay = heatmap = None
    if gc_layer is not None:
        try:
            gc      = GradCAM(model, gc_layer)
            cam_map = gc.generate(img_tensor, class_idx, device)
            overlay, heatmap = apply_colormap(img_rgb, cam_map)
        except Exception:
            pass

    confidence_pct = float(conf.item() * 100)
    normal_prob    = float(probs[0][0].item() * 100)
    parkinson_prob = float(probs[0][1].item() * 100)

    if class_idx == 0:
        risk = "Low"
    else:
        risk = "High" if confidence_pct >= 85 else "Moderate"

    return {
        "prediction":     classes[class_idx],
        "class_idx":      class_idx,
        "confidence":     confidence_pct,
        "normal_prob":    normal_prob,
        "parkinson_prob": parkinson_prob,
        "risk_level":     risk,
        "cam_overlay":    overlay,
        "cam_heatmap":    heatmap,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image":          img_rgb,
        "model_name":     mn,
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

    title_s = ParagraphStyle("T", parent=styles["Heading1"], fontSize=20,
        textColor=colors.HexColor("#0d2b5e"), spaceAfter=6,
        alignment=TA_CENTER, fontName="Helvetica-Bold")
    sub_s   = ParagraphStyle("S", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#3a7bd5"), spaceAfter=20, alignment=TA_CENTER)
    head_s  = ParagraphStyle("H", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#0d2b5e"), spaceAfter=8,
        spaceBefore=14, fontName="Helvetica-Bold")
    body_s  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#1a1a2e"), leading=14)
    warn_s  = ParagraphStyle("W", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#7f1d1d"), leading=13,
        backColor=colors.HexColor("#fff1f1"), borderPad=6)

    story.append(Paragraph("NeuroScan AI", title_s))
    story.append(Paragraph("Parkinson's Disease MRI Analysis Report", sub_s))

    def tbl(rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#0d2b5e")),
            ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,0), 11),
            ("BOTTOMPADDING", (0,0),(-1,0), 9),
            ("TOPPADDING",    (0,0),(-1,0), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#f0f6ff"), colors.white]),
            ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,1),(-1,-1), 10),
            ("TOPPADDING",    (0,1),(-1,-1), 7),
            ("BOTTOMPADDING", (0,1),(-1,-1), 7),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ]))
        return t

    story.append(Paragraph("Patient Information", head_s))
    story.append(tbl([
        ["Field",           "Details"],
        ["Patient Name",    patient["name"]],
        ["Patient ID",      patient["patient_id"]],
        ["Age",             str(patient["age"])],
        ["Gender",          patient["gender"]],
        ["Scan Date",       patient["scan_date"]],
        ["Referring Doctor",patient.get("doctor", "—")],
    ], [2*inch, 4.5*inch]))

    if patient.get("medical_history","").strip():
        story.append(Spacer(1, 8))
        story.append(Paragraph("Medical History", head_s))
        story.append(Paragraph(patient["medical_history"], body_s))

    story.append(Paragraph("AI Analysis Results", head_s))
    story.append(tbl([
        ["Metric",                  "Value"],
        ["Diagnosis",               result["prediction"]],
        ["Confidence Score",        f"{result['confidence']:.2f}%"],
        ["Normal Probability",      f"{result['normal_prob']:.2f}%"],
        ["Parkinson's Probability", f"{result['parkinson_prob']:.2f}%"],
        ["Risk Level",              result["risk_level"]],
        ["Analysis Time",           result["timestamp"]],
        ["AI Model",                result["model_name"]],
    ], [2.5*inch, 4*inch]))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Brain MRI Scan & Grad-CAM Heatmap", head_s))
    img_buf = io.BytesIO(); result["image"].save(img_buf, "PNG"); img_buf.seek(0)

    if result["cam_overlay"] is not None:
        cam_buf = io.BytesIO(); result["cam_overlay"].save(cam_buf, "PNG"); cam_buf.seek(0)
        img_tbl = Table([[
            RLImage(img_buf, width=2.8*inch, height=2.8*inch),
            RLImage(cam_buf, width=2.8*inch, height=2.8*inch),
        ]], colWidths=[3.2*inch, 3.2*inch])
        caption_txt = "Left: Original MRI &nbsp;&nbsp;|&nbsp;&nbsp; Right: Grad-CAM Overlay"
    else:
        img_tbl = Table([[
            RLImage(img_buf, width=2.8*inch, height=2.8*inch),
        ]], colWidths=[3.2*inch])
        caption_txt = "Original MRI Scan"

    img_tbl.setStyle(TableStyle([
        ("ALIGN",  (0,0),(-1,-1), "CENTER"),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("BOX",    (0,0),(-1,-1), 0, colors.white),
    ]))
    story.append(img_tbl)
    caption_s = ParagraphStyle("C", parent=styles["Normal"], fontSize=8,
        textColor=colors.grey, alignment=TA_CENTER, spaceBefore=4)
    story.append(Paragraph(caption_txt, caption_s))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system for research and educational "
        "purposes only. It must NOT replace clinical diagnosis by a qualified medical professional. "
        "Always consult a licensed neurologist for any medical decisions.", warn_s))

    story.append(Spacer(1, 20))
    footer_s = ParagraphStyle("F", parent=styles["Normal"], fontSize=8,
        textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph(
        f"NeuroScan AI · BVC College of Engineering, Palacharla · "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_s))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_logo_b64(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext  = path.rsplit(".", 1)[-1].lower()
            mime = "image/jpeg" if ext in ("jpg","jpeg") else f"image/{ext}"
            return f"data:{mime};base64,{data}"
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroScan AI — Parkinson's MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Keep ALL the original royal CSS from your existing app ──────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,400&family=Cinzel:wght@400;500;600;700;900&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');
:root {
  --gold:#b8860b;--gold-bright:#9a6e00;--gold-dim:#c9a84c;
  --gold-muted:rgba(184,134,11,0.10);--gold-line:rgba(184,134,11,0.20);
  --gold-border:rgba(184,134,11,0.40);--text:#2c1a00;--text-2:#7a5a1a;
  --radius:12px;--radius-sm:7px;
  --shadow-royal:0 4px 24px rgba(184,134,11,0.10),0 1px 6px rgba(0,0,0,0.06);
}
html,body,.stApp{background:#fdf8f0!important;font-family:'EB Garamond','Cormorant Garamond',serif!important;color:var(--text)!important;}
.block-container{padding-top:0!important;max-width:1380px!important;}
#MainMenu,footer,header{visibility:hidden;}
.card{background:linear-gradient(145deg,#fffdf8 0%,#fff8ec 100%);border:1px solid var(--gold-line);border-radius:var(--radius);padding:1.8rem 2rem;margin:.7rem 0;box-shadow:var(--shadow-royal);}
.sec-head{display:flex;align-items:center;gap:.9rem;font-family:'Cinzel',serif;font-size:.68rem;font-weight:600;color:var(--gold);letter-spacing:3.5px;text-transform:uppercase;margin:2rem 0 1rem;}
.sec-head::before,.sec-head::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--gold-dim),transparent);}
.stTextInput input,.stNumberInput input,.stTextArea textarea,.stDateInput input{background:#fffdf8!important;color:var(--text)!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius-sm)!important;font-family:'EB Garamond',serif!important;font-size:1rem!important;}
div[data-baseweb="select"]>div{background:#fffdf8!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius-sm)!important;color:var(--text)!important;}
label{color:var(--text-2)!important;font-family:'Cinzel',serif!important;font-size:.68rem!important;font-weight:600!important;letter-spacing:1.5px!important;text-transform:uppercase!important;}
.stButton>button{background:linear-gradient(135deg,#8a6e2f 0%,#c9a84c 40%,#e8c96a 60%,#c9a84c 80%,#8a6e2f 100%)!important;color:#07090f!important;border:1px solid rgba(232,201,106,0.5)!important;border-radius:var(--radius-sm)!important;padding:.75rem 1.6rem!important;font-family:'Cinzel',serif!important;font-size:.78rem!important;font-weight:700!important;letter-spacing:2.5px!important;text-transform:uppercase!important;width:100%!important;transition:all .3s ease!important;box-shadow:0 3px 16px rgba(201,168,76,0.3)!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 24px rgba(201,168,76,0.45)!important;}
.stDownloadButton>button{background:linear-gradient(135deg,#0a2e18 0%,#1a6a38 40%,#2a8a50 60%,#1a6a38 80%,#0a2e18 100%)!important;color:#c9e8c8!important;border:1px solid rgba(74,222,128,0.3)!important;border-radius:var(--radius-sm)!important;padding:.75rem 1.6rem!important;font-family:'Cinzel',serif!important;font-size:.78rem!important;font-weight:700!important;letter-spacing:2.5px!important;text-transform:uppercase!important;width:100%!important;}
[data-testid="stFileUploader"]{background:#fffdf8!important;border:2px dashed var(--gold-line)!important;border-radius:var(--radius)!important;padding:1.6rem!important;}
.diag-badge{display:inline-flex;align-items:center;gap:.5rem;font-family:'Cinzel',serif;font-size:1.3rem;font-weight:700;padding:.5rem 1.2rem;border-radius:4px;letter-spacing:1px;}
.diag-normal{background:rgba(30,132,73,0.10);color:#4ade80;border:1px solid rgba(74,222,128,0.35);}
.diag-parkinson{background:rgba(192,57,43,0.10);color:#f87171;border:1px solid rgba(248,113,113,0.35);}
.stat-tile{background:linear-gradient(145deg,#fffdf8,#fff6e8);border:1px solid var(--gold-line);border-radius:var(--radius);padding:1.1rem 1.3rem;text-align:center;box-shadow:var(--shadow-royal);}
.stat-value{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#9a6e00;line-height:1.1;margin-bottom:.3rem;}
.stat-label{font-family:'Cinzel',serif;font-size:.58rem;color:var(--text-2);letter-spacing:2px;text-transform:uppercase;}
.risk-low{background:rgba(74,222,128,.1);color:#4ade80;border:1px solid rgba(74,222,128,.3);border-radius:4px;padding:.25rem 1rem;font-size:.75rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1px;}
.risk-moderate{background:rgba(251,191,36,.1);color:#fbbf24;border:1px solid rgba(251,191,36,.3);border-radius:4px;padding:.25rem 1rem;font-size:.75rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1px;}
.risk-high{background:rgba(248,113,113,.1);color:#f87171;border:1px solid rgba(248,113,113,.3);border-radius:4px;padding:.25rem 1rem;font-size:.75rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1px;}
.prob-row{margin:.6rem 0;}
.prob-label{font-family:'Cinzel',serif;font-size:.65rem;color:var(--text-2);margin-bottom:.3rem;display:flex;justify-content:space-between;letter-spacing:1px;}
.prob-track{background:rgba(255,255,255,.04);border-radius:3px;height:7px;overflow:hidden;border:1px solid rgba(201,168,76,0.1);}
.prob-fill-n{height:100%;border-radius:3px;background:linear-gradient(90deg,#065f46,#4ade80);}
.prob-fill-p{height:100%;border-radius:3px;background:linear-gradient(90deg,#7f1d1d,#f87171);}
.img-frame{border:1px solid var(--gold-line);border-radius:var(--radius);overflow:hidden;background:#f5f0e8;box-shadow:var(--shadow-royal);}
.img-caption{font-family:'Cinzel',serif;font-size:.6rem;color:#b89040;text-align:center;padding:.45rem 0 .25rem;letter-spacing:1.5px;text-transform:uppercase;background:rgba(245,240,232,0.8);}
.hm-legend{display:flex;align-items:center;gap:.6rem;font-family:'Cinzel',serif;font-size:.6rem;color:#b89040;margin-top:.6rem;letter-spacing:1px;}
.hm-bar{flex:1;height:6px;border-radius:3px;background:linear-gradient(90deg,#00008b,#0000ff,#00ff00,#ffff00,#ff0000);}
.stTabs [data-baseweb="tab-list"]{background:#fffdf8!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius)!important;padding:.35rem!important;}
.stTabs [data-baseweb="tab"]{font-family:'Cinzel',serif!important;font-size:.75rem!important;font-weight:600!important;color:#7a5a1a!important;border-radius:var(--radius-sm)!important;padding:.6rem 1.3rem!important;letter-spacing:1.5px!important;}
.stTabs [aria-selected="true"]{background:rgba(184,134,11,0.10)!important;color:#7a4f00!important;border:1px solid rgba(184,134,11,0.3)!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#fdf8f0,#fdf3e7)!important;border-right:1px solid var(--gold-line)!important;}
[data-testid="stMetric"]{background:linear-gradient(145deg,#fffdf8,#fff6e8)!important;border:1px solid var(--gold-line)!important;border-radius:var(--radius)!important;padding:1rem 1.2rem!important;}
[data-testid="stMetricValue"]{font-family:'Playfair Display',serif!important;font-size:1.6rem!important;font-weight:700!important;color:#9a6e00!important;}
[data-testid="stMetricLabel"]{color:#7a5a1a!important;font-family:'Cinzel',serif!important;font-size:.58rem!important;letter-spacing:1.5px!important;text-transform:uppercase!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#fdf8f0;}
::-webkit-scrollbar-thumb{background:#c9a84c;border-radius:10px;}
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,var(--gold-line),transparent)!important;margin:1.8rem 0!important;}
.team-card{background:linear-gradient(145deg,#fffdf8,#fff6e8);border:1px solid var(--gold-line);border-radius:var(--radius);padding:1.4rem 1rem;text-align:center;transition:border-color .3s,transform .3s;}
.team-card:hover{border-color:var(--gold-border);transform:translateY(-4px);}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
logo_src  = get_logo_b64("logo.png")
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
    'display:flex;flex-direction:column;align-items:center;gap:1rem;position:relative;overflow:hidden;">'
    '<div style="position:absolute;top:0;left:0;right:0;height:3px;'
    'background:linear-gradient(90deg,transparent,#c9a84c,#e8c96a,#c9a84c,transparent);"></div>'
    '<div style="position:absolute;top:14px;left:24px;font-size:.9rem;color:rgba(184,134,11,0.4);'
    'font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'
    '<div style="position:absolute;top:14px;right:24px;font-size:.9rem;color:rgba(184,134,11,0.4);'
    'font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'
    + logo_html +
    '<div style="font-family:Cinzel,serif;font-size:2.8rem;font-weight:900;color:#2c1a00;'
    'letter-spacing:8px;line-height:1;text-align:center;">NEUROSCAN&nbsp;'
    '<span style="color:#b8860b;">AI</span></div>'
    '<div style="font-family:Cormorant Garamond,serif;font-size:1rem;font-style:italic;'
    'color:#9a7030;letter-spacing:4px;text-align:center;">'
    "Parkinson's Detection · Brain MRI · Hybrid Deep Learning</div>"
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

# ── Download + load model ──────────────────────────────────────────────────
download_model()

# ── Session state ──────────────────────────────────────────────────────────
for k, v in [
    ("prediction_made",   False),
    ("patient_data",      {}),
    ("prediction_result", {}),
    ("batch_results",     []),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:.8rem 0 1.2rem;">'
        '<div style="font-family:Cinzel,serif;font-size:1.1rem;font-weight:700;'
        'color:#8a5e00;letter-spacing:3px;">⚕ NEUROSCAN AI</div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,'
        'rgba(184,134,11,0.4),transparent);margin:.6rem 0;"></div></div>',
        unsafe_allow_html=True,
    )
    st.info("Advanced hybrid deep-learning system for brain MRI analysis.")

    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#9a7030;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.5rem;">⚙ Model</div>',
        unsafe_allow_html=True,
    )

    # Model selector — picks which .pth to load
    model_choice = st.selectbox(
        "Select Model",
        ["HybridNet_RV (ResNet50+ViT)", "HybridNet_EV (EfficientNet+ViT)",
         "ResNet50", "EfficientNet-B4", "ViT-B16", "MiniSegNet"],
        index=0
    )

    # Map choice to filename
    MODEL_MAP = {
        "HybridNet_RV (ResNet50+ViT)":      "hybridnet_rv_best.pth",
        "HybridNet_EV (EfficientNet+ViT)":  "hybridnet_ev_best.pth",
        "ResNet50":                          "resnet50_best.pth",
        "EfficientNet-B4":                   "efficientnet-b4_best.pth",
        "ViT-B16":                           "vit-b16_best.pth",
        "MiniSegNet":                        "minisegnet_best.pth",
    }
    MODEL_PATH = MODEL_MAP[model_choice]

    st.success(
        f"**Active:** {model_choice}\n\n"
        "**Dataset:** irfansheriff/parkinsons-brain-mri-dataset\n\n"
        "**CV Accuracy:** 100% (ResNet50/EffNet/ViT)\n\n"
        "**XAI:** Grad-CAM\n\n"
        "**Status:** 🟢 Online"
    )
    st.warning("⚠️ For **research/academic** purposes only.")

    c1, c2 = st.columns(2)
    with c1: st.metric("Precision", "100%")
    with c2: st.metric("Recall",    "98%")


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_scan, tab_batch, tab_about = st.tabs([
    "🧠  Single MRI Analysis",
    "📦  Batch Analysis",
    "🏛  About",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — SINGLE SCAN
# ─────────────────────────────────────────────────────────────────────────────
with tab_scan:
    col_form, col_upload = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="sec-head">✦ Patient Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        patient_name = st.text_input("Full Name *", placeholder="Patient's full name")
        c1, c2 = st.columns(2)
        with c1: patient_age    = st.number_input("Age *", 0, 120, 45)
        with c2: patient_gender = st.selectbox("Gender *", ["Male","Female","Other"])
        c3, c4 = st.columns(2)
        with c3: patient_id = st.text_input("Patient ID *", placeholder="P-2024-0001")
        with c4: scan_date  = st.date_input("Scan Date *", value=datetime.now())
        referring_doctor = st.text_input("Referring Doctor", placeholder="Dr. Name (optional)")
        medical_history  = st.text_area("Medical History", placeholder="Relevant history (optional)", height=90)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_upload:
        st.markdown('<div class="sec-head">✦ MRI Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Scan", type=["png","jpg","jpeg"],
            help="PNG / JPG / JPEG · any resolution",
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded MRI", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            w, h = image.size
            st.markdown(
                f'<div style="font-family:Cinzel,monospace;font-size:.62rem;color:#9a7030;'
                f'margin-top:.5rem;text-align:center;">'
                f'{image.mode} · {w}×{h}px · {uploaded_file.size/1024:.1f} KB</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Upload a brain MRI scan to begin analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head">✦ Analysis & Results</div>', unsafe_allow_html=True)
    btn_col, report_col = st.columns([1, 1], gap="large")

    with btn_col:
        if st.button("⚕ Analyze MRI Scan"):
            if not patient_name.strip():
                st.error("Please enter the patient's full name.")
            elif not patient_id.strip():
                st.error("Please enter a Patient ID.")
            elif uploaded_file is None:
                st.error("Please upload a brain MRI scan.")
            else:
                with st.spinner("Running AI analysis…"):
                    st.session_state.patient_data = {
                        "name":            patient_name.strip(),
                        "age":             patient_age,
                        "gender":          patient_gender,
                        "patient_id":      patient_id.strip(),
                        "scan_date":       scan_date.strftime("%Y-%m-%d"),
                        "doctor":          referring_doctor.strip() or "—",
                        "medical_history": medical_history.strip(),
                    }
                    try:
                        model, classes, mn, device = load_model()
                        result = predict(model, device, classes, mn, image)
                        st.session_state.prediction_result = result
                        st.session_state.prediction_made   = True
                        st.success("Analysis complete!")
                        st.balloons()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

    if st.session_state.prediction_made:
        r        = st.session_state.prediction_result
        is_normal = r["class_idx"] == 0
        diag_cls  = "diag-normal" if is_normal else "diag-parkinson"
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
                f'{"✦" if is_normal else "⚠"} {r["prediction"].upper()}</span></div></div>',
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
                f'<div style="margin-top:.6rem;"><span class="{risk_cls}">'
                f'{r["risk_level"]} Risk</span></div></div>',
                unsafe_allow_html=True,
            )
        with d4:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Model Used</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.9rem;'
                f'color:#a89060;margin-top:.35rem;">{r["model_name"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>✦ Normal</span><span>{r["normal_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-n" style="width:{r["normal_prob"]:.1f}%"></div></div>'
            f'</div>'
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>⚠ Parkinson\'s</span><span>{r["parkinson_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-p" style="width:{r["parkinson_prob"]:.1f}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if r["cam_overlay"] is not None:
            st.markdown('<div class="sec-head">✦ Grad-CAM Explainability</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            ic1, ic2, ic3 = st.columns(3)
            for col, img_obj, cap in [
                (ic1, r["image"],       "Original MRI"),
                (ic2, r["cam_heatmap"], "Attention Heatmap"),
                (ic3, r["cam_overlay"], "Grad-CAM Overlay"),
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
            if st.button("📜 Generate PDF Report"):
                with st.spinner("Building report…"):
                    try:
                        pdf_bytes = build_pdf(
                            st.session_state.patient_data,
                            st.session_state.prediction_result,
                        )
                        p = st.session_state.patient_data
                        fname = f"NeuroScan_{p['patient_id']}_{datetime.now().strftime('%Y%m%d')}.pdf"
                        st.download_button(
                            "⬇ Download PDF Report",
                            data=pdf_bytes, file_name=fname, mime="application/pdf",
                        )
                        st.success("PDF ready!")
                    except Exception as exc:
                        st.error(f"PDF error: {exc}")
        else:
            st.info("Run an analysis first to generate a report.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — BATCH
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="sec-head">✦ Batch MRI Analysis</div>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Upload Multiple Brain MRI Scans",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True, key="batch_uploader",
    )

    if batch_files:
        st.info(f"{len(batch_files)} file(s) ready.")
        if st.button("⚕ Run Batch Analysis"):
            try:
                model, classes, mn, device = load_model()
            except Exception as exc:
                st.error(f"Failed to load model: {exc}"); st.stop()

            batch_results = []
            prog   = st.progress(0)
            status = st.empty()
            for i, f in enumerate(batch_files):
                status.markdown(
                    f'<p style="font-family:Cinzel,serif;font-size:.75rem;color:#c9a84c;">'
                    f'Processing {f.name} ({i+1}/{len(batch_files)})…</p>',
                    unsafe_allow_html=True,
                )
                try:
                    res = predict(model, device, classes, mn, Image.open(f))
                    res["filename"] = f.name
                    batch_results.append(res)
                except Exception as exc:
                    st.warning(f"Skipped {f.name}: {exc}")
                prog.progress((i+1)/len(batch_files))

            st.session_state.batch_results = batch_results
            status.empty()
            st.success(f"Done! {len(batch_results)} scan(s) processed.")
            st.rerun()

    if st.session_state.batch_results:
        results  = st.session_state.batch_results
        n_total  = len(results)
        n_normal = sum(1 for r in results if r["class_idx"] == 0)
        n_park   = n_total - n_normal
        avg_conf = np.mean([r["confidence"] for r in results])

        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Total Scans",    n_total)
        with m2: st.metric("Normal",         n_normal)
        with m3: st.metric("Parkinson's",    n_park)
        with m4: st.metric("Avg Confidence", f"{avg_conf:.1f}%")

        ch_col, tb_col = st.columns([1,1], gap="large")
        with ch_col:
            fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor="#fdf8f0")
            ax.set_facecolor("#fdf8f0")
            if n_normal > 0 or n_park > 0:
                ax.pie([n_normal, n_park],
                       labels=["Normal","Parkinson's"],
                       autopct="%1.0f%%",
                       colors=["#4ade80","#f87171"],
                       startangle=90,
                       wedgeprops=dict(edgecolor="#0a0e1a", linewidth=2.5),
                       textprops=dict(color="#2c1a00", fontsize=10))
            ax.set_title("Scan Distribution", color="#7a5a1a", fontsize=10, pad=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with tb_col:
            df = pd.DataFrame([{
                "File":          r["filename"],
                "Prediction":    r["prediction"],
                "Confidence":    f"{r['confidence']:.1f}%",
                "Normal %":      f"{r['normal_prob']:.1f}%",
                "Parkinson's %": f"{r['parkinson_prob']:.1f}%",
                "Risk":          r["risk_level"],
                "Model":         r["model_name"],
            } for r in results])
            st.dataframe(df, use_container_width=True, height=280)
            st.download_button(
                "⬇ Download CSV",
                data=df.to_csv(index=False),
                file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        st.markdown('<div class="sec-head">✦ Per-Image Results</div>', unsafe_allow_html=True)
        for r in results:
            is_n = r["class_idx"] == 0
            col  = "#4ade80" if is_n else "#f87171"
            icon = "✦" if is_n else "⚠"
            rc1, rc2, rc3, rc4 = st.columns([1,2,1,1])
            with rc1:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(r["image"], use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(
                    f'<p style="font-family:Cinzel,serif;font-size:.62rem;'
                    f'color:#9a7030;margin-bottom:.3rem;">{r["filename"]}</p>'
                    f'<p style="font-family:Playfair Display,serif;font-size:1.4rem;'
                    f'font-weight:700;color:{col};margin:0;">{icon} {r["prediction"]}</p>'
                    f'<p style="font-family:Cinzel,serif;font-size:.6rem;'
                    f'color:#9a7030;margin-top:.35rem;">{r["model_name"]} · {r["timestamp"]}</p>',
                    unsafe_allow_html=True,
                )
            with rc3:
                st.metric("Confidence", f"{r['confidence']:.1f}%")
                st.metric("Normal %",   f"{r['normal_prob']:.1f}%")
            with rc4:
                if r["cam_overlay"] is not None:
                    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                    st.image(r["cam_overlay"], use_column_width=True)
                    st.markdown('<div class="img-caption">Grad-CAM</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<hr>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — ABOUT  (identical to your original)
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    college_logo = get_logo_b64("bvcr.jpg")
    clg_html = (
        f'<img src="{college_logo}" style="width:110px;height:110px;object-fit:contain;'
        f'margin-bottom:1.2rem;border:2px solid rgba(184,134,11,0.45);border-radius:10px;"/>'
        if college_logo else '<div style="font-size:4rem;margin-bottom:1.2rem;">🏛</div>'
    )

    st.markdown(
        f'<div style="background:linear-gradient(160deg,#fdf3dc,#fdf8f0,#fce8b0);'
        f'border:1px solid rgba(184,134,11,0.28);border-radius:16px;'
        f'padding:3rem 2.5rem 2.5rem;margin-bottom:2rem;text-align:center;">'
        + clg_html +
        f'<div style="font-family:Cinzel,serif;font-size:1.9rem;font-weight:900;'
        f'color:#2c1a00;letter-spacing:4px;margin-bottom:.5rem;">BVC College of Engineering</div>'
        f'<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:1.05rem;'
        f'color:#b8860b;letter-spacing:3px;margin-bottom:.8rem;">Palacharla, Andhra Pradesh</div>'
        f'<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a5a1a;'
        f'line-height:1.8;max-width:680px;margin:0 auto;">'
        f'BVC College of Engineering is a premier autonomous institution in Andhra Pradesh, '
        f'recognized for academic excellence and cutting-edge research.</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-head">✦ Model Performance (5-Fold Cross Validation)</div>', unsafe_allow_html=True)
    pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
    for col, val, lbl in [
        (pm1, "100%",  "ResNet50"),
        (pm2, "100%",  "EfficientNet"),
        (pm3, "99.88%","ViT-B16"),
        (pm4, "98.4%", "MiniSegNet"),
        (pm5, "98.4%", "HybridNet_RV"),
        (pm6, "99.2%", "HybridNet_EV"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
                f'padding:1.2rem .8rem;text-align:center;">'
                f'<div style="font-family:Playfair Display,serif;font-size:1.5rem;'
                f'font-weight:900;color:#7a4f00;">{val}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:#9a7030;'
                f'letter-spacing:1px;text-transform:uppercase;margin-top:.3rem;">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sec-head" style="margin-top:2.4rem;">✦ Project Team</div>', unsafe_allow_html=True)
    team = [
        {"roll":"236M5A0408","name":"G Srinivasu",      "icon":"👨‍💻","role":"AI Model & Backend"},
        {"roll":"226M1A0460","name":"S Anusha Devi",    "icon":"👩‍💻","role":"UI/UX & Frontend"},
        {"roll":"226M1A0473","name":"V V Siva Vardhan", "icon":"👨‍💻","role":"Data & Testing"},
        {"roll":"236M5A0415","name":"N L Sandeep",      "icon":"👨‍💻","role":"Reports & Integration"},
    ]
    tcols = st.columns(4)
    for i, m in enumerate(team):
        with tcols[i]:
            st.markdown(
                f'<div class="team-card">'
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(192,57,43,0.05),rgba(192,57,43,0.02));'
        'border:1px solid rgba(192,57,43,0.2);border-left:4px solid #c0392b;'
        'border-radius:10px;padding:1.2rem 1.8rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#c0392b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">⚕ Medical Disclaimer</div>'
        '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a1a1a;line-height:1.7;">'
        'This project is developed <strong>for academic and research purposes only</strong>. '
        'Always consult a qualified neurologist for any medical decisions.</div></div>',
        unsafe_allow_html=True,
    )


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;padding:1.6rem;background:linear-gradient(180deg,#fdf8f0,#fdf3dc);">'
    '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#9a7030;'
    'letter-spacing:3px;text-transform:uppercase;">'
    'Research &amp; Educational Use Only · Not for Clinical Diagnosis'
    '</div>'
    '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#9a7030;margin-top:.4rem;">'
    'NeuroScan AI · ResNet50+ViT · EfficientNet+ViT · Grad-CAM · PyTorch · Streamlit'
    '</div></div>',
    unsafe_allow_html=True,
)
