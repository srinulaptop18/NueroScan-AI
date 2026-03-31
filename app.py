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
MODEL_FILE_ID  = "1uXL90WvSm7akZbgpYZCinGAzLUhwhl6i"
MODEL_FILENAME = "hybridnet_ev_best.pth"
MODEL_LABEL    = "HybridNet_EV (EfficientNet-B4 + ViT)"

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class CrossAttentionFusion(nn.Module):
    def __init__(self, d=512, h=8, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.ffn  = nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(drop),nn.Linear(d*2,d))
    def forward(self, cnn_t, vit_t):
        a,_ = self.attn(cnn_t,vit_t,vit_t)
        x   = self.n1(cnn_t+a)
        return self.n2(x+self.ffn(x))

class HybridNet_EV(nn.Module):
    def __init__(self, nc=2, d_model=512, n_heads=8, dropout=0.2):
        super().__init__()
        self.cnn      = timm.create_model('efficientnet_b4',pretrained=False,features_only=True)
        cnn_ch        = self.cnn.feature_info[-1]['num_chs']
        self.vit      = timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
        self.cnn_proj = nn.Linear(cnn_ch,d_model)
        self.vit_proj = nn.Linear(768,d_model)
        self.fusion   = CrossAttentionFusion(d_model,n_heads,dropout)
        self.gap      = nn.AdaptiveAvgPool1d(1)
        self.head     = nn.Sequential(nn.LayerNorm(d_model),nn.Dropout(dropout),nn.Linear(d_model,256),nn.GELU(),nn.Dropout(dropout/2),nn.Linear(256,nc))
    def forward(self, x):
        cf=self.cnn(x)[-1]; ct=self.cnn_proj(cf.flatten(2).transpose(1,2))
        vt=self.vit_proj(self.vit.forward_features(x)[:,1:,:]); fused=self.fusion(ct,vt)
        return self.head(self.gap(fused.transpose(1,2)).squeeze(-1))
    def get_gradcam_layer(self):
        return self.cnn.blocks[-1][-1].conv_pwl

def download_model(path):
    if not os.path.exists(path):
        with st.spinner("Downloading model — please wait…"):
            try:
                gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}",path,quiet=False)
                st.success("Model downloaded.")
            except Exception as e:
                st.error(f"Download failed: {e}"); st.stop()

@st.cache_resource(show_spinner=False)
def load_model_cached(path):
    device = torch.device("cpu")
    try: ck = torch.load(path,map_location=device,weights_only=False)
    except Exception as e: st.error(f"Load failed: {e}"); st.stop()
    if isinstance(ck,dict) and 'model_state_dict' in ck:
        sd=ck['model_state_dict']; nc=ck.get('config',{}).get('num_classes',2)
        raw=ck.get('classes',['normal','parkinson'])
    elif isinstance(ck,dict):
        sd=ck; nc=2; raw=['normal','parkinson']
    else: st.error("Unknown checkpoint."); st.stop()
    cm2={'normal':'Normal','healthy':'Normal','parkinson':"Parkinson's Disease",'parkinsons':"Parkinson's Disease"}
    classes=[cm2.get(c.lower(),c.title()) for c in raw]
    m=HybridNet_EV(nc=nc)
    try: m.load_state_dict(sd,strict=True)
    except RuntimeError as e: st.error(f"Weight mismatch: {e}"); st.stop()
    return m.eval().to(device),classes,device

class GradCAM:
    def __init__(self,model,layer):
        self.model=model; self._a=self._g=None
        layer.register_forward_hook(lambda m,i,o:setattr(self,'_a',o))
        layer.register_full_backward_hook(lambda m,gi,go:setattr(self,'_g',go[0]))
    def generate(self,t,idx,dev):
        self._a=self._g=None; self.model.eval()
        x=t.clone().to(dev).requires_grad_(True)
        with torch.enable_grad():
            out=self.model(x); self.model.zero_grad(); out[0,idx].backward(retain_graph=True)
        if self._g is None or self._a is None: return np.zeros((224,224))
        w=self._g.mean(dim=[2,3],keepdim=True)
        cam=F.relu((w*self._a).sum(dim=1,keepdim=True))
        cam=F.interpolate(cam,(224,224),mode='bilinear',align_corners=False).squeeze().cpu().detach().numpy()
        lo,hi=cam.min(),cam.max()
        return (cam-lo)/(hi-lo) if hi-lo>1e-8 else np.zeros((224,224))

def apply_colormap(pil,cam):
    h=(cm.jet(cam)[:,:,:3]*255).astype(np.uint8)
    o=np.array(pil.resize((224,224))).astype(np.float32)
    return Image.fromarray(np.clip(.55*o+.45*h.astype(np.float32),0,255).astype(np.uint8)),Image.fromarray(h)

TRANSFORM=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def predict(model,device,classes,pil):
    img=pil.convert('RGB'); t=TRANSFORM(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out=model(t); p=F.softmax(out,dim=1)[0].cpu().numpy()
    ci=int(p.argmax()); conf=float(p[ci]*100)
    ni=pi=None
    for i,c in enumerate(classes):
        cl=c.lower()
        if 'normal' in cl or 'healthy' in cl: ni=i
        if 'parkinson' in cl: pi=i
    ni=ni or 0; pi=pi if pi is not None else 1
    np_=float(p[ni]*100); pp=float(p[pi]*100)
    isk=(ci==pi); risk=('High' if conf>=85 else 'Moderate') if isk else 'Low'
    ov=hm=None
    try:
        gc_=GradCAM(model,model.get_gradcam_layer()); fresh=TRANSFORM(img).unsqueeze(0)
        cm_=gc_.generate(fresh,ci,device)
        if cm_.max()>0: ov,hm=apply_colormap(img,cm_)
    except: pass
    del t; gc.collect()
    return {'prediction':classes[ci],'class_idx':ci,'is_parkinson':isk,'confidence':conf,
            'normal_prob':np_,'parkinson_prob':pp,'risk_level':risk,
            'cam_overlay':ov,'cam_heatmap':hm,'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image':img,'model_name':MODEL_LABEL}

def build_pdf(patient,result):
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=letter,leftMargin=.75*inch,rightMargin=.75*inch,topMargin=.75*inch,bottomMargin=.75*inch)
    story=[]; styles=getSampleStyleSheet()
    GD=colors.HexColor('#7a4f00'); GL=colors.HexColor('#fdf3dc'); PM=colors.HexColor('#fdf8f0')
    DP=colors.HexColor('#2c1a00'); MG=colors.HexColor('#c9a84c'); CR=colors.HexColor('#c0392b')
    EM=colors.HexColor('#1a6a38'); GT=colors.HexColor('#7a5a1a'); GC=colors.HexColor('#b8860b')
    def gline():
        t=Table([['','','']],colWidths=[.5*inch,6.5*inch,.5*inch])
        t.setStyle(TableStyle([('LINEABOVE',(1,0),(1,0),1,MG),('ALIGN',(0,0),(-1,-1),'CENTER'),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        return t
    def gtbl(rows,cw):
        t=Table(rows,colWidths=cw)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),GD),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),10),('BOTTOMPADDING',(0,0),(-1,0),8),('TOPPADDING',(0,0),(-1,0),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[GL,PM]),('GRID',(0,0),(-1,-1),.4,MG),('FONTNAME',(0,1),(-1,-1),'Helvetica'),('FONTSIZE',(0,1),(-1,-1),10),('TEXTCOLOR',(0,1),(-1,-1),DP),('TOPPADDING',(0,1),(-1,-1),6),('BOTTOMPADDING',(0,1),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),10),('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),('TEXTCOLOR',(0,1),(0,-1),GD)]))
        return t
    ts=ParagraphStyle('T',parent=styles['Heading1'],fontSize=22,textColor=GD,spaceAfter=2,alignment=TA_CENTER,fontName='Helvetica-Bold')
    ss=ParagraphStyle('S',parent=styles['Normal'],fontSize=9,textColor=GC,spaceAfter=4,alignment=TA_CENTER,fontName='Helvetica')
    ps=ParagraphStyle('P',parent=styles['Normal'],fontSize=10,textColor=DP,spaceAfter=16,alignment=TA_CENTER,fontName='Helvetica-Bold')
    hs=ParagraphStyle('H',parent=styles['Heading2'],fontSize=12,textColor=GD,spaceAfter=6,spaceBefore=12,fontName='Helvetica-Bold')
    bs=ParagraphStyle('B',parent=styles['Normal'],fontSize=10,textColor=DP,leading=14)
    ws=ParagraphStyle('W',parent=styles['Normal'],fontSize=9,textColor=colors.HexColor('#7f1d1d'),leading=13,backColor=colors.HexColor('#fff1f1'),borderPad=6)
    fs=ParagraphStyle('F',parent=styles['Normal'],fontSize=8,textColor=GT,alignment=TA_CENTER)
    story+=[Paragraph('NeuroScan AI',ts),Paragraph("Parkinson's Disease MRI Analysis Report",ss),Paragraph('PARKINSONS DISEASE DETECTION USING DEEP LEARNING ON BRAIN MRI',ps),gline(),Spacer(1,4)]
    dc=CR if result.get('is_parkinson') else EM
    vt=Table([[result['prediction'].upper()]],colWidths=[7*inch])
    vt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),dc),('TEXTCOLOR',(0,0),(-1,-1),colors.white),('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),16),('ALIGN',(0,0),(-1,-1),'CENTER'),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
    story+=[vt,Spacer(1,10),Paragraph('Patient Information',hs),gline()]
    story.append(gtbl([['Field','Details'],['Patient Name',patient['name']],['Patient ID',patient['patient_id']],['Age',str(patient['age'])],['Gender',patient['gender']],['Scan Date',patient['scan_date']],['Referring Doctor',patient.get('doctor','—')]],[2.2*inch,4.8*inch]))
    if patient.get('medical_history','').strip():
        story+=[Spacer(1,8),Paragraph('Medical History',hs),gline(),Paragraph(patient['medical_history'],bs)]
    story+=[Paragraph('AI Analysis Results',hs),gline()]
    story.append(gtbl([['Metric','Value'],['Diagnosis',result['prediction']],['Confidence',f"{result['confidence']:.2f}%"],['Normal Probability',f"{result['normal_prob']:.2f}%"],["Parkinson's Probability",f"{result['parkinson_prob']:.2f}%"],['Risk Level',result['risk_level']],['Analysis Time',result['timestamp']],['AI Model',result['model_name']]],[2.8*inch,4.2*inch]))
    story+=[Spacer(1,10),Paragraph('Brain MRI Scan & Grad-CAM Heatmap',hs),gline()]
    ib=io.BytesIO(); result['image'].save(ib,'PNG'); ib.seek(0)
    if result.get('cam_overlay'):
        cb=io.BytesIO(); result['cam_overlay'].save(cb,'PNG'); cb.seek(0)
        it=Table([[RLImage(ib,width=2.8*inch,height=2.8*inch),RLImage(cb,width=2.8*inch,height=2.8*inch)]],colWidths=[3.2*inch,3.2*inch]); ct2='Left: Original MRI  |  Right: Grad-CAM Overlay'
    else:
        it=Table([[RLImage(ib,width=2.8*inch,height=2.8*inch)]],colWidths=[3.2*inch]); ct2='Original MRI Scan'
    it.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOX',(0,0),(-1,-1),1.5,MG),('BACKGROUND',(0,0),(-1,-1),PM)]))
    story+=[it,Paragraph(ct2,ParagraphStyle('C',parent=styles['Normal'],fontSize=8,textColor=GC,alignment=TA_CENTER,spaceBefore=4)),Spacer(1,14),gline(),Paragraph('⚠️ DISCLAIMER: This report is generated by an AI system for research and educational purposes only. It must NOT replace clinical diagnosis by a qualified medical professional. Always consult a licensed neurologist.',ws),Spacer(1,16),gline(),Spacer(1,4),Paragraph(f"NeuroScan AI | Parkinson's Detection | BVC College of Engineering, Rajahmundry | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",fs)]
    doc.build(story); buf.seek(0); return buf.read()

def get_logo_b64(path):
    try:
        if os.path.exists(path):
            with open(path,'rb') as f: data=base64.b64encode(f.read()).decode()
            ext=path.rsplit('.',1)[-1].lower()
            return f"data:{'image/jpeg' if ext in ('jpg','jpeg') else f'image/{ext}'};base64,{data}"
    except: pass
    return None

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="NeuroScan AI — Parkinson's MRI Analysis",page_icon='🧠',layout='wide',initial_sidebar_state='expanded')
for k,v in [('prediction_made',False),('patient_data',{}),('prediction_result',{}),('batch_results',[])]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════════
#  ULTRA-PREMIUM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400;1,700&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400;1,600&family=Cinzel:wght@400;500;600;700;900&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap');

:root{
  --g1:#b8860b;--g2:#c9a84c;--g3:#e8c96a;--g4:#f5e0a0;--g5:#7a4f00;
  --glow:rgba(201,168,76,0.35);--glow2:rgba(184,134,11,0.18);
  --bdr:rgba(184,134,11,0.5);--bdr2:rgba(184,134,11,0.25);
  --cr:#c0392b;--cr2:rgba(192,57,43,0.12);
  --em:#1a7a3c;--em2:rgba(26,122,60,0.12);
  --tx:#1e1000;--tx2:#6b4c11;--tx3:#a07828;
  --bg:#fdf8ef;--bg2:#fffdf7;--bg3:#fdf3dc;
  --r:14px;--rs:8px;--rl:20px;
  --s1:0 4px 24px rgba(184,134,11,0.14),0 1px 6px rgba(0,0,0,0.06);
  --s2:0 8px 40px rgba(184,134,11,0.22),0 2px 12px rgba(0,0,0,0.08);
  --s3:0 16px 64px rgba(184,134,11,0.28),0 4px 20px rgba(0,0,0,0.10);
}

/* ══ BASE ══ */
html,body,.stApp{background:var(--bg)!important;font-family:'EB Garamond','Cormorant Garamond',Georgia,serif!important;color:var(--tx)!important;}
.block-container{padding-top:0!important;max-width:1420px!important;position:relative;z-index:1;}
#MainMenu,footer,header{visibility:hidden;}

/* ══ ANIMATED BACKGROUND ══ */
.stApp::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 100% 60% at 50% -10%,rgba(184,134,11,0.09) 0%,transparent 60%),
    radial-gradient(ellipse 50% 50% at 0% 100%,rgba(184,134,11,0.06) 0%,transparent 55%),
    radial-gradient(ellipse 50% 50% at 100% 60%,rgba(201,168,76,0.05) 0%,transparent 55%),
    repeating-linear-gradient(45deg,transparent,transparent 60px,rgba(184,134,11,0.018) 60px,rgba(184,134,11,0.018) 61px),
    repeating-linear-gradient(-45deg,transparent,transparent 60px,rgba(184,134,11,0.018) 60px,rgba(184,134,11,0.018) 61px);
}
.stApp::after{
  content:'';position:fixed;top:0;left:0;right:0;height:4px;z-index:9999;pointer-events:none;
  background:linear-gradient(90deg,transparent 0%,#c9a84c 20%,#f5e0a0 40%,#e8c96a 50%,#f5e0a0 60%,#c9a84c 80%,transparent 100%);
}

/* ══ PREMIUM CARDS ══ */
.card{
  background:linear-gradient(160deg,#fffef9 0%,#fffbf0 60%,#fff8e8 100%);
  border:1px solid var(--bdr);
  border-radius:var(--r);
  padding:2rem 2.2rem;
  margin:.8rem 0;
  box-shadow:var(--s1);
  position:relative;
  transition:all .4s cubic-bezier(.22,1,.36,1);
}
.card::before{
  content:'';position:absolute;inset:0;border-radius:var(--r);
  background:linear-gradient(160deg,rgba(255,255,255,0.7) 0%,transparent 50%);
  pointer-events:none;
}
.card::after{
  content:'';position:absolute;top:0;left:8%;right:8%;height:2px;
  background:linear-gradient(90deg,transparent,var(--g3),var(--g4),var(--g3),transparent);
  border-radius:0 0 6px 6px;pointer-events:none;
}
.card:hover{
  border-color:rgba(184,134,11,0.75);
  box-shadow:var(--s2),0 0 60px rgba(184,134,11,0.08);
  transform:translateY(-2px);
}

/* ══ GLASS CARD ══ */
.glass-card{
  background:rgba(255,253,247,0.85);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border:1px solid rgba(201,168,76,0.45);
  border-radius:var(--r);
  padding:1.8rem 2rem;
  box-shadow:var(--s1),inset 0 1px 0 rgba(255,255,255,0.9);
  position:relative;overflow:hidden;
  transition:all .4s cubic-bezier(.22,1,.36,1);
}
.glass-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.9),transparent);
}
.glass-card:hover{box-shadow:var(--s2),inset 0 1px 0 rgba(255,255,255,0.9);transform:translateY(-1px);}

/* ══ SECTION HEADINGS ══ */
.sec-head{
  display:flex;align-items:center;gap:1rem;
  font-family:'Cinzel',serif;font-size:.7rem;font-weight:600;
  color:var(--g1);letter-spacing:4px;text-transform:uppercase;
  margin:2.2rem 0 1.1rem;
}
.sec-head .ornament{font-size:.8rem;color:var(--g2);flex-shrink:0;}
.sec-head::before{content:'';flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--g2),rgba(232,201,106,0.5));}
.sec-head::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(232,201,106,0.5),var(--g2),transparent);}

/* ══ INPUTS ══ */
.stTextInput input,.stNumberInput input,.stTextArea textarea,.stDateInput input{
  background:linear-gradient(135deg,#fffef9,#fffbf0)!important;
  color:var(--tx)!important;
  border:1.5px solid rgba(184,134,11,0.3)!important;
  border-radius:var(--rs)!important;
  font-family:'EB Garamond',serif!important;font-size:1.05rem!important;
  transition:all .3s!important;
  box-shadow:inset 0 2px 6px rgba(184,134,11,0.06),0 1px 3px rgba(0,0,0,0.04)!important;
}
.stTextInput input:focus,.stNumberInput input:focus,.stTextArea textarea:focus{
  border-color:var(--g1)!important;
  box-shadow:0 0 0 4px rgba(184,134,11,0.12),inset 0 2px 6px rgba(184,134,11,0.06)!important;
  outline:none!important;
  background:linear-gradient(135deg,#ffffff,#fffdf5)!important;
}
div[data-baseweb="select"]>div{
  background:linear-gradient(135deg,#fffef9,#fffbf0)!important;
  border:1.5px solid rgba(184,134,11,0.3)!important;
  border-radius:var(--rs)!important;
  color:var(--tx)!important;font-family:'EB Garamond',serif!important;
  box-shadow:inset 0 1px 4px rgba(184,134,11,0.05)!important;
}
label{
  color:var(--tx2)!important;font-family:'Cinzel',serif!important;
  font-size:.66rem!important;font-weight:600!important;
  letter-spacing:1.8px!important;text-transform:uppercase!important;
}

/* ══ GOLD BUTTONS ══ */
.stButton>button{
  background:linear-gradient(135deg,#5a3c0a 0%,#8a6020 20%,#b8902a 38%,#d4a840 48%,#e8c050 52%,#d4a840 62%,#b8902a 78%,#7a5018 100%)!important;
  color:#1a0800!important;
  border:1px solid rgba(240,210,100,0.5)!important;
  border-radius:var(--rs)!important;
  padding:.82rem 1.8rem!important;
  font-family:'Cinzel',serif!important;font-size:.8rem!important;
  font-weight:700!important;letter-spacing:3px!important;
  text-transform:uppercase!important;width:100%!important;
  transition:all .35s cubic-bezier(.22,1,.36,1)!important;
  box-shadow:0 4px 20px rgba(184,134,11,0.4),0 1px 4px rgba(0,0,0,0.15),inset 0 1px 0 rgba(255,240,180,0.4),inset 0 -1px 0 rgba(0,0,0,0.15)!important;
  position:relative!important;overflow:hidden!important;
}
.stButton>button::after{
  content:'';position:absolute;top:0;left:-100%;width:60%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.25),transparent);
  transition:left .6s!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,#6a4c1a 0%,#9a7030 20%,#c8a038 38%,#e4b848 48%,#f8d058 52%,#e4b848 62%,#c8a038 78%,#8a6028 100%)!important;
  transform:translateY(-3px) scale(1.01)!important;
  box-shadow:0 8px 32px rgba(184,134,11,0.55),0 2px 8px rgba(0,0,0,0.15),inset 0 1px 0 rgba(255,240,180,0.5)!important;
}
.stButton>button:hover::after{left:150%!important;}
.stButton>button:active{transform:translateY(-1px) scale(1.0)!important;}

.stDownloadButton>button{
  background:linear-gradient(135deg,#041a0c 0%,#0a4020 30%,#156030 50%,#0a4020 75%,#041a0c 100%)!important;
  color:#a0e8b8!important;
  border:1px solid rgba(80,200,120,0.35)!important;
  border-radius:var(--rs)!important;
  padding:.82rem 1.8rem!important;
  font-family:'Cinzel',serif!important;font-size:.8rem!important;
  font-weight:700!important;letter-spacing:3px!important;
  text-transform:uppercase!important;width:100%!important;
  transition:all .35s cubic-bezier(.22,1,.36,1)!important;
  box-shadow:0 4px 20px rgba(20,100,50,0.4),inset 0 1px 0 rgba(80,200,120,0.2)!important;
}
.stDownloadButton>button:hover{transform:translateY(-3px)!important;box-shadow:0 8px 32px rgba(20,100,50,0.55)!important;}

/* ══ FILE UPLOADER ══ */
[data-testid="stFileUploader"]{
  background:linear-gradient(135deg,#fffef9,#fffbf0)!important;
  border:2px dashed rgba(184,134,11,0.5)!important;
  border-radius:var(--r)!important;
  padding:2rem!important;
  transition:all .3s!important;
  box-shadow:inset 0 3px 12px rgba(184,134,11,0.06)!important;
}
[data-testid="stFileUploader"]:hover{
  border-color:var(--g1)!important;
  box-shadow:0 0 40px rgba(184,134,11,0.12),inset 0 3px 12px rgba(184,134,11,0.06)!important;
}

/* ══ DIAGNOSIS BADGE ══ */
.diag-badge{
  display:inline-flex;align-items:center;gap:.6rem;
  font-family:'Cinzel',serif;font-size:1.1rem;font-weight:700;
  padding:.55rem 1.4rem;border-radius:6px;letter-spacing:1.5px;
}
.diag-normal{
  background:linear-gradient(135deg,rgba(20,100,50,0.12),rgba(40,180,90,0.06));
  color:#2a9a50;border:1.5px solid rgba(50,200,100,0.45);
  box-shadow:0 2px 14px rgba(30,150,70,0.15),inset 0 1px 0 rgba(100,255,150,0.2);
}
.diag-parkinson{
  background:linear-gradient(135deg,rgba(160,30,20,0.12),rgba(220,60,50,0.06));
  color:#c83030;border:1.5px solid rgba(220,80,70,0.45);
  box-shadow:0 2px 14px rgba(180,40,30,0.15),inset 0 1px 0 rgba(255,120,100,0.2);
}

/* ══ STAT TILES ══ */
.stat-tile{
  background:linear-gradient(160deg,#fffef9 0%,#fffbef 60%,#fff6e2 100%);
  border:1px solid rgba(184,134,11,0.45);
  border-radius:var(--r);
  padding:1.2rem 1rem;
  text-align:center;
  box-shadow:var(--s1);
  position:relative;overflow:hidden;
  transition:all .35s cubic-bezier(.22,1,.36,1);
}
.stat-tile::before{
  content:'';position:absolute;top:0;left:15%;right:15%;height:2px;
  background:linear-gradient(90deg,transparent,var(--g3),var(--g4),var(--g3),transparent);
}
.stat-tile::after{
  content:'';position:absolute;bottom:0;left:30%;right:30%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);
}
.stat-tile:hover{transform:translateY(-3px);box-shadow:var(--s2);}
.stat-value{
  font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:700;
  background:linear-gradient(135deg,#7a4f00,#c9a84c,#8a6000);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  line-height:1.1;margin-bottom:.3rem;
}
.stat-label{font-family:'Cinzel',serif;font-size:.56rem;color:var(--tx2);letter-spacing:2.5px;text-transform:uppercase;}

/* ══ RISK PILLS ══ */
.risk-low{background:linear-gradient(135deg,rgba(30,140,60,0.12),rgba(50,180,90,0.06));color:#1a8a40;border:1.5px solid rgba(50,180,90,0.4);border-radius:6px;padding:.3rem 1.1rem;font-size:.72rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1.5px;box-shadow:0 2px 10px rgba(30,140,60,0.1);}
.risk-moderate{background:linear-gradient(135deg,rgba(200,150,0,0.12),rgba(230,180,20,0.06));color:#b08000;border:1.5px solid rgba(220,170,20,0.45);border-radius:6px;padding:.3rem 1.1rem;font-size:.72rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1.5px;box-shadow:0 2px 10px rgba(200,150,0,0.1);}
.risk-high{background:linear-gradient(135deg,rgba(180,30,20,0.12),rgba(220,60,50,0.06));color:#b82020;border:1.5px solid rgba(220,70,60,0.45);border-radius:6px;padding:.3rem 1.1rem;font-size:.72rem;font-family:'Cinzel',serif;font-weight:600;letter-spacing:1.5px;box-shadow:0 2px 10px rgba(180,30,20,0.1);}

/* ══ PROBABILITY BARS ══ */
.prob-row{margin:.7rem 0;}
.prob-label{font-family:'Cinzel',serif;font-size:.63rem;color:var(--tx2);margin-bottom:.35rem;display:flex;justify-content:space-between;letter-spacing:1.2px;}
.prob-track{
  background:rgba(0,0,0,0.05);border-radius:6px;height:10px;overflow:hidden;
  border:1px solid rgba(184,134,11,0.15);
  box-shadow:inset 0 2px 5px rgba(0,0,0,0.06);
}
.prob-fill-n{height:100%;border-radius:6px;background:linear-gradient(90deg,#0a4a28,#1a8a48,#3ab86a);transition:width 1.2s cubic-bezier(.22,1,.36,1);box-shadow:0 1px 6px rgba(30,140,70,0.3);}
.prob-fill-p{height:100%;border-radius:6px;background:linear-gradient(90deg,#6a0a08,#c02020,#e84040);transition:width 1.2s cubic-bezier(.22,1,.36,1);box-shadow:0 1px 6px rgba(180,30,20,0.3);}

/* ══ IMAGE FRAMES ══ */
.img-frame{
  border:1.5px solid rgba(184,134,11,0.5);
  border-radius:var(--r);overflow:hidden;
  background:linear-gradient(135deg,#f8f3e8,#f0ead8);
  box-shadow:var(--s1),inset 0 0 0 1px rgba(255,255,255,0.6);
  position:relative;transition:box-shadow .3s;
}
.img-frame:hover{box-shadow:var(--s2);}
.img-caption{
  font-family:'Cinzel',serif;font-size:.6rem;color:var(--tx3);
  text-align:center;padding:.5rem 0 .3rem;letter-spacing:2px;text-transform:uppercase;
  background:linear-gradient(180deg,transparent,rgba(184,134,11,0.04));
}
.hm-legend{display:flex;align-items:center;gap:.7rem;font-family:'Cinzel',serif;font-size:.6rem;color:var(--tx3);margin-top:.7rem;letter-spacing:1.5px;}
.hm-bar{flex:1;height:7px;border-radius:4px;background:linear-gradient(90deg,#00008b,#0040ff,#00c0ff,#00ff80,#ffff00,#ff8000,#ff0000);box-shadow:0 1px 4px rgba(0,0,0,0.15);}

/* ══ TABS ══ */
.stTabs{margin-top:.6rem;}
.stTabs [data-baseweb="tab-list"]{
  background:linear-gradient(135deg,#fdf3dc,#fffef7,#fdf3dc)!important;
  border:1.5px solid rgba(184,134,11,0.4)!important;
  border-radius:var(--rl)!important;
  padding:.45rem .6rem!important;gap:.4rem!important;
  justify-content:center!important;align-items:center!important;
  box-shadow:0 4px 20px rgba(184,134,11,0.10),inset 0 1px 0 rgba(255,255,255,0.8)!important;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Cinzel',serif!important;font-size:.72rem!important;
  font-weight:600!important;color:#8a6020!important;
  border-radius:12px!important;padding:.7rem 1.8rem!important;
  letter-spacing:2px!important;transition:all .3s!important;
  white-space:nowrap!important;
}
.stTabs [data-baseweb="tab"]:hover{
  color:#5a3a00!important;
  background:rgba(184,134,11,0.10)!important;
  box-shadow:0 2px 10px rgba(184,134,11,0.12)!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,rgba(184,134,11,0.22),rgba(232,201,106,0.14))!important;
  color:#3a1800!important;
  border:1.5px solid rgba(184,134,11,0.5)!important;
  box-shadow:0 3px 14px rgba(184,134,11,0.20),inset 0 1px 0 rgba(255,240,180,0.5)!important;
}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.3rem!important;}

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"]{
  background:linear-gradient(175deg,#fdf3dc 0%,#fdf8ef 40%,#fdf3e0 100%)!important;
  border-right:2px solid rgba(184,134,11,0.35)!important;
  box-shadow:6px 0 30px rgba(184,134,11,0.10)!important;
}
[data-testid="stSidebar"]::before{
  content:'';position:absolute;top:0;bottom:0;right:0;width:1px;
  background:linear-gradient(180deg,transparent,rgba(232,201,106,0.6),transparent);
  pointer-events:none;
}
[data-testid="stSidebar"] *{font-family:'EB Garamond',serif!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-family:'Cinzel',serif!important;color:#8a5e00!important;letter-spacing:2.5px!important;}

/* Sidebar toggle */
[data-testid="stSidebarCollapsedControl"] button,[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button,button[kind="header"]{
  background:linear-gradient(135deg,#7a5a20,#b8902a,#e0c050,#f5d870,#e0c050,#b8902a,#7a5a20)!important;
  border:2px solid rgba(240,210,100,0.8)!important;border-radius:14px!important;
  width:46px!important;height:46px!important;min-width:46px!important;min-height:46px!important;
  box-shadow:0 4px 20px rgba(184,134,11,0.5),inset 0 1px 0 rgba(255,245,180,0.4)!important;
  transition:all .35s!important;
}
[data-testid="stSidebarCollapsedControl"] button:hover,[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover,button[kind="header"]:hover{
  transform:scale(1.12) rotate(5deg)!important;
  box-shadow:0 8px 28px rgba(184,134,11,0.65)!important;
}
[data-testid="stSidebarCollapsedControl"] button svg,[data-testid="collapsedControl"] button svg,
[data-testid="stSidebarCollapseButton"] button svg,button[kind="header"] svg{
  fill:#2a1000!important;stroke:#2a1000!important;width:20px!important;height:20px!important;
}

/* ══ SIDEBAR METRICS ══ */
[data-testid="stMetric"]{
  background:linear-gradient(145deg,#fffef9,#fff8e8)!important;
  border:1px solid rgba(184,134,11,0.45)!important;
  border-radius:var(--r)!important;
  padding:1rem 1.2rem!important;
  box-shadow:var(--s1)!important;
  position:relative;overflow:hidden;
}
[data-testid="stMetric"]::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--g3),transparent);
}
[data-testid="stMetricValue"]{font-family:'Playfair Display',serif!important;font-size:1.7rem!important;font-weight:700!important;color:#8a5e00!important;}
[data-testid="stMetricLabel"]{color:#8a6020!important;font-family:'Cinzel',serif!important;font-size:.57rem!important;letter-spacing:1.8px!important;text-transform:uppercase!important;}
[data-testid="stMetricDelta"]{font-family:'Cinzel',serif!important;font-size:.65rem!important;}
.stProgress>div>div>div{background:linear-gradient(90deg,var(--g2),var(--g3),var(--g1))!important;border-radius:4px!important;}
[data-testid="stDataFrame"]{border-radius:var(--r)!important;border:1.5px solid rgba(184,134,11,0.45)!important;overflow:hidden!important;box-shadow:var(--s1)!important;}

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg);border-radius:10px;}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--g2),var(--g1));border-radius:10px;}
::-webkit-scrollbar-thumb:hover{background:var(--g1);}

/* ══ HR ══ */
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.4),rgba(240,210,100,0.6),rgba(201,168,76,0.4),transparent)!important;margin:2rem 0!important;}

/* ══ ALERTS ══ */
.stAlert{border-radius:var(--r)!important;font-family:'EB Garamond',serif!important;border-left-width:4px!important;}

/* ══ ORNATE DIVIDER ══ */
.orn-div{text-align:center;margin:1.4rem 0;color:rgba(184,134,11,0.45);font-size:.7rem;letter-spacing:6px;}

/* ══ SHIMMER ANIMATION ══ */
@keyframes shimmer{0%{background-position:-200% center}100%{background-position:200% center}}
@keyframes pulse-gold{0%,100%{box-shadow:0 0 0 0 rgba(184,134,11,0.3)}50%{box-shadow:0 0 0 8px rgba(184,134,11,0)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}

/* ══ SHIMMER TEXT ══ */
.shimmer-text{
  background:linear-gradient(90deg,#7a4f00 0%,#c9a84c 30%,#f5e0a0 50%,#c9a84c 70%,#7a4f00 100%);
  background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:shimmer 4s linear infinite;
}

/* ══ GLOWING MODEL BADGE ══ */
.model-badge{
  display:inline-block;
  background:linear-gradient(135deg,rgba(184,134,11,0.15),rgba(232,201,106,0.08));
  border:1.5px solid rgba(184,134,11,0.45);
  border-radius:10px;padding:.7rem 2.2rem;
  font-family:'Cinzel',serif;font-size:.88rem;font-weight:700;
  color:#6a3a00;letter-spacing:2.5px;text-align:center;text-transform:uppercase;
  box-shadow:0 4px 20px rgba(184,134,11,0.15),inset 0 1px 0 rgba(255,240,180,0.4);
  position:relative;overflow:hidden;
}
.model-badge::before{
  content:'';position:absolute;top:0;left:-100%;width:60%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,240,180,0.25),transparent);
  animation:shimmer 3s infinite;
}

/* ══ FEATURE PILLS ══ */
.feat-pill{
  display:inline-block;
  background:linear-gradient(135deg,rgba(184,134,11,0.10),rgba(201,168,76,0.06));
  border:1px solid rgba(184,134,11,0.32);
  border-radius:24px;padding:.3rem 1rem;
  font-family:'Cinzel',serif;font-size:.62rem;color:#7a5000;letter-spacing:1.2px;
  transition:all .25s;cursor:default;
}
.feat-pill:hover{background:rgba(184,134,11,0.16);border-color:rgba(184,134,11,0.55);transform:translateY(-1px);box-shadow:0 3px 12px rgba(184,134,11,0.15);}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ABOUT DIALOG
# ══════════════════════════════════════════════════════════════════════════════
@st.dialog("📋 About the Project", width="large")
def _show_project_dialog():
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#b8860b;letter-spacing:2.5px;'
        'text-transform:uppercase;margin-bottom:.6rem;padding:.4rem .9rem;background:rgba(184,134,11,0.06);'
        'border-radius:6px;border-left:3px solid #c9a84c;display:inline-block;">'
        '⚗ B.Tech Final Year · 2025-26 · ECE · BVC College of Engineering</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Playfair Display,serif;font-size:1.45rem;font-weight:700;'
        'color:#1e1000;margin:.6rem 0 1rem;line-height:1.3;">'
        "Parkinson's Disease Detection Using Deep Learning on Brain MRI</div>",
        unsafe_allow_html=True)
    _d1,_d2=st.columns(2,gap='medium')
    for _col,_items in [
        (_d1,[('📦 Dataset','831 scans (Kaggle) · 610 Normal · 221 Parkinson<br>581 Train · 125 Val · 125 Test<br>WeightedRandomSampler for class balance'),
              ('⚙ Training','SEED=42 · IMG=224×224 · BS=16 · EPOCHS=30<br>MixUp(α=0.3) · Label Smoothing(ε=0.1)<br>CosineAnnealingLR · Early Stopping(p=7)')]),
        (_d2,[('📊 Results','HybridNet_EV: <b style="color:#1a6a38;">100%</b> accuracy<br>F1: 1.000 · Precision: 1.000 · Recall: 1.000<br>AUC: 1.0000 · 5-fold cross-validation'),
              ('🖥 Stack','PyTorch · timm · EfficientNet-B4 · ViT-B/16<br>Cross-Attention Fusion · Streamlit · ReportLab<br>Google Colab · Tesla T4 GPU')])]:
        with _col:
            for _t,_b in _items:
                st.markdown(f'<div style="background:linear-gradient(135deg,rgba(184,134,11,0.07),rgba(184,134,11,0.03));border:1px solid rgba(184,134,11,0.22);border-radius:10px;padding:1rem 1.2rem;margin-bottom:.8rem;box-shadow:0 2px 10px rgba(184,134,11,0.07);"><div style="font-family:Cinzel,serif;font-size:.58rem;color:#b8860b;letter-spacing:2px;text-transform:uppercase;margin-bottom:.5rem;">{_t}</div><div style="font-family:EB Garamond,serif;font-size:.94rem;color:#4a3008;line-height:1.75;">{_b}</div></div>',unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(26,120,60,0.08),rgba(26,120,60,0.04));'
        'border:1px solid rgba(26,120,60,0.25);border-left:4px solid #1a6a38;border-radius:10px;padding:1rem 1.3rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.58rem;color:#1a6a38;letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">🏆 Key Innovation</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.96rem;color:#0a2a18;line-height:1.8;">'
        '<strong>Cross-Attention Fusion</strong> — EfficientNet-B4 spatial feature maps fused with ViT-B/16 global '
        'patch tokens via multi-head cross-attention, combining local texture with global context '
        'for 100% test accuracy with full Grad-CAM explainability.</div></div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
logo_src  = get_logo_b64('logo.png')
logo_html = (
    f'<img src="{logo_src}" style="width:90px;height:90px;object-fit:contain;border-radius:50%;'
    f'border:3px solid rgba(201,168,76,0.7);'
    f'box-shadow:0 0 0 6px rgba(184,134,11,0.10),0 0 40px rgba(184,134,11,0.30),0 4px 20px rgba(0,0,0,0.15);'
    f'animation:float 4s ease-in-out infinite;"/>'
    if logo_src else
    '<div style="width:90px;height:90px;background:linear-gradient(145deg,#fdf3dc,#f0d880);'
    'border:3px solid rgba(201,168,76,0.7);border-radius:50%;display:flex;align-items:center;'
    'justify-content:center;font-size:2.5rem;'
    'box-shadow:0 0 0 6px rgba(184,134,11,0.10),0 0 40px rgba(184,134,11,0.30);'
    'animation:float 4s ease-in-out infinite;">🧠</div>'
)

st.markdown(
    '<div style="'
    'background:linear-gradient(175deg,#fdf0cc 0%,#fdf8ef 45%,#fdf8ef 100%);'
    'border-bottom:2px solid rgba(184,134,11,0.3);'
    'padding:3rem 2rem 2.4rem;'
    'display:flex;flex-direction:column;align-items:center;gap:1rem;'
    'position:relative;overflow:hidden;">'

    # Top triple bar
    '<div style="position:absolute;top:0;left:0;right:0;height:5px;'
    'background:linear-gradient(90deg,transparent 0%,#a07020 10%,#c9a84c 25%,#f0d870 40%,#f8e890 50%,#f0d870 60%,#c9a84c 75%,#a07020 90%,transparent 100%);"></div>'
    '<div style="position:absolute;top:6px;left:0;right:0;height:1px;'
    'background:linear-gradient(90deg,transparent,rgba(201,168,76,0.4),rgba(240,216,120,0.6),rgba(201,168,76,0.4),transparent);"></div>'
    '<div style="position:absolute;top:9px;left:0;right:0;height:1px;'
    'background:linear-gradient(90deg,transparent,rgba(201,168,76,0.15),transparent);"></div>'

    # Ornate corner decorations
    '<div style="position:absolute;top:20px;left:28px;font-family:serif;font-size:1rem;color:rgba(184,134,11,0.35);letter-spacing:5px;">✦ ✦ ✦</div>'
    '<div style="position:absolute;top:20px;right:28px;font-family:serif;font-size:1rem;color:rgba(184,134,11,0.35);letter-spacing:5px;">✦ ✦ ✦</div>'
    '<div style="position:absolute;bottom:20px;left:28px;font-family:serif;font-size:.7rem;color:rgba(184,134,11,0.2);letter-spacing:4px;">· · ·</div>'
    '<div style="position:absolute;bottom:20px;right:28px;font-family:serif;font-size:.7rem;color:rgba(184,134,11,0.2);letter-spacing:4px;">· · ·</div>'

    # Radial glow behind logo
    '<div style="position:absolute;top:40px;left:50%;transform:translateX(-50%);'
    'width:200px;height:200px;'
    'background:radial-gradient(ellipse,rgba(184,134,11,0.12) 0%,transparent 70%);'
    'border-radius:50%;pointer-events:none;"></div>'

    + logo_html +

    # Title with shimmer
    '<div class="shimmer-text" style="font-family:Cinzel,serif;font-size:3rem;font-weight:900;'
    'letter-spacing:10px;line-height:1;text-align:center;">'
    'NEUROSCAN&nbsp;AI</div>'

    '<div style="font-family:Cormorant Garamond,serif;font-size:1.05rem;font-style:italic;'
    'color:#9a7030;letter-spacing:5px;text-align:center;margin-top:-.2rem;">'
    "Parkinson\u2019s Detection \u00b7 Brain MRI \u00b7 Deep Learning</div>"

    # Divider
    '<div style="display:flex;align-items:center;gap:1.2rem;width:70%;max-width:440px;">'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.7));"></div>'
    '<span style="color:var(--g2,#c9a84c);font-size:.9rem;line-height:1;">⬡</span>'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(201,168,76,0.7),transparent);"></div>'
    '</div>'

    # Model badge
    '<div class="model-badge">'
    "HybridNet\u2009EV \u00b7 EfficientNet-B4 \u002b ViT Cross-Attention \u00b7 100% Accuracy"
    '</div>'

    # Feature pills
    '<div style="display:flex;gap:.7rem;flex-wrap:wrap;justify-content:center;">'
    '<span class="feat-pill">⚙ EfficientNet-B4 + ViT-B/16</span>'
    '<span class="feat-pill">◈ 100% Test Accuracy</span>'
    '<span class="feat-pill">✦ Grad-CAM XAI</span>'
    '<span class="feat-pill">⬡ Batch Analysis</span>'
    '<span class="feat-pill">📄 PDF Reports</span>'
    '</div>'

    # Guidance pill
    '<div style="display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;justify-content:center;">'
    '<div style="height:1px;width:50px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.5));"></div>'
    '<div style="background:linear-gradient(135deg,rgba(184,134,11,0.10),rgba(201,168,76,0.06));'
    'border:1px solid rgba(184,134,11,0.32);border-radius:24px;padding:.38rem 1.4rem;'
    'box-shadow:0 2px 10px rgba(184,134,11,0.08);">'
    '<span style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:.9rem;color:#9a7030;">'
    'Under the Esteemed Guidance of&nbsp;</span>'
    '<span style="font-family:Cinzel,serif;font-size:.84rem;font-weight:700;color:#6a3a00;">'
    'Mr. N P U V S N Pavan Kumar</span>'
    '<span style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:.82rem;color:#b8860b;">'
    '&nbsp;· Asst. Professor, ECE</span>'
    '</div>'
    '<div style="height:1px;width:50px;background:linear-gradient(90deg,rgba(184,134,11,0.5),transparent);"></div>'
    '</div>'

    # Bottom bar
    '<div style="position:absolute;bottom:0;left:0;right:0;height:2px;'
    'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),rgba(201,168,76,0.4),rgba(184,134,11,0.25),transparent);"></div>'
    '</div>',
    unsafe_allow_html=True)

_b1,_b2,_b3=st.columns([4,2,4])
with _b2:
    if st.button('📋 About Project',key='hero_about_btn'): _show_project_dialog()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="background:linear-gradient(160deg,#fdf0cc,#fdf8ef);'
        'border-bottom:2px solid rgba(184,134,11,0.3);'
        'padding:1.4rem 1rem 1.1rem;margin:-1rem -1rem .9rem;text-align:center;position:relative;">'
        '<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        'background:linear-gradient(90deg,transparent,#c9a84c,#f5e090,#c9a84c,transparent);"></div>'
        '<div style="font-size:2rem;margin-bottom:.3rem;animation:float 3s ease-in-out infinite;">🧠</div>'
        '<div style="font-family:Cinzel,serif;font-size:1rem;font-weight:900;color:#6a3a00;letter-spacing:4px;">⚕ NEUROSCAN AI</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.8rem;font-style:italic;color:#9a7030;margin-top:.2rem;">Model Control Panel</div>'
        '<div style="height:2px;background:linear-gradient(90deg,transparent,#c9a84c,transparent);margin:.7rem 0 0;"></div>'
        '</div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(26,120,60,0.12),rgba(40,160,80,0.06));'
        'border:1.5px solid rgba(26,120,60,0.4);border-left:4px solid #1a6a38;'
        'border-radius:12px;padding:.9rem 1.1rem;margin:.5rem 0 .9rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.55rem;color:#1a6a38;letter-spacing:2px;text-transform:uppercase;margin-bottom:.3rem;">✦ Active Model</div>'
        '<div style="font-family:Playfair Display,serif;font-size:1.05rem;font-weight:700;color:#0a3a1a;">HybridNet EV</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.88rem;color:#2a6a40;margin-top:.15rem;">EfficientNet-B4 + ViT-B/16</div>'
        '<div style="margin-top:.5rem;display:flex;gap:.4rem;flex-wrap:wrap;">'
        '<span style="background:rgba(26,120,60,0.12);border:1px solid rgba(26,120,60,0.3);border-radius:4px;padding:.15rem .6rem;font-family:Cinzel,serif;font-size:.55rem;color:#1a6a38;letter-spacing:1px;">✓ 100% Acc</span>'
        '<span style="background:rgba(26,120,60,0.12);border:1px solid rgba(26,120,60,0.3);border-radius:4px;padding:.15rem .6rem;font-family:Cinzel,serif;font-size:.55rem;color:#1a6a38;letter-spacing:1px;">AUC: 1.000</span>'
        '</div></div>',
        unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.4),transparent);margin:.4rem 0;"></div>',unsafe_allow_html=True)
    st.markdown('<div style="font-family:Cinzel,serif;font-size:.6rem;color:#b8860b;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.4rem;">◈ Performance</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.metric('Accuracy','100%')
    with c2: st.metric('AUC','1.000')
    c3,c4=st.columns(2)
    with c3: st.metric('F1-Score','1.000')
    with c4: st.metric('Recall','100%')

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.4),transparent);margin:.5rem 0;"></div>',unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#b8860b;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.5rem;">⚗ Architecture</div>'
        '<div style="background:linear-gradient(135deg,rgba(184,134,11,0.06),rgba(184,134,11,0.02));'
        'border:1px solid rgba(184,134,11,0.2);border-radius:10px;padding:.8rem 1rem;">'
        '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3808;line-height:2.1;">'
        '🔹 EfficientNet-B4 backbone<br>🔹 ViT-B/16 transformer<br>'
        '🔹 Cross-Attention Fusion<br>🔹 105M total parameters<br>🔹 44.3M trainable params'
        '</div></div>',
        unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.4),transparent);margin:.5rem 0;"></div>',unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#b8860b;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.4rem;">✦ Features</div>'
        '<div style="display:flex;flex-direction:column;gap:.3rem;">'
        + ''.join(f'<div style="background:linear-gradient(135deg,rgba(184,134,11,0.05),transparent);border-left:2px solid rgba(201,168,76,0.5);border-radius:0 6px 6px 0;padding:.4rem .8rem;font-family:EB Garamond,serif;font-size:.9rem;color:#5a3808;">{f}</div>' for f in ['⚜ Single-scan analysis','⚜ Batch processing','⚜ Grad-CAM heatmap','⚜ Risk scoring','⚜ Gold PDF report','⚜ CSV export'])
        + '</div>',
        unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.4),transparent);margin:.5rem 0;"></div>',unsafe_allow_html=True)
    st.warning("⚠️ Research/academic use only.")
    st.markdown('<div style="font-family:Cinzel,serif;font-size:.54rem;color:#9a7030;letter-spacing:1px;text-align:center;margin-top:.4rem;">💡 Single model · CPU-optimised · Memory safe</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_scan,tab_batch,tab_about=st.tabs(['🧠  Single MRI Analysis','📦  Batch Analysis','👥  About Us'])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab_scan:
    st.markdown('<div class="sec-head"><span class="ornament">⚕</span>HybridNet EV — EfficientNet-B4 + ViT Cross-Attention</div>',unsafe_allow_html=True)

    # Architecture showcase card
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(26,120,60,0.07),rgba(184,134,11,0.06),rgba(26,80,160,0.04));'
        'border:1.5px solid rgba(184,134,11,0.35);border-radius:var(--r,14px);'
        'padding:1.2rem 1.8rem;margin-bottom:1.4rem;'
        'box-shadow:0 4px 24px rgba(184,134,11,0.10),inset 0 1px 0 rgba(255,255,255,0.8);">'
        '<div style="display:flex;align-items:center;gap:2rem;flex-wrap:wrap;">'
        '<div style="font-size:2.5rem;animation:float 4s ease-in-out infinite;">🔀</div>'
        '<div style="flex:1;min-width:200px;">'
        '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#b8860b;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.25rem;">Active Model</div>'
        '<div style="font-family:Playfair Display,serif;font-size:1.15rem;font-weight:700;color:#1e1000;">'
        'HybridNet_EV &nbsp;—&nbsp; EfficientNet-B4 + ViT Cross-Attention Fusion</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.92rem;color:#6b4c11;margin-top:.2rem;">'
        '105M params · 44.3M trainable · Grad-CAM · CPU-optimised</div>'
        '</div>'
        '<div style="display:flex;gap:.8rem;flex-wrap:wrap;">'
        '<div style="text-align:center;background:linear-gradient(135deg,rgba(26,120,60,0.12),rgba(26,120,60,0.06));border:1.5px solid rgba(26,120,60,0.35);border-radius:10px;padding:.6rem 1rem;">'
        '<div style="font-family:Playfair Display,serif;font-size:1.6rem;font-weight:900;color:#1a7a3c;">100%</div>'
        '<div style="font-family:Cinzel,serif;font-size:.52rem;color:#2a8a50;letter-spacing:1.5px;">TEST ACC</div>'
        '</div>'
        '<div style="text-align:center;background:linear-gradient(135deg,rgba(184,134,11,0.12),rgba(184,134,11,0.06));border:1.5px solid rgba(184,134,11,0.35);border-radius:10px;padding:.6rem 1rem;">'
        '<div style="font-family:Playfair Display,serif;font-size:1.6rem;font-weight:900;color:#8a5e00;">1.000</div>'
        '<div style="font-family:Cinzel,serif;font-size:.52rem;color:#9a6e00;letter-spacing:1.5px;">AUC ROC</div>'
        '</div>'
        '</div></div></div>',
        unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.35),rgba(240,210,100,0.5),rgba(201,168,76,0.35),transparent);margin:.5rem 0 1.2rem;"></div>',unsafe_allow_html=True)

    col_form,col_upload=st.columns([1,1],gap='large')

    with col_form:
        st.markdown('<div class="sec-head"><span class="ornament">✦</span>Patient Information</div>',unsafe_allow_html=True)
        st.markdown('<div class="card">',unsafe_allow_html=True)
        patient_name=st.text_input('Full Name *',placeholder="Patient's full name")
        c1,c2=st.columns(2)
        with c1: patient_age=st.number_input('Age *',0,120,45)
        with c2: patient_gender=st.selectbox('Gender *',['Male','Female','Other'])
        c3,c4=st.columns(2)
        with c3: patient_id=st.text_input('Patient ID *',placeholder='P-2024-0001')
        with c4: scan_date=st.date_input('Scan Date *',value=datetime.now())
        referring_doctor=st.text_input('Referring Doctor',placeholder='Dr. Name (optional)')
        medical_history=st.text_area('Medical History',placeholder='Relevant history (optional)',height=88)
        st.markdown('</div>',unsafe_allow_html=True)

    with col_upload:
        st.markdown('<div class="sec-head"><span class="ornament">◈</span>MRI Image</div>',unsafe_allow_html=True)
        st.markdown('<div class="card">',unsafe_allow_html=True)
        uploaded_file=st.file_uploader('Upload Brain MRI Scan',type=['png','jpg','jpeg'],help='PNG / JPG · any resolution')
        if uploaded_file:
            image=Image.open(uploaded_file)
            st.markdown('<div class="img-frame">',unsafe_allow_html=True)
            st.image(image,caption='Uploaded MRI',use_column_width=True)
            st.markdown('</div>',unsafe_allow_html=True)
            w,h=image.size
            st.markdown(f'<div style="font-family:Cinzel,monospace;font-size:.62rem;color:#9a7030;margin-top:.5rem;text-align:center;letter-spacing:1px;">{image.mode} · {w}×{h}px · {uploaded_file.size/1024:.1f} KB</div>',unsafe_allow_html=True)
        else:
            st.info('Upload a brain MRI scan to begin analysis.')
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.35),rgba(240,210,100,0.5),rgba(201,168,76,0.35),transparent);margin:1.5rem 0;"></div>',unsafe_allow_html=True)
    st.markdown('<div class="sec-head"><span class="ornament">⚗</span>Analysis & Results</div>',unsafe_allow_html=True)

    btn_col,report_col=st.columns([1,1],gap='large')

    with btn_col:
        if st.button('⚕ Analyze MRI Scan'):
            if not patient_name.strip(): st.error("Please enter the patient's full name.")
            elif not patient_id.strip(): st.error('Please enter a Patient ID.')
            elif uploaded_file is None: st.error('Please upload a brain MRI scan.')
            else:
                with st.spinner('Running HybridNet_EV analysis…'):
                    st.session_state.patient_data={'name':patient_name.strip(),'age':patient_age,'gender':patient_gender,'patient_id':patient_id.strip(),'scan_date':scan_date.strftime('%Y-%m-%d'),'doctor':referring_doctor.strip() or '—','medical_history':medical_history.strip()}
                    try:
                        download_model(MODEL_FILENAME)
                        model,classes,device=load_model_cached(MODEL_FILENAME)
                        result=predict(model,device,classes,image)
                        st.session_state.prediction_result=result
                        st.session_state.prediction_made=True
                        st.success('Analysis complete!')
                        st.balloons()
                        st.rerun()
                    except Exception as e: st.error(f'Error: {e}')

    if st.session_state.prediction_made:
        r=st.session_state.prediction_result
        is_n=not r.get('is_parkinson',False)
        dc='diag-normal' if is_n else 'diag-parkinson'
        rc=f"risk-{r['risk_level'].lower()}"

        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:Cinzel,serif;font-size:.72rem;font-weight:600;'
            'color:#8a5e00;letter-spacing:3px;text-transform:uppercase;'
            'text-align:center;margin-bottom:1.3rem;padding-bottom:.8rem;'
            'border-bottom:1px solid rgba(184,134,11,0.15);">◈ Diagnostic Summary ◈</div>',
            unsafe_allow_html=True)
        d1,d2,d3,d4=st.columns(4)
        with d1:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">Diagnosis</div><div style="margin-top:.6rem;"><span class="{dc} diag-badge">{"✦" if is_n else "⚠"} {r["prediction"].upper()}</span></div></div>',unsafe_allow_html=True)
        with d2:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">Confidence</div><div class="stat-value">{r["confidence"]:.1f}%</div></div>',unsafe_allow_html=True)
        with d3:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">Risk Level</div><div style="margin-top:.65rem;"><span class="{rc}">{r["risk_level"]} Risk</span></div></div>',unsafe_allow_html=True)
        with d4:
            st.markdown(f'<div class="stat-tile"><div class="stat-label">Model</div><div style="font-family:EB Garamond,serif;font-size:.82rem;color:#a89060;margin-top:.4rem;line-height:1.5;">HybridNet EV<br><span style="font-size:.75rem;color:#c9a84c;">EfficientNet-B4+ViT</span></div></div>',unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1.2rem;padding:1rem 1.2rem;background:linear-gradient(135deg,rgba(184,134,11,0.04),transparent);border-radius:8px;border:1px solid rgba(184,134,11,0.12);">',unsafe_allow_html=True)
        st.markdown(
            f'<div class="prob-row"><div class="prob-label"><span>✦ Normal</span><span style="font-weight:600;">{r["normal_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-n" style="width:{r["normal_prob"]:.1f}%"></div></div></div>'
            f'<div class="prob-row"><div class="prob-label"><span>⚠ Parkinson\'s Disease</span><span style="font-weight:600;">{r["parkinson_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-p" style="width:{r["parkinson_prob"]:.1f}%"></div></div></div>',
            unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

        if r.get('cam_overlay') is not None:
            st.markdown('<div class="sec-head"><span class="ornament">🔬</span>Grad-CAM Explainability</div>',unsafe_allow_html=True)
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.markdown(
                '<p style="font-family:EB Garamond,serif;font-size:.97rem;color:#9a7850;margin-bottom:1.3rem;line-height:1.75;">'
                "Grad-CAM highlights the brain regions that most influenced the model's decision. "
                '<strong style="color:#c9a84c;">Warmer colours (red/yellow)</strong> indicate high neural attention — '
                'the regions most associated with the diagnosis.</p>',unsafe_allow_html=True)
            ic1,ic2,ic3=st.columns(3)
            for col,img_obj,cap in [(ic1,r['image'],'Original MRI'),(ic2,r['cam_heatmap'],'Attention Heatmap'),(ic3,r['cam_overlay'],'Grad-CAM Overlay')]:
                with col:
                    st.markdown('<div class="img-frame">',unsafe_allow_html=True)
                    st.image(img_obj,use_column_width=True)
                    st.markdown(f'<div class="img-caption">{cap}</div>',unsafe_allow_html=True)
                    st.markdown('</div>',unsafe_allow_html=True)
            st.markdown(
                '<div class="hm-legend"><span>Low attention</span><div class="hm-bar"></div><span>High attention</span></div>',
                unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

    with report_col:
        if st.session_state.prediction_made:
            if st.button('📜 Generate PDF Report'):
                with st.spinner('Building report…'):
                    try:
                        pdf=build_pdf(st.session_state.patient_data,st.session_state.prediction_result)
                        p=st.session_state.patient_data
                        fn=f"NeuroScan_{p['patient_id']}_{datetime.now().strftime('%Y%m%d')}.pdf"
                        st.download_button('⬇ Download PDF Report',data=pdf,file_name=fn,mime='application/pdf')
                        st.success('PDF ready!')
                    except Exception as e: st.error(f'PDF error: {e}')
        else:
            st.info('Run an analysis first to generate a report.')

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="sec-head"><span class="ornament">📦</span>Batch MRI Analysis</div>',unsafe_allow_html=True)
    st.markdown('<p style="font-family:EB Garamond,serif;font-size:1.02rem;color:#9a7850;margin-bottom:1.3rem;line-height:1.75;">Upload multiple brain MRI scans for simultaneous analysis. Results include per-scan predictions, confidence scores, Grad-CAM visualisations and a downloadable CSV summary.</p>',unsafe_allow_html=True)

    batch_files=st.file_uploader('Upload Multiple Brain MRI Scans',type=['png','jpg','jpeg'],accept_multiple_files=True,key='batch_uploader')
    if batch_files:
        st.info(f'{len(batch_files)} file(s) ready.')
        if st.button('⚕ Run Batch Analysis'):
            try:
                download_model(MODEL_FILENAME)
                model,classes,device=load_model_cached(MODEL_FILENAME)
            except Exception as e: st.error(f'Load failed: {e}'); st.stop()
            results=[];prog=st.progress(0);status=st.empty()
            for i,f in enumerate(batch_files):
                status.markdown(f'<p style="font-family:Cinzel,serif;font-size:.75rem;color:#c9a84c;letter-spacing:1px;">Processing {f.name} ({i+1}/{len(batch_files)})…</p>',unsafe_allow_html=True)
                try:
                    res=predict(model,device,classes,Image.open(f)); res['filename']=f.name; results.append(res)
                except Exception as e: st.warning(f'Skipped {f.name}: {e}')
                prog.progress((i+1)/len(batch_files)); gc.collect()
            st.session_state.batch_results=results; status.empty()
            st.success(f'Done! {len(results)} scan(s) processed.'); st.rerun()

    if st.session_state.batch_results:
        R=st.session_state.batch_results
        nt=len(R); nn=sum(1 for r in R if r['class_idx']==0); np2=nt-nn; ac=np.mean([r['confidence'] for r in R])
        st.markdown('<div class="sec-head"><span class="ornament">◈</span>Summary</div>',unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4)
        with m1: st.metric('Total Scans',nt)
        with m2: st.metric('Normal',nn)
        with m3: st.metric("Parkinson's",np2)
        with m4: st.metric('Avg Confidence',f'{ac:.1f}%')

        st.markdown('<div class="sec-head"><span class="ornament">⬡</span>Distribution</div>',unsafe_allow_html=True)
        ch_c,tb_c=st.columns([1,1],gap='large')
        with ch_c:
            fig,ax=plt.subplots(figsize=(4.5,3.8),facecolor='#fdf8ef'); ax.set_facecolor('#fdf8ef')
            if nn>0 or np2>0:
                w,t,at=ax.pie([nn,np2],labels=['Normal',"Parkinson's"],autopct='%1.0f%%',colors=['#2a9a50','#c83030'],startangle=90,wedgeprops=dict(edgecolor='#1e1000',linewidth=2,width=0.7),textprops=dict(color='#1e1000',fontsize=10))
                for a2 in at: a2.set_color('#ffffff'); a2.set_fontweight('bold')
            ax.set_title('Scan Distribution',color='#6b4c11',fontsize=10,pad=10)
            plt.tight_layout(); st.pyplot(fig); plt.close(); gc.collect()
        with tb_c:
            df=pd.DataFrame([{'File':r['filename'],'Prediction':r['prediction'],'Confidence':f"{r['confidence']:.1f}%",'Normal %':f"{r['normal_prob']:.1f}%","Parkinson's %":f"{r['parkinson_prob']:.1f}%",'Risk':r['risk_level']} for r in R])
            st.dataframe(df,use_container_width=True,height=280)
            st.download_button('⬇ Download CSV',data=df.to_csv(index=False),file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",mime='text/csv')

        st.markdown('<div class="sec-head"><span class="ornament">🔬</span>Per-Image Results</div>',unsafe_allow_html=True)
        for r in R:
            is_n2=not r.get('is_parkinson',False); col2='#2a9a50' if is_n2 else '#c83030'; icon='✦' if is_n2 else '⚠'
            rc1,rc2,rc3,rc4=st.columns([1,2,1,1])
            with rc1:
                st.markdown('<div class="img-frame">',unsafe_allow_html=True); st.image(r['image'],use_column_width=True); st.markdown('</div>',unsafe_allow_html=True)
            with rc2:
                st.markdown(f'<p style="font-family:Cinzel,serif;font-size:.62rem;color:#9a7030;margin-bottom:.3rem;letter-spacing:1px;">{r["filename"]}</p><p style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;color:{col2};margin:0;">{icon} {r["prediction"]}</p><p style="font-family:Cinzel,serif;font-size:.6rem;color:#9a7030;margin-top:.35rem;letter-spacing:1px;">{r["timestamp"]}</p>',unsafe_allow_html=True)
            with rc3:
                st.metric('Confidence',f"{r['confidence']:.1f}%"); st.metric('Normal %',f"{r['normal_prob']:.1f}%")
            with rc4:
                if r.get('cam_overlay'):
                    st.markdown('<div class="img-frame">',unsafe_allow_html=True); st.image(r['cam_overlay'],use_column_width=True); st.markdown('<div class="img-caption">Grad-CAM</div></div>',unsafe_allow_html=True)
            st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.2),transparent);margin:.8rem 0;"></div>',unsafe_allow_html=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab_about:
    clg_logo=get_logo_b64('bvcr.jpg')
    clg_html=(f'<img src="{clg_logo}" style="width:115px;height:115px;object-fit:contain;margin-bottom:1.3rem;border:3px solid rgba(184,134,11,0.55);border-radius:12px;box-shadow:0 8px 32px rgba(184,134,11,0.22),0 0 0 6px rgba(184,134,11,0.07);"/>' if clg_logo else '<div style="font-size:4.5rem;margin-bottom:1.3rem;">🏛</div>')

    st.markdown(
        '<div style="background:linear-gradient(165deg,#fdf0cc 0%,#fdf8ef 45%,#fce8b0 100%);'
        'border:2px solid rgba(184,134,11,0.38);border-radius:20px;'
        'padding:3.2rem 2.5rem 2.8rem;margin-bottom:2.2rem;text-align:center;'
        'position:relative;overflow:hidden;'
        'box-shadow:0 8px 48px rgba(184,134,11,0.12),inset 0 1px 0 rgba(255,255,255,0.8);">'
        '<div style="position:absolute;top:0;left:0;right:0;height:5px;'
        'background:linear-gradient(90deg,transparent,#a07020,#c9a84c,#f0d870,#f8ea90,#f0d870,#c9a84c,#a07020,transparent);"></div>'
        '<div style="position:absolute;top:6px;left:0;right:0;height:1px;'
        'background:linear-gradient(90deg,transparent,rgba(201,168,76,0.45),transparent);"></div>'
        '<div style="position:absolute;top:40px;left:40px;font-size:1.2rem;color:rgba(184,134,11,0.3);font-family:serif;">✦</div>'
        '<div style="position:absolute;top:40px;right:40px;font-size:1.2rem;color:rgba(184,134,11,0.3);font-family:serif;">✦</div>'
        + clg_html +
        '<div style="font-family:Cinzel,serif;font-size:2rem;font-weight:900;color:#1e1000;letter-spacing:4px;margin-bottom:.45rem;">BVC College of Engineering</div>'
        '<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:1.08rem;color:#b8860b;letter-spacing:3.5px;margin-bottom:.35rem;">Rajahmundry, Andhra Pradesh</div>'
        '<div style="margin-bottom:1.1rem;"><a href="https://bvcr.edu.in" target="_blank" style="font-family:EB Garamond,serif;font-size:.95rem;color:#b8860b;text-decoration:none;border-bottom:1px solid rgba(184,134,11,0.4);">bvcr.edu.in</a></div>'
        '<div style="display:flex;gap:.9rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.3rem;">'
        + ''.join(f'<span style="background:rgba({bg});border:1.5px solid rgba({bdr});border-radius:6px;padding:.38rem 1.2rem;font-family:Cinzel,serif;font-size:.68rem;color:{col};letter-spacing:1.5px;font-weight:600;box-shadow:0 2px 8px rgba({sh});">{txt}</span>' for txt,bg,bdr,col,sh in [
            ('🎓 Autonomous','184,134,11,0.10','184,134,11,0.38','#7a4f00','184,134,11,0.08'),
            ('✓ NAAC A Grade','30,132,73,0.09','30,132,73,0.35','#1a6a38','30,132,73,0.08'),
            ('⚙ AICTE Approved','0,100,180,0.07','0,100,180,0.28','#004a90','0,100,180,0.06'),
            ('📜 Affiliated JNTUK','140,40,40,0.08','140,40,40,0.28','#7a1a1a','140,40,40,0.06'),
        ])
        + '</div>'
        '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a5a1a;line-height:1.85;max-width:680px;margin:0 auto;">'
        'Premier autonomous institution in Andhra Pradesh permanently affiliated to JNTUK. '
        'NAAC A Grade · AICTE Approved · Known for academic excellence and outstanding placements.'
        '</div>'
        '<div style="position:absolute;bottom:0;left:0;right:0;height:2px;'
        'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),rgba(201,168,76,0.5),rgba(184,134,11,0.3),transparent);"></div>'
        '</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sec-head" style="margin-top:1rem;"><span class="ornament">⚗</span>About the Project</div>',unsafe_allow_html=True)
    pc1,pc2=st.columns([3,2],gap='large')
    with pc1:
        st.markdown(
            '<div class="glass-card">'
            '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#b8860b;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.9rem;">⚗ Project Overview</div>'
            '<div style="font-family:Playfair Display,serif;font-size:1.3rem;font-weight:700;color:#1e1000;margin-bottom:1rem;line-height:1.35;">'
            "Parkinson's Disease Detection Using Deep Learning on Brain MRI</div>"
            '<div style="font-family:EB Garamond,serif;font-size:1.02rem;color:#4a3008;line-height:1.9;margin-bottom:1.3rem;">'
            'B.Tech Final Year Project (2025-26) at ECE, BVC College of Engineering. '
            'Deploys <strong style="color:#7a4f00;">HybridNet_EV</strong> — a novel architecture fusing '
            '<strong style="color:#7a4f00;">EfficientNet-B4</strong> spatial features with '
            '<strong style="color:#7a4f00;">ViT-B/16</strong> global patch tokens via '
            '<strong style="color:#7a4f00;">Multi-Head Cross-Attention Fusion</strong>. '
            'Trained on 831 brain MRI scans achieving '
            '<strong style="color:#1a6a38;">100% test accuracy</strong> with full Grad-CAM explainability.'
            '</div>'
            '<div style="display:flex;gap:.6rem;flex-wrap:wrap;">'
            + ''.join(f'<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.28);border-radius:6px;padding:.28rem .9rem;font-family:Cinzel,serif;font-size:.62rem;color:#7a4f00;letter-spacing:1px;transition:all .2s;">{t}</span>' for t in ['🔀 EfficientNet-B4+ViT','⚡ Cross-Attention','📊 Grad-CAM XAI','🔬 Brain MRI','✦ 5-Fold CV'])
            + '</div></div>',
            unsafe_allow_html=True)
    with pc2:
        st.markdown(
            '<div class="glass-card">'
            '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#b8860b;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:1rem;">📋 Project Specs</div>'
            '<div style="display:flex;flex-direction:column;gap:.7rem;">'
            + ''.join(f'<div style="display:flex;gap:.9rem;align-items:flex-start;padding:.75rem .85rem;background:rgba(184,134,11,0.05);border-radius:9px;border-left:3px solid rgba(201,168,76,0.6);"><span style="font-size:1.1rem;flex-shrink:0;">{ico}</span><div><div style="font-family:Cinzel,serif;font-size:.6rem;color:#8a5e00;font-weight:700;letter-spacing:1px;margin-bottom:.2rem;">{lbl}</div><div style="font-family:EB Garamond,serif;font-size:.9rem;color:#4a3008;line-height:1.65;">{val}</div></div></div>' for ico,lbl,val in [
                ('📦','Dataset','831 scans · 610 Normal · 221 PD<br>581 Train · 125 Val · 125 Test'),
                ('🤖','Deployed Model','HybridNet_EV<br>EfficientNet-B4 + ViT-B/16<br>Cross-Attention Fusion'),
                ('⚙','Training','LR=5e-5 · BS=16 · EPOCHS=30<br>MixUp(α=0.3) · AdamW · CosineAnnealingLR'),
                ('🏆','Results','Accuracy: 100% · F1: 1.000<br>AUC: 1.0000 · Recall: 100%'),
            ])
            + '</div></div>',
            unsafe_allow_html=True)

    # Guide Banner
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;"><span class="ornament">⭐</span>Who We Are</div>',unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,#fdf0cc,#fdf8ef,#fdf0cc);'
        'border:2px solid rgba(184,134,11,0.5);border-radius:18px;'
        'padding:1.8rem 2.5rem;margin-bottom:2rem;position:relative;overflow:hidden;'
        'box-shadow:0 6px 32px rgba(184,134,11,0.13),inset 0 1px 0 rgba(255,255,255,0.8);">'
        '<div style="position:absolute;top:0;left:0;right:0;height:4px;'
        'background:linear-gradient(90deg,transparent,#c9a84c,#f5e090,#e8c060,#f5e090,#c9a84c,transparent);"></div>'
        '<div style="display:flex;align-items:center;gap:2rem;flex-wrap:wrap;">'
        '<div style="font-size:3.2rem;animation:float 3.5s ease-in-out infinite;">🎓</div>'
        '<div style="flex:1;">'
        '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#b8860b;letter-spacing:3px;text-transform:uppercase;margin-bottom:.35rem;">⭐ Under the Esteemed Guidance of</div>'
        '<div style="font-family:Playfair Display,serif;font-size:1.45rem;font-weight:700;color:#1e1000;line-height:1.2;margin-bottom:.3rem;">Mr. N P U V S N Pavan Kumar, M.Tech</div>'
        '<div style="font-family:EB Garamond,serif;font-size:.97rem;font-style:italic;color:#8a5e00;">Assistant Professor · Dept. of ECE · Deputy CoE III · BVC College of Engineering, Rajahmundry</div>'
        '</div>'
        '<div style="background:linear-gradient(135deg,#6a3a00,#b8860b,#c9a84c);border-radius:12px;padding:.9rem 1.5rem;text-align:center;flex-shrink:0;box-shadow:0 4px 16px rgba(184,134,11,0.35);">'
        '<div style="font-family:Cinzel,serif;font-size:.55rem;color:rgba(255,255,255,0.75);letter-spacing:2px;text-transform:uppercase;margin-bottom:.2rem;">Project Guide</div>'
        '<div style="font-size:1.8rem;">🏆</div>'
        '</div></div></div>',
        unsafe_allow_html=True)

    # Team cards
    _team=[
        {'roll':'236M5A0408','name':'G Srinivasu','role':'AI Model Architect & Backend','icon':'🤖','c1':'#0f2218','c2':'#1c5030','acc':'#4ade80','tags':['PyTorch','Model Training','Backend'],'desc':'Designed HybridNet_EV including the Cross-Attention Fusion architecture. Led backend integration and model optimisation on T4 GPU.'},
        {'roll':'226M1A0460','name':'S Anusha Devi','role':'UI/UX Designer & Frontend','icon':'🎨','c1':'#180f28','c2':'#50206a','acc':'#d08af0','tags':['Streamlit','CSS Design','UX'],'desc':'Crafted the royal gold aesthetic UI including hero banner, animated tabs, PDF layout and all interactive features.'},
        {'roll':'226M1A0473','name':'V V Siva Vardhan','role':'Data Engineer & QA','icon':'📊','c1':'#0f1828','c2':'#183060','acc':'#60b0fa','tags':['Data Pipeline','Augmentation','Testing'],'desc':'Managed dataset ingestion, augmentation strategy (MixUp, ColorJitter, RandomErasing) and model evaluation.'},
        {'roll':'236M5A0415','name':'N L Sandeep','role':'Documentation & Integration','icon':'📝','c1':'#201408','c2':'#603018','acc':'#f0a060','tags':['PDF Reports','Integration','Docs'],'desc':'Built the gold PDF report system with ReportLab. Handled system integration and project report writing.'},
    ]
    tc1,tc2=st.columns(2,gap='large')
    for col,m in zip([tc1,tc2],_team[:2]):
        with col:
            tags=''.join(f'<span style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.18);border-radius:20px;padding:.22rem .75rem;font-family:Cinzel,serif;font-size:.56rem;color:rgba(255,255,255,0.82);letter-spacing:.8px;">{t}</span>' for t in m['tags'])
            st.markdown(
                f'<div style="background:linear-gradient(145deg,{m["c1"]},{m["c2"]});'
                f'border:1px solid rgba(255,255,255,0.08);border-radius:18px;padding:1.9rem 1.7rem;'
                f'position:relative;overflow:hidden;box-shadow:0 10px 40px rgba(0,0,0,0.3);">'
                f'<div style="position:absolute;top:-40px;right:-40px;width:120px;height:120px;'
                f'background:radial-gradient(circle,{m["acc"]}20,transparent 70%);border-radius:50%;"></div>'
                f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,{m["acc"]},transparent);"></div>'
                f'<div style="position:absolute;bottom:0;left:20%;right:20%;height:1px;background:linear-gradient(90deg,transparent,{m["acc"]}40,transparent);"></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.1rem;">'
                f'<div style="width:58px;height:58px;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.13);border-radius:15px;display:flex;align-items:center;justify-content:center;font-size:1.9rem;box-shadow:0 4px 12px rgba(0,0,0,0.2);">{m["icon"]}</div>'
                f'<div style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);border-radius:7px;padding:.22rem .65rem;font-family:Cinzel,serif;font-size:.54rem;color:{m["acc"]};letter-spacing:1px;">{m["roll"]}</div>'
                f'</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.3rem;font-weight:700;color:#ffffff;margin-bottom:.22rem;line-height:1.25;">{m["name"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.6rem;color:{m["acc"]};letter-spacing:1.8px;text-transform:uppercase;margin-bottom:.85rem;">{m["role"]}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.92rem;color:rgba(255,255,255,0.72);line-height:1.75;margin-bottom:1.1rem;">{m["desc"]}</div>'
                f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;">{tags}</div>'
                f'</div>',unsafe_allow_html=True)
    st.markdown('<div style="height:.9rem;"></div>',unsafe_allow_html=True)
    tc3,tc4=st.columns(2,gap='large')
    for col,m in zip([tc3,tc4],_team[2:]):
        with col:
            tags=''.join(f'<span style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.18);border-radius:20px;padding:.22rem .75rem;font-family:Cinzel,serif;font-size:.56rem;color:rgba(255,255,255,0.82);letter-spacing:.8px;">{t}</span>' for t in m['tags'])
            st.markdown(
                f'<div style="background:linear-gradient(145deg,{m["c1"]},{m["c2"]});'
                f'border:1px solid rgba(255,255,255,0.08);border-radius:18px;padding:1.9rem 1.7rem;'
                f'position:relative;overflow:hidden;box-shadow:0 10px 40px rgba(0,0,0,0.3);">'
                f'<div style="position:absolute;top:-40px;right:-40px;width:120px;height:120px;'
                f'background:radial-gradient(circle,{m["acc"]}20,transparent 70%);border-radius:50%;"></div>'
                f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,{m["acc"]},transparent);"></div>'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.1rem;">'
                f'<div style="width:58px;height:58px;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.13);border-radius:15px;display:flex;align-items:center;justify-content:center;font-size:1.9rem;box-shadow:0 4px 12px rgba(0,0,0,0.2);">{m["icon"]}</div>'
                f'<div style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);border-radius:7px;padding:.22rem .65rem;font-family:Cinzel,serif;font-size:.54rem;color:{m["acc"]};letter-spacing:1px;">{m["roll"]}</div>'
                f'</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.3rem;font-weight:700;color:#ffffff;margin-bottom:.22rem;line-height:1.25;">{m["name"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.6rem;color:{m["acc"]};letter-spacing:1.8px;text-transform:uppercase;margin-bottom:.85rem;">{m["role"]}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.92rem;color:rgba(255,255,255,0.72);line-height:1.75;margin-bottom:1.1rem;">{m["desc"]}</div>'
                f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;">{tags}</div>'
                f'</div>',unsafe_allow_html=True)

    # Guidance
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;"><span class="ornament">⭐</span>Project Guidance</div>',unsafe_allow_html=True)
    g1,g2,g3=st.columns(3,gap='medium')
    for col,rl,ico,rc,bg,name,tags2,desc in [
        (g1,'Project Guide','👨‍🏫','#b8860b','linear-gradient(160deg,#fdf0cc,#fdf8ef)','Mr. N P U V S N Pavan Kumar, M.Tech',['Asst. Professor','Dept. of ECE','Deputy CoE III'],'Primary guide who mentored model design, training strategy, and full deployment of NeuroScan AI.'),
        (g2,'Project Coordinator','📋','#1a6a38','linear-gradient(160deg,#edf8f0,#f5fdf7)','Mr. K Anji Babu, M.Tech',['Asst. Professor','Dept. of ECE'],'Coordinated project milestones, review sessions and facilitated academic compliance.'),
        (g3,'Head of Department','👨‍💼','#7a1a1a','linear-gradient(160deg,#fdf0f0,#fdf8f8)','Dr. S A Vara Prasad, Ph.D',['Professor & HOD','ECE Dept.','Chairman BoS'],'Provided departmental leadership and institutional resources for this Final Year Project.'),
    ]:
        th=''.join(f'<span style="background:rgba(0,0,0,0.04);border:1px solid rgba(0,0,0,0.10);border-radius:5px;padding:.18rem .7rem;font-family:Cinzel,serif;font-size:.57rem;color:{rc};letter-spacing:.8px;font-weight:600;">{t}</span> ' for t in tags2)
        with col:
            st.markdown(
                f'<div style="background:{bg};border:1.5px solid rgba(0,0,0,0.07);border-radius:16px;padding:1.9rem 1.5rem;text-align:center;position:relative;box-shadow:0 4px 24px rgba(0,0,0,0.07);">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,transparent,{rc},transparent);border-radius:16px 16px 0 0;"></div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:{rc};letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.9rem;font-weight:700;">⭐ {rl}</div>'
                f'<div style="width:68px;height:68px;background:linear-gradient(135deg,{rc}1a,{rc}0d);border:2px solid {rc}44;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.9rem;margin:0 auto .9rem;box-shadow:0 4px 16px {rc}20;">{ico}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.02rem;font-weight:700;color:#1e1000;margin-bottom:.65rem;line-height:1.4;">{name}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:.3rem;justify-content:center;margin-bottom:.85rem;">{th}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#4a3008;line-height:1.7;font-style:italic;">{desc}</div>'
                f'</div>',unsafe_allow_html=True)

    st.markdown('<br>',unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(160,20,10,0.05),rgba(200,40,30,0.02));'
        'border:1.5px solid rgba(180,30,20,0.25);border-left:5px solid #c0392b;'
        'border-radius:12px;padding:1.3rem 1.9rem;'
        'box-shadow:0 4px 20px rgba(160,20,10,0.06);">'
        '<div style="font-family:Cinzel,serif;font-size:.64rem;color:#c0392b;letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.45rem;">⚕ Medical Disclaimer</div>'
        '<div style="font-family:EB Garamond,serif;font-size:1.02rem;color:#700a0a;line-height:1.75;">'
        'This project is developed <strong>for academic and research purposes only</strong>. '
        "The AI system is not a certified medical device and must not replace professional clinical diagnosis. "
        'Always consult a qualified neurologist for medical advice.'
        '</div></div>',
        unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="height:2px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),rgba(201,168,76,0.5),rgba(240,210,100,0.6),rgba(201,168,76,0.5),rgba(184,134,11,0.25),transparent);margin:2.5rem 0 0;"></div>'
    '<div style="text-align:center;padding:1.8rem 2rem;background:linear-gradient(180deg,#fdf8ef,#fdf0cc);">'
    '<div style="font-family:Cinzel,serif;font-size:.58rem;color:#9a7030;letter-spacing:3.5px;text-transform:uppercase;margin-bottom:.6rem;">'
    'Research &amp; Educational Use Only · Not for Clinical Diagnosis'
    '</div>'
    '<div style="display:flex;align-items:center;gap:1rem;justify-content:center;margin-bottom:.6rem;">'
    '<div style="flex:1;max-width:100px;height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.4));"></div>'
    '<span style="color:rgba(201,168,76,0.5);font-size:.75rem;">✦ ✦ ✦</span>'
    '<div style="flex:1;max-width:100px;height:1px;background:linear-gradient(90deg,rgba(201,168,76,0.4),transparent);"></div>'
    '</div>'
    '<div style="font-family:EB Garamond,serif;font-size:.95rem;color:#9a7030;">'
    'NeuroScan AI · HybridNet_EV · EfficientNet-B4 + ViT · Grad-CAM · PyTorch · Streamlit'
    '</div></div>',
    unsafe_allow_html=True)
