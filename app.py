# app.py — Plant Sight (improved with PDF export, About section, better remedies display)
"""
Plant Sight - Streamlit app for crop disease detection.
"""

import os
import io
import base64
import traceback
from datetime import datetime
from fpdf import FPDF

import streamlit as st
from PIL import Image
import numpy as np
import yaml

# ---------------- Config ----------------
MODEL_PATH = "best.pt"          # place your trained model next to app.py
TOP_K = 1                       # just top prediction
REJECTED_DIR = "rejected"

os.makedirs(REJECTED_DIR, exist_ok=True)

# ---------------- UI CSS ----------------
st.set_page_config(page_title="Plant Sight", page_icon="🌿", layout="wide")
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800;900&display=swap');
:root{ --brand-green:#178f2d; --muted:#9aa4ad; }
body { background: #071016; color: #E6EEF3; font-family: 'Poppins', sans-serif; }
.header { display:flex; align-items:center; gap:16px; padding:18px; background: var(--brand-green); border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,0.5); }
.logo-box { width:64px; height:64px; border-radius:12px; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,0.06); }
.title { font-size:28px; font-weight:900; color:white; margin:0; }
.subtitle { color:rgba(255,255,255,0.92); margin-top:2px; font-size:13px; }
.container { margin-top:22px; max-width:1100px; margin-left:auto; margin-right:auto; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:20px; box-shadow: 0 8px 30px rgba(2,6,12,0.6); border: 1px solid rgba(255,255,255,0.04); }
.hero-title { font-size:34px; font-weight:900; margin:0; }
.hero-desc { color:#cde6d9; margin-top:10px; font-size:15px; line-height:1.6; }
.uploader { margin-top:18px; padding:14px; background:#0c1113; border-radius:10px; border:1px solid rgba(255,255,255,0.03); text-align: center; }
.footer { margin-top:30px; color:var(--muted); font-size:12px; text-align:center; }
.stButton>button { background-color: #178f2d; color: white; font-weight: 600; border-radius: 8px; border: none; }
@media (max-width: 768px) {
  .header { flex-direction: column; text-align:center; }
  .hero-title { font-size:24px; }
}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Helpers ----------------
def find_logo_file():
    for fname in os.listdir("."):
        if fname.lower().startswith("logo") and os.path.isfile(fname):
            return fname
    return None

def image_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# ---------------- Model import ----------------
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception as e:
    st.error("ultralytics import failed. Install it in your environment.")
    st.stop()

def load_model_safe(path):
    return YOLO(path)

def extract_topk_from_result(result, k=TOP_K):
    try:
        probs_obj = getattr(result, "probs", None)
        if probs_obj is not None:
            arr = np.array(probs_obj).flatten()
            if arr.size:
                idxs = np.argsort(-arr)[:k]
                return [(int(i), float(arr[i])) for i in idxs]
    except Exception:
        pass
    return []

# ---------------- Load labels/remedies YAML ----------------
def load_labels_remedies(path="labels_remedies.yaml"):
    if not os.path.exists(path):
        return {}, {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    class_map, remedies_full = {}, {}
    for k, v in data.items():
        try:
            kid = int(k)
            class_map[kid] = v.get("name", f"Class {kid}")
            remedies_full[kid] = v
        except:
            pass
    return class_map, remedies_full

CLASS_MAPPING, REMEDIES_FULL = load_labels_remedies("labels_remedies.yaml")

# ---------------- Header ----------------
logo_file = find_logo_file()
if logo_file:
    with open(logo_file, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:64px;height:64px;border-radius:10px;object-fit:cover;">'
else:
    logo_html = '<div class="logo-box"><span style="font-weight:800;color:#fff">PS</span></div>'

st.markdown(f"""
<div class="header">
    {logo_html}
    <div>
        <div class="title">Plant Sight</div>
        <div class="subtitle">Fast disease ID • Clear remedial steps • Farmer-friendly</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- About Section ----------------
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">About Plant Sight</div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero-desc">
Plant Sight is an AI-powered crop disease detection platform. Currently, we support five key crops:
<strong>Cotton, Maize, Wheat, Rice, and Sugarcane.</strong>  
Upload a clear close-up of the affected part (leaf, stem, boll, or ear), and our system instantly detects pests and diseases, 
offering structured, farmer-friendly guidance for cultural, biological, and chemical control methods.  
Our mission is to empower farmers with fast, accessible, and reliable crop protection insights.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- File uploader ----------------
st.markdown('<div class="uploader card">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload a plant image", type=["jpg","jpeg","png"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Model load ----------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    MODEL = load_model_safe(MODEL_PATH)
except Exception as e:
    st.error("Model load failed: " + str(e))
    st.stop()

# ---------------- Prediction ----------------
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    with st.spinner("Analyzing image..."):
        results = MODEL.predict(pil, save=False, verbose=False)
        r = results[0]
        preds_raw = extract_topk_from_result(r, k=1)

    if preds_raw:
        cid, conf = preds_raw[0]
        disease_name = CLASS_MAPPING.get(cid, f"Class {cid}")
        remedy = REMEDIES_FULL.get(cid, {})

        # Layout
        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(pil, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### 🔎 Prediction: {disease_name}")

            # Remedies (full structured)
            st.markdown("### 🌱 Recommended Action")
            for section, content in remedy.items():
                if section.lower() != "name":
                    st.markdown(f"**{section.capitalize()}**")
                    if isinstance(content, str):
                        st.markdown(content.replace("-", "•"))
            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- PDF Export ----------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Plant Sight Report", ln=1, align="C")

        # Add image
        img_path = "temp_input.png"
        pil.save(img_path)
        pdf.image(img_path, x=10, y=30, w=90)
        pdf.ln(100)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Prediction: {disease_name}", ln=1)

        pdf.set_font("Arial", "", 11)
        for section, content in remedy.items():
            if section.lower() != "name":
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, section.capitalize(), ln=1)
                pdf.set_font("Arial", "", 11)
                if isinstance(content, str):
                    for line in content.split("\n"):
                        pdf.multi_cell(0, 6, line)

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="plantsight_report.pdf", mime="application/pdf")

# ---------------- Footer ----------------
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Plant Sight • Fast disease ID • Made by Code_Avengers</div>', unsafe_allow_html=True)
