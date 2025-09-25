import os
import io
import base64
import traceback
from datetime import datetime

import streamlit as st
from PIL import Image
import numpy as np
import yaml
from fpdf import FPDF

# ---------------- Config ----------------
MODEL_PATH = "best.pt"
TOP_K = 1   # only top prediction
CROP_KEYWORDS = ["cotton", "maize", "wheat", "rice", "sugarcane"]

# ---------------- UI CSS ----------------
st.set_page_config(page_title="Plant Sight", page_icon="🌿", layout="wide")
CSS = """<style>
body { background: #071016; color: #E6EEF3; font-family: 'Poppins', sans-serif; }
.header { display:flex; align-items:center; gap:16px; padding:18px; background:#178f2d; border-radius:12px; }
.title { font-size:28px; font-weight:900; color:white; margin:0; }
.subtitle { color:rgba(255,255,255,0.92); margin-top:2px; font-size:13px; }
.container { margin-top:22px; max-width:1100px; margin-left:auto; margin-right:auto; }
.card { background: rgba(255,255,255,0.02); border-radius:12px; padding:20px; }
.footer { margin-top:30px; color:#9aa4ad; font-size:12px; text-align:center; }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Model import ----------------
from ultralytics import YOLO
MODEL = YOLO(MODEL_PATH)

def extract_topk_from_result(result, k=TOP_K):
    try:
        arr = np.array(result.probs).flatten()
        if arr.size:
            idxs = np.argsort(-arr)[:k]
            return [(int(i), float(arr[i])) for i in idxs]
    except Exception:
        pass
    return []

# ---------------- Load labels/remedies ----------------
def load_labels_remedies(path="labels_remedies.yaml"):
    if not os.path.exists(path):
        return {}, {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    class_map, remedies_full = {}, {}
    for k, v in data.items():
        kid = int(k)
        class_map[kid] = v.get("name", f"Class {kid}")
        remedies_full[kid] = v
    return class_map, remedies_full

CLASS_MAPPING, REMEDIES_FULL = load_labels_remedies("labels_remedies.yaml")

# ---------------- Header ----------------
st.markdown(f"""
<div class="header">
    <div>
        <div class="title">Plant Sight</div>
        <div class="subtitle">Fast disease ID • Clear remedial steps • Farmer-friendly</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- About Section ----------------
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## About Plant Sight")
st.markdown("""
Plant Sight is an AI-powered crop disease detection platform.  
Currently, we support five key crops: **Cotton, Maize, Wheat, Rice, and Sugarcane.**  

Upload a clear close-up of the affected part (leaf, stem, boll, or ear), and our system instantly detects pests/diseases, 
offering structured, farmer-friendly guidance for cultural, biological, and chemical control methods.  
Our mission is to empower farmers with fast, accessible, and reliable crop protection insights.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- File Uploader ----------------
uploaded = st.file_uploader("Upload a plant image", type=["jpg","jpeg","png"])

# ---------------- Prediction ----------------
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    results = MODEL.predict(pil, save=False, verbose=False)
    r = results[0]
    preds_raw = extract_topk_from_result(r, k=1)

    if preds_raw:
        cid, conf = preds_raw[0]
        disease_name = CLASS_MAPPING.get(cid, f"Class {cid}")
        remedy = REMEDIES_FULL.get(cid, {})

        # Show prediction
        st.markdown(f"### 🔎 Prediction: {disease_name}")

        # Show remedies directly (no expander)
        st.markdown("### 🌱 Recommended Action")
        for section, content in remedy.items():
            if section.lower() != "name":
                st.markdown(f"**{section.capitalize()}**")
                if isinstance(content, str):
                    st.markdown(content.replace("-", "•"))

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
        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                           file_name="plantsight_report.pdf", mime="application/pdf")

# ---------------- Footer ----------------
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Plant Sight • Fast disease ID • Guidance only — consult local extension for chemicals & dosages • Made by Code_Avengers</div>', unsafe_allow_html=True)
