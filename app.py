import os
import io
import base64
import traceback
from datetime import datetime

import streamlit as st
from PIL import Image
import numpy as np
import yaml
from fpdf import FPDF   # for PDF export

# ---------------- Config ----------------
MODEL_PATH = "best.pt"
TOP_K = 3
CONF_THRESHOLD = 0.70
REJECTED_DIR = "rejected"
CROP_KEYWORDS = ["cotton", "maize", "wheat", "rice", "sugarcane"]

os.makedirs(REJECTED_DIR, exist_ok=True)

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="Plant Sight", page_icon="🌿", layout="wide")

# File uploader
uploaded = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])  # ✅ define uploaded here

# ---------------- Helpers ----------------
def normalize_conf(c):
    try:
        f = float(c)
    except Exception:
        return 0.0
    if f > 1.0: f = f / 100.0
    if f < 0: f = 0.0
    if f > 1.0: f = 1.0
    return f

def is_crop_name(name: str):
    if not name: return False
    n = name.lower()
    return any(kw in n for kw in CROP_KEYWORDS)

def load_labels_remedies(path="labels_remedies.yaml"):
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None, None
    class_map, remedies_full = {}, {}
    for k, v in data.items():
        try:
            kid = int(k)
            class_map[kid] = v.get("name", f"Class {kid}")
            remedies_full[kid] = v
        except:
            continue
    return class_map, remedies_full

# ---------------- Model ----------------
try:
    from ultralytics import YOLO
    MODEL = YOLO(MODEL_PATH)
except Exception as e:
    st.error("⚠️ Model load failed: " + str(e))
    st.stop()

CLASS_MAPPING, REMEDIES_FULL = load_labels_remedies("labels_remedies.yaml")

# ---------------- Prediction ----------------
if uploaded:
    try:
        uploaded_bytes = uploaded.getvalue()
        pil = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    except Exception as e:
        st.error("❌ Could not open image: " + str(e))
        st.stop()

    with st.spinner("Analyzing image..."):
        try:
            results = MODEL.predict(pil, save=False, verbose=False)
            r = results[0]
            preds_raw = []
            if hasattr(r, "probs") and r.probs is not None:
                arr = np.array(r.probs).flatten()
                idxs = np.argsort(-arr)[:TOP_K]
                preds_raw = [(int(i), float(arr[i])) for i in idxs]
        except Exception as e:
            st.error("❌ Prediction error: " + str(e))
            st.stop()

    display_preds = []
    for cid, conf in preds_raw:
        nconf = normalize_conf(conf)
        name = CLASS_MAPPING.get(int(cid), f"Class {cid}")
        display_preds.append((name, nconf, int(cid)))

    if display_preds:
        top_name, top_conf, top_id = display_preds[0]

        st.image(pil, caption="Uploaded Image", use_container_width=True)
        st.markdown(f"### 🔎 Prediction: {top_name}")
        st.markdown(f"**Confidence:** {int(top_conf * 100)}%")

        entry = REMEDIES_FULL.get(top_id, {})
        summary = entry.get("summary", "")
        details = entry.get("details", "")

        st.markdown("### 🌱 Recommended Action")
        st.write(summary)
        if details:
            st.markdown(details.replace("\n", "  \n"))

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
        pdf.cell(0, 10, f"Prediction: {top_name}", ln=1)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, f"Confidence: {int(top_conf*100)}%")

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Recommended Action", ln=1)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, summary)
        pdf.multi_cell(0, 8, details)

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                           file_name="plantsight_report.pdf",
                           mime="application/pdf",
                           use_container_width=True)

