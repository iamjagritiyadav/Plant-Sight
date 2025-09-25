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
MODEL_PATH = "best.pt"          # place your trained model next to app.py
TOP_K = 3
CONF_THRESHOLD = 0.70          # require >= 0.70 confidence to show prediction/remedy
REJECTED_DIR = "rejected"
CROP_KEYWORDS = ["cotton", "maize", "wheat", "rice", "sugarcane"]  # used to reject non-crop images

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
.hero-desc { color:#cde6d9; margin-top:10px; font-size:15px; }
.uploader { margin-top:18px; padding:14px; background:#0c1113; border-radius:10px; border:1px solid rgba(255,255,255,0.03); text-align: center; }
.conf-bar { height:12px; background: rgba(255,255,255,0.06); border-radius:999px; overflow:hidden; margin-top:8px; }
.conf-fill { height:100%; background: linear-gradient(90deg,#178f2d,#2de36a); transition: width 0.5s ease-in-out; }
.small { color:#9aa4ad; font-size:13px; }
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

# ---------------- Header UI ----------------
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

# ---------------- Main container ----------------
st.markdown('<div class="container">', unsafe_allow_html=True)

# Hero / intro
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Plant Sight — Protect your crop, protect your livelihood</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-desc">Upload a clear close-up of the affected plant part (leaf, stem, boll, ear). The app detects crop pests/diseases and gives actionable guidance. If unsure, it will ask you to retake or crop the photo.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- File uploader ----------------
st.markdown('<div class="uploader card">', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag and drop file here", type=["jpg","jpeg","png"])
st.markdown('</div>', unsafe_allow_html=True)

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

        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(pil, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"#### 🔎 **Prediction:** {top_name}")
            st.markdown(f"**Confidence:** {int(top_conf * 100)}%")
            st.markdown(f'<div class="conf-bar"><div class="conf-fill" style="width:{int(top_conf*100)}%"></div></div>', unsafe_allow_html=True)
            st.markdown('---')

            entry = REMEDIES_FULL.get(top_id, {})
            summary = entry.get("summary", "")
            details = entry.get("details", "")

            st.markdown("#### 🌱 **Recommended Action**")
            st.write(summary)
            if details:
                with st.expander("Read detailed guidance"):
                    st.markdown(details.replace("\n", "  \n"))

            st.caption("Remedies are guidance only — consult local extension for chemicals & dosages.")

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
            st.download_button("📄 Download PDF Report",
                               data=pdf_bytes,
                               file_name="plantsight_report.pdf",
                               mime="application/pdf",
                               use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Plant Sight • Fast disease ID • Guidance only — consult local extension for chemicals & dosages</div>', unsafe_allow_html=True)


