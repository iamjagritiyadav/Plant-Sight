import os
import io
import sys
import base64
import traceback
from datetime import datetime

import streamlit as st
from PIL import Image
import numpy as np
import yaml

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

def image_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def save_rejected_upload(uploaded_bytes: bytes, reason: str = "rejected"):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = os.path.join(REJECTED_DIR, f"{reason}_{ts}.png")
    try:
        with open(fname, "wb") as f:
            f.write(uploaded_bytes)
    except Exception:
        pass
    return fname

def normalize_conf(c):
    """Return confidence as float in 0..1 space. Accepts 0..1 or 0..100."""
    try:
        f = float(c)
    except Exception:
        return 0.0
    if f > 1.0:
        f = f / 100.0
    if f < 0: f = 0.0
    if f > 1.0: f = 1.0
    return f

def is_crop_name(name: str):
    if not name: return False
    n = name.lower()
    return any(kw in n for kw in CROP_KEYWORDS)

# ---------------- Model import (do NOT pip install here) ----------------
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception as e:
    # Provide useful message in-streamlit UI rather than trying to install
    st.error("ultralytics import failed. Make sure your environment has 'ultralytics' installed (build-time).")
    st.write("Detailed error:", str(e))
    st.stop()

def load_model_safe(path):
    if YOLO is None:
        raise RuntimeError("ultralytics not available")
    try:
        return YOLO(path)
    except Exception:
        try:
            return YOLO(path, device="cpu")
        except Exception as e:
            raise

def extract_topk_from_result(result, k=TOP_K):
    """Robustly extract top-k (class_id, confidence) tuples from ultralytics result."""
    try:
        probs_obj = getattr(result, "probs", None)
        if probs_obj is not None:
            # try numpy array conversion
            try:
                arr = np.array(probs_obj).flatten()
                if arr.size:
                    idxs = np.argsort(-arr)[:k]
                    return [(int(i), float(arr[i])) for i in idxs]
            except Exception:
                pass
            # fallback to top1/top1conf
            try:
                if hasattr(probs_obj, "top1") and hasattr(probs_obj, "top1conf"):
                    return [(int(probs_obj.top1), float(probs_obj.top1conf))]
            except Exception:
                pass
    except Exception:
        pass
    # fallback to boxes
    try:
        boxes = getattr(result, "boxes", None)
        if boxes and len(boxes) > 0:
            pairs = []
            for b in boxes:
                try:
                    pairs.append((int(b.cls), float(b.conf)))
                except Exception:
                    pass
            pairs = sorted(pairs, key=lambda x: -x[1])[:k]
            return pairs
    except Exception:
        pass
    return []

# ---------------- Load labels/remedies YAML ----------------
def load_labels_remedies(path="labels_remedies.yaml"):
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None, None
    class_map = {}
    remedies_full = {}
    if not isinstance(data, dict):
        return None, None
    for k, v in data.items():
        try:
            kid = int(k)
        except Exception:
            continue
        class_map[kid] = v.get("name", f"Class {kid}")
        remedies_full[kid] = v
    return class_map, remedies_full

# Fallback built-in mapping & minimal remedy
BUILTIN_CLASS_MAPPING = {
  0: "American Bollworm on Cotton",1: "Anthracnose on Cotton",2: "Army worm",3: "Bacterial Blight in cotton",
  4: "Becterial Blight in Rice",5: "Brownspot",6: "Common_Rust",7: "Cotton Aphid",8: "Flag Smut",
  9: "Gray_Leaf_Spot",10: "Healthy Maize",11: "Healthy Wheat",12: "Healthy cotton",13: "Leaf Curl",
  14: "Leaf smut",15: "Mosaic sugarcane",16: "RedRot sugarcane",17: "RedRust sugarcane",18: "Rice Blast",
  19: "Sugarcane Healthy",20: "Tungro",21: "Wheat Brown leaf Rust",22: "Wheat Brown leaf rust",23: "Wheat Stem fly",
  24: "Wheat aphid",25: "Wheat black rust",26: "Wheat leaf blight",27: "Wheat mite",28: "Wheat powdery mildew",
  29: "Wheat scab",30: "Wheat___Yellow_Rust",31: "Wilt",32: "Yellow Rust Sugarcane",33: "bacterial_blight in Cotton",
  34: "bollrot on Cotton",35: "bollworm on Cotton",36: "cotton mealy bug",37: "cotton whitefly",38: "maize ear rot",
  39: "maize fall armyworm",40: "maize stem borer",41: "pink bollworm in cotton",42: "red cotton bug",43: "thirps on  cotton"
}
BUILTIN_REMEDIES_SHORT = {k: "General guidance — consult local extension for crop-specific chemical/dosage." for k in BUILTIN_CLASS_MAPPING.keys()}

# Load YAML if present
CLASS_MAPPING, REMEDIES_FULL = load_labels_remedies("labels_remedies.yaml")
if CLASS_MAPPING is None:
    CLASS_MAPPING = BUILTIN_CLASS_MAPPING
if REMEDIES_FULL is None:
    REMEDIES_FULL = {k: {"name": CLASS_MAPPING.get(k, f"Class {k}"), "summary": BUILTIN_REMEDIES_SHORT.get(k, ""), "details": BUILTIN_REMEDIES_SHORT.get(k, "")} for k in CLASS_MAPPING.keys()}

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

# Uploader
st.markdown('<div class="uploader card">', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag and drop file here", type=["jpg","jpeg","png"])
st.markdown('</div>', unsafe_allow_html=True)

# Model check
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place your trained model next to app.py and name it {MODEL_PATH}")
    st.stop()

try:
    MODEL = load_model_safe(MODEL_PATH)
except Exception as e:
    st.error("Failed to load model. Check ultralytics installation and model file. Error: " + str(e))
    st.write(traceback.format_exc())
    st.stop()

# ---------------- When uploaded -> predict & show ----------------
if uploaded:
    try:
        uploaded_bytes = uploaded.getvalue()
    except Exception:
        uploaded_bytes = None

    try:
        pil = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    except Exception:
        st.error("Unable to read uploaded image. Try another file.")
        st.stop()

    with st.spinner("Analyzing image..."):
        try:
            results = MODEL.predict(pil, save=False, verbose=False)
            r = results[0]
            preds_raw = extract_topk_from_result(r, k=TOP_K)   # list of (id, conf)
        except Exception as e:
            st.error("Prediction error: " + str(e))
            st.write(traceback.format_exc())
            st.stop()

    # Build display_preds: (name, conf_normalized, id)
    display_preds = []
    for cid, conf in preds_raw:
        nconf = normalize_conf(conf)
        name = CLASS_MAPPING.get(int(cid), f"Class {cid}")
        display_preds.append((name, nconf, int(cid)))

    # fallback: try r.probs top1 if nothing found
    if not display_preds:
        try:
            if hasattr(r, "probs") and hasattr(r.probs, "top1") and hasattr(r.probs, "top1conf"):
                tid = int(r.probs.top1)
                rc = float(r.probs.top1conf)
                rc = normalize_conf(rc)
                display_preds = [(CLASS_MAPPING.get(tid, f"Class {tid}"), rc, tid)]
        except Exception:
            pass

    # Layout: image left, prediction right
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(pil, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        top_name, top_conf, top_id = ("Unknown", 0.0, None)
        if display_preds:
            top_name, top_conf, top_id = display_preds[0]

        # Crop-only guard
        name_crop_flag = is_crop_name(top_name)
        valid_class = (top_id in CLASS_MAPPING)
        confident = (top_conf >= CONF_THRESHOLD)

        if confident and valid_class and name_crop_flag:
            # accepted prediction
            st.markdown(f"#### 🔎 **Prediction:** {top_name}")
            st.markdown(f"**Confidence:** {int(top_conf * 100)}%")
            st.markdown(f'<div class="conf-bar"><div class="conf-fill" style="width:{int(top_conf*100)}%"></div></div>', unsafe_allow_html=True)
            st.markdown('---')

            entry = REMEDIES_FULL.get(top_id, None)
            if entry:
                summary = entry.get("summary", "")
                details = entry.get("details", "")
                st.markdown("#### 🌱 **Recommended Action**")
                st.write(summary)
                with st.expander("Read detailed guidance"):
                    st.markdown(details.replace("\n", "  \n"))
            else:
                st.markdown("#### 🌱 **Recommended Action**")
                st.write(BUILTIN_REMEDIES_SHORT.get(top_id, "General guidance — consult local extension."))

            st.caption("Remedies are guidance only — consult local extension for chemicals & dosages.")
            # Downloadable summary
            rep = f"Plant Sight result\nTop prediction: {top_name} ({top_conf:.2f})\nRemedy: {(entry.get('summary') if entry else BUILTIN_REMEDIES_SHORT.get(top_id,''))}\n"
            st.download_button("📥 Download summary (.pdf)", rep, file_name="plantsight_result.pdf", use_container_width=True)
        else:
            # reject - not confident or not crop-like
            reason = "not_crop_or_low_conf"
            saved = None
            if uploaded_bytes:
                saved = save_rejected_upload(uploaded_bytes, reason=reason)
            st.markdown("#### ⚠️ **No valid crop prediction**")
            if not confident:
                st.error("Model confidence is low. Try a clearer close-up, better lighting, or crop the diseased area.")
            elif not name_crop_flag:
                st.error("This image does not look like a crop part. Please upload a leaf, stem, boll, or ear photo.")
            else:
                st.error("No valid prediction — try another image.")
            st.markdown('<div class="small">Tips: crop the disease patch, avoid humans/animals/food/objects, use natural light and fill the frame with the affected area.</div>', unsafe_allow_html=True)
            if saved:
                st.markdown(f'<div class="small">Image saved for review: <code>{saved}</code></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Always allow image download
    st.download_button("🖼️ Download image", image_bytes(pil), file_name="input.png", use_container_width=True)

# End container/footer
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Plant Sight • Fast disease ID • Guidance only — consult local extension for chemicals & dosages</div>', unsafe_allow_html=True)
