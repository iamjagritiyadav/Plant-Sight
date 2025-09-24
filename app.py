# app.py — Plant Sight (full file with YAML remedies, confidence gate, rejected-image logging)
import os
import io
import sys
import time
import base64
import traceback
import subprocess
from datetime import datetime

import streamlit as st
from PIL import Image
import numpy as np
import yaml

# ---------------- Try import ultralytics (best-effort) ----------------
YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception as e:
    # Try one-time pip install (only if absolutely missing) — runtimes prefer build-time deps
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO
    except Exception:
        YOLO = None  # load_model_safe will raise a clear error later

# ---------------- Config ----------------
MODEL_PATH = "best.pt"
TOP_K = 3
CONF_THRESHOLD = 0.70   # require >=70% to show prediction/remedy
REJECTED_DIR = "rejected"

os.makedirs(REJECTED_DIR, exist_ok=True)

# ---------------- Page + CSS ----------------
st.set_page_config(page_title="Plant Sight", page_icon="🌿", layout="wide")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800;900&display=swap');
:root{ --brand-green:#178f2d; --muted:#9aa4ad; }
body { background: #071016; color: #E6EEF3; font-family: 'Poppins', sans-serif; }
.header { display:flex; align-items:center; gap:16px; padding:18px; background: var(--brand-green); border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,0.5); }
.logo-box { width:64px; height:64px; border-radius:12px; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,0.06); }
.title { font-size:30px; font-weight:900; color:white; margin:0; }
.subtitle { color:rgba(255,255,255,0.92); margin-top:2px; font-size:14px; }
.container { margin-top:22px; max-width:920px; margin-left:auto; margin-right:auto; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:24px; box-shadow: 0 8px 30px rgba(2,6,12,0.6); border: 1px solid rgba(255,255,255,0.05); }
.hero-title { font-size:38px; font-weight:900; margin:0; }
.hero-desc { color:#cde6d9; margin-top:10px; font-size:16px; }
.uploader { margin-top:22px; padding:18px; background:#0c1113; border-radius:10px; border:1px solid rgba(255,255,255,0.03); text-align: center; }
.pred-item { background: rgba(255,255,255,0.02); padding:12px; border-radius:10px; }
.conf-bar { height:12px; background: rgba(255,255,255,0.06); border-radius:999px; overflow:hidden; }
.conf-fill { height:100%; background: linear-gradient(90deg,#178f2d,#2de36a); transition: width 0.5s ease-in-out; }
.footer { margin-top:30px; color:var(--muted); font-size:12px; text-align:center; }
.stButton>button { background-color: #178f2d; color: white; font-weight: 600; border-radius: 8px; border: none; }
.small { color: #9aa4ad; font-size:13px; }
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

def save_rejected_upload(uploaded_bytes, reason="low_confidence"):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"rejected/{reason}_{ts}.png"
    try:
        with open(fname, "wb") as f:
            f.write(uploaded_bytes)
    except Exception:
        pass
    return fname

def load_model_safe(path):
    if YOLO is None:
        raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics (or fix requirements on your platform).")
    try:
        return YOLO(path)
    except Exception:
        try:
            return YOLO(path, device="cpu")
        except Exception as e:
            raise e

def extract_topk_from_result(result, k=TOP_K):
    # robust extraction supporting probs or boxes
    try:
        probs_obj = getattr(result, "probs", None)
        if probs_obj is not None:
            # try topk-like behavior
            try:
                arr = np.array(probs_obj).flatten()
                if arr.size:
                    ids = np.argsort(-arr)[:k]
                    return [(int(i), float(arr[i])) for i in ids]
            except Exception:
                pass
            # fallback to top1 fields
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
        if boxes:
            cls_list, conf_list = [], []
            for b in boxes:
                try:
                    cls_list.append(int(b.cls)); conf_list.append(float(b.conf))
                except Exception:
                    pass
            if cls_list:
                pairs = sorted(zip(cls_list, conf_list), key=lambda x:-x[1])[:k]
                return [(int(c), float(s)) for c,s in pairs]
    except Exception:
        pass
    return []

# ---------------- Load labels & remedies from YAML ----------------
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
    for k, v in (data.items() if isinstance(data, dict) else []):
        try:
            kid = int(k)
        except Exception:
            continue
        class_map[kid] = v.get("name", f"Class {kid}")
        remedies_full[kid] = v
    return class_map, remedies_full

# Built-in fallback for class mapping + short remedies (used only if YAML missing)
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

# minimal remedies fallback
BUILTIN_REMEDIES = {k: "General guidance: consult labels and local extension." for k in BUILTIN_CLASS_MAPPING.keys()}

# Load YAML if present
CLASS_MAPPING, REMEDIES_FULL = load_labels_remedies("labels_remedies.yaml")
if CLASS_MAPPING is None:
    CLASS_MAPPING = BUILTIN_CLASS_MAPPING
if REMEDIES_FULL is None:
    REMEDIES_FULL = {k: {"name": CLASS_MAPPING.get(k, f"Class {k}"), "summary": BUILTIN_REMEDIES.get(k, ""), "details": BUILTIN_REMEDIES.get(k, "")} for k in CLASS_MAPPING.keys()}

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

st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Plant Sight — Protect your crop, protect your livelihood</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-desc">Upload a clear close-up photo of the affected part (leaf, stem, boll, ear). The app will only identify crop pests/diseases.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Uploader ----------------
st.markdown('<div class="uploader card">', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag and drop file here", type=["jpg","jpeg","png"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Model load ----------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place your trained model next to app.py and rename to {MODEL_PATH}")
    st.stop()

try:
    MODEL = load_model_safe(MODEL_PATH)
except Exception as e:
    st.error("Failed to load model. Check ultralytics installation and model file. Error: " + str(e))
    st.write(traceback.format_exc())
    st.stop()

# ---------------- When uploaded -> predict & show ----------------
if uploaded:
    # read bytes for possible saving
    try:
        uploaded_bytes = uploaded.getvalue()
    except Exception:
        uploaded_bytes = None

    try:
        pil = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    except Exception:
        st.error("Unable to read the image. Try another file.")
        st.stop()

    with st.spinner("Analyzing image..."):
        try:
            results = MODEL.predict(pil, save=False, verbose=False)
            r = results[0]
            preds = extract_topk_from_result(r, k=TOP_K)  # list of (class_id, conf)
        except Exception as e:
            st.error("Prediction error: " + str(e))
            st.write(traceback.format_exc())
            st.stop()

    # build display_preds: convert ids to names and normalize conf
    display_preds = []
    for cid, conf in preds:
        try:
            cfloat = float(conf)
        except:
            cfloat = 0.0
        # If conf looks >1 assume percent
        if cfloat > 1.0:
            cfloat = cfloat / 100.0
        name = CLASS_MAPPING.get(cid, f"Class {cid}")
        display_preds.append((name, cfloat, int(cid)))

    # fallback: if no preds but r.probs exists try to use top1
    if not display_preds:
        try:
            if hasattr(r, "probs") and hasattr(r.probs, "top1conf") and hasattr(r.probs, "top1"):
                rawc = float(r.probs.top1conf)
                if rawc > 1.0: rawc = rawc / 100.0
                tid = int(r.probs.top1)
                display_preds = [(CLASS_MAPPING.get(tid, f"Class {tid}"), rawc, tid)]
        except Exception:
            pass

    # ---------- Visual layout with crop-only guard ----------
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(pil, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        top_name, top_conf, top_id = "Unknown", 0.0, None
        if display_preds:
            top_name, top_conf, top_id = display_preds[0]

        # Gate: must be known crop class and confidence >= threshold
        valid_class = (top_id in CLASS_MAPPING)
        if top_conf >= CONF_THRESHOLD and valid_class:
            st.markdown(f"#### 🔎 **Prediction:** {top_name}")
            st.markdown(f"**Confidence:** {int(top_conf*100)}%")
            st.markdown(f'<div class="conf-bar"><div class="conf-fill" style="width:{int(top_conf*100)}%"></div></div>', unsafe_allow_html=True)
            st.markdown('---')

            # Remedies from YAML (REMEDIES_FULL) prefered
            entry = REMEDIES_FULL.get(top_id)
            if entry:
                remedy_short = entry.get("summary", "")
                remedy_detailed = entry.get("details", "")
                st.markdown("#### 🌱 **Recommended Action**")
                st.write(remedy_short)
                with st.expander("Read detailed guidance"):
                    st.markdown(remedy_detailed.replace("\n", "  \n"))
            else:
                st.markdown("#### 🌱 **Recommended Action**")
                st.write(BUILTIN_REMEDIES.get(top_id, "No remedy available."))
            st.caption("Remedies are guidance — consult local extension for chemicals & dosages.")
            # allow download summary
            rep = f"Plant Sight result\nTop prediction: {top_name} ({top_conf:.2f})\nRemedy: {(entry.get('summary') if entry else BUILTIN_REMEDIES.get(top_id,''))}\n"
            st.download_button("📥 Download summary (.txt)", rep, file_name="plantsight_result.txt", use_container_width=True)
        else:
            # reject: save rejected image for analysis
            reason = "not_crop_or_low_conf"
            saved = None
            if uploaded_bytes:
                saved = save_rejected_upload(uploaded_bytes, reason=reason)
            st.markdown("#### ⚠️ **No valid crop prediction**")
            st.error("This app is only designed for crop disease detection. Please upload a clear crop/leaf/stem image.")
            st.markdown('<div class="small">Tips: crop the disease patch, avoid humans/animals/objects, use good lighting and close-up photos.</div>', unsafe_allow_html=True)
            if saved:
                st.markdown(f'<div class="small">Image saved for review: <code>{saved}</code></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # always allow image download
    st.download_button("🖼️ Download image", image_bytes(pil), file_name="input.png", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Plant Sight • Fast disease ID • Guidance only — consult local extension for chemicals & dosages</div>', unsafe_allow_html=True)
