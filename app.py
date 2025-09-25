"""app.py — Plant Sight (rewritten)

Streamlit app for crop disease detection using an Ultralytics YOLO model.
This version is cleaned, typed, and production-friendly.

Requirements (install at build time / in your environment):
- streamlit
- ultralytics
- torch (if needed by your ultralytics build)
- pillow
- numpy
- pyyaml

Notes:
- Avoids runtime pip installs; install dependencies at build time.
- Use Streamlit Cloud's packages.txt / runtime.txt for system libs (libGL) and Python version.

Place your trained model file (default: `best.pt`) and optional
`labels_remedies.yaml` next to this file.
"""

from __future__ import annotations

import io
import os
import base64
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import numpy as np
import yaml
from PIL import Image

# ---------------------- Configuration ----------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
LABELS_REMEDIES_YAML = os.environ.get("LABELS_REMEDIES_YAML", "labels_remedies.yaml")
TOP_K = 3
CONF_THRESHOLD = 0.70  # require >= 0.70 confidence to accept prediction
REJECTED_DIR = "rejected"
CROP_KEYWORDS = ["cotton", "maize", "wheat", "rice", "sugarcane"]
ALLOWED_EXTS = ("jpg", "jpeg", "png")

os.makedirs(REJECTED_DIR, exist_ok=True)

# ---------------------- UI / Styling ----------------------
st.set_page_config(page_title="Plant Sight", page_icon="🌿", layout="wide")

CSS = r"""
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
@media (max-width: 768px) { .header { flex-direction: column; text-align:center; } .hero-title { font-size:24px; } }
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------- Helpers ----------------------


def find_logo_file() -> Optional[str]:
    """Return first file in working dir whose name starts with 'logo' (case-insensitive)."""
    for fname in os.listdir("."):
        if fname.lower().startswith("logo") and os.path.isfile(fname):
            return fname
    return None


def image_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def save_rejected_upload(uploaded_bytes: bytes, reason: str = "rejected") -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = os.path.join(REJECTED_DIR, f"{reason}_{ts}.png")
    try:
        with open(fname, "wb") as f:
            f.write(uploaded_bytes)
    except Exception:
        # best-effort save — ignore failures
        pass
    return fname


def normalize_conf(c: Any) -> float:
    """Normalize confidence to [0.0, 1.0]. Accepts 0..1 or 0..100."""
    try:
        f = float(c)
    except Exception:
        return 0.0
    if f > 1.0:
        f = f / 100.0
    if f < 0:
        return 0.0
    return min(1.0, f)


def is_crop_name(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.lower()
    return any(kw in n for kw in CROP_KEYWORDS)


# ---------------------- Model handling ----------------------

YOLO = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception as e:  # pragma: no cover - handled at runtime
    YOLO = None


@st.cache_resource
def load_model_safe(path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics not available — install at build time")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    # Allow ultralytics decide device; if it fails try cpu explicitly
    try:
        return YOLO(path)
    except Exception:
        return YOLO(path, device="cpu")


def extract_topk_from_result(result: Any, k: int = TOP_K) -> List[Tuple[int, float]]:
    """Extract top-k (class_id, confidence) tuples from an ultralytics result object.
    This function is defensive against different ultralytics return shapes.
    """
    try:
        probs_obj = getattr(result, "probs", None)
        if probs_obj is not None:
            # try convert to numpy if possible
            try:
                arr = np.array(probs_obj).flatten()
                if arr.size:
                    idxs = np.argsort(-arr)[:k]
                    return [(int(i), float(arr[i])) for i in idxs]
            except Exception:
                # fallback to attributes
                if hasattr(probs_obj, "top1") and hasattr(probs_obj, "top1conf"):
                    return [(int(probs_obj.top1), float(probs_obj.top1conf))]
    except Exception:
        pass

    # fallback to boxes (common for detection models)
    try:
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            pairs: List[Tuple[int, float]] = []
            for b in boxes:
                try:
                    pairs.append((int(b.cls), float(b.conf)))
                except Exception:
                    continue
            pairs = sorted(pairs, key=lambda x: -x[1])[:k]
            return pairs
    except Exception:
        pass

    return []


# ---------------------- Labels & Remedies ----------------------

BUILTIN_CLASS_MAPPING: Dict[int, str] = {
    0: "American Bollworm on Cotton",
    1: "Anthracnose on Cotton",
    2: "Army worm",
    10: "Healthy Maize",
    11: "Healthy Wheat",
}

BUILTIN_REMEDIES_SHORT = {k: "General guidance — consult local extension for crop-specific chemical/dosage." for k in BUILTIN_CLASS_MAPPING}


def load_labels_remedies(path: str = LABELS_REMEDIES_YAML) -> Tuple[Dict[int, str], Dict[int, Dict[str, Any]]]:
    if not os.path.exists(path):
        return BUILTIN_CLASS_MAPPING, {k: {"name": BUILTIN_CLASS_MAPPING[k], "summary": BUILTIN_REMEDIES_SHORT.get(k, ""), "details": ""} for k in BUILTIN_CLASS_MAPPING}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return BUILTIN_CLASS_MAPPING, {k: {"name": BUILTIN_CLASS_MAPPING[k], "summary": BUILTIN_REMEDIES_SHORT.get(k, ""), "details": ""} for k in BUILTIN_CLASS_MAPPING}

    class_map: Dict[int, str] = {}
    remedies_full: Dict[int, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                kid = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                class_map[kid] = v.get("name", f"Class {kid}")
                remedies_full[kid] = {"name": class_map[kid], "summary": v.get("summary", ""), "details": v.get("details", "")}
    if not class_map:
        class_map = BUILTIN_CLASS_MAPPING
        remedies_full = {k: {"name": BUILTIN_CLASS_MAPPING[k], "summary": BUILTIN_REMEDIES_SHORT.get(k, ""), "details": ""} for k in BUILTIN_CLASS_MAPPING}
    return class_map, remedies_full


CLASS_MAPPING, REMEDIES_FULL = load_labels_remedies()

# ---------------------- Header UI ----------------------

logo_file = find_logo_file()
if logo_file:
    with open(logo_file, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:64px;height:64px;border-radius:10px;object-fit:cover;">'
else:
    logo_html = '<div class="logo-box"><span style="font-weight:800;color:#fff">PS</span></div>'

st.markdown(
    f"""
<div class="header">
  {logo_html}
  <div>
    <div class="title">Plant Sight</div>
    <div class="subtitle">Fast disease ID • Clear remedial steps • Farmer-friendly</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------- Main container ----------------------

st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Plant Sight — Protect your crop, protect your livelihood</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-desc">Upload a clear close-up of the affected plant part (leaf, stem, boll, ear). The app detects crop pests/diseases and gives actionable guidance.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="uploader card">', unsafe_allow_html=True)
uploaded = st.file_uploader("Drag and drop file here", type=list(ALLOWED_EXTS))
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Model availability check ----------------------

if YOLO is None:
    st.error("ultralytics import failed. Make sure 'ultralytics' is installed in your environment at build time.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place your trained model next to app.py.")
    st.stop()

# load model (cached)
try:
    MODEL = load_model_safe(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.write(traceback.format_exc())
    st.stop()

# ---------------------- Handle upload -> predict ----------------------

if uploaded is not None:
    try:
        uploaded_bytes = uploaded.getvalue()
    except Exception:
        st.error("Unable to read uploaded file.")
        st.stop()

    try:
        pil = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    except Exception:
        st.error("Unable to decode image. Try a different file.")
        st.stop()

    with st.spinner("Analyzing image..."):
        try:
            # ultralytics accepts PIL.Image
            results = MODEL.predict(pil, save=False, verbose=False)
            r = results[0]
            preds_raw = extract_topk_from_result(r, k=TOP_K)
        except Exception as e:
            st.error("Prediction error: " + str(e))
            st.write(traceback.format_exc())
            st.stop()

    # convert into display-friendly tuples: (name, conf_norm, id)
    display_preds: List[Tuple[str, float, int]] = []
    for cid, conf in preds_raw:
        nconf = normalize_conf(conf)
        name = CLASS_MAPPING.get(int(cid), f"Class {cid}")
        display_preds.append((name, nconf, int(cid)))

    # fallback attempt
    if not display_preds:
        try:
            if hasattr(r, "probs") and hasattr(r.probs, "top1") and hasattr(r.probs, "top1conf"):
                tid = int(r.probs.top1)
                rc = normalize_conf(float(r.probs.top1conf))
                display_preds = [(CLASS_MAPPING.get(tid, f"Class {tid}"), rc, tid)]
        except Exception:
            pass

    # Layout
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

        name_crop_flag = is_crop_name(top_name)
        valid_class = (top_id in CLASS_MAPPING)
        confident = (top_conf >= CONF_THRESHOLD)

        if confident and valid_class and name_crop_flag:
            st.markdown(f"#### 🔎 **Prediction:** {top_name}")
            st.markdown(f"**Confidence:** {int(top_conf * 100)}%")
            st.markdown(f'<div class="conf-bar"><div class="conf-fill" style="width:{int(top_conf*100)}%"></div></div>', unsafe_allow_html=True)
            st.markdown('---')

            entry = REMEDIES_FULL.get(top_id)
            remedy_summary = entry.get("summary") if entry else BUILTIN_REMEDIES_SHORT.get(top_id, "General guidance — consult local extension.")

            st.markdown("#### 🌱 **Recommended Action**")
            st.write(remedy_summary)
            if entry and entry.get("details"):
                with st.expander("Read detailed guidance"):
                    st.markdown(entry.get("details").replace("\n", "  \n"))

            st.caption("Remedies are guidance only — consult local extension for chemicals & dosages.")

            rep = (
                f"Plant Sight result\nTop prediction: {top_name} ({top_conf:.2f})\nRemedy: {remedy_summary}\n"
            )
            st.download_button("📥 Download summary (.txt)", rep, file_name="plantsight_result.txt", use_container_width=True)
        else:
            reason = "not_crop_or_low_conf"
            saved = save_rejected_upload(uploaded_bytes, reason=reason)

            st.markdown("#### ⚠️ **No valid crop prediction**")
            if not confident:
                st.error("Model confidence is low. Try a clearer close-up, better lighting, or crop the diseased area.")
            elif not name_crop_flag:
                st.error("This image does not look like a crop part. Please upload a leaf, stem, boll, or ear photo.")
            else:
                st.error("No valid prediction — try another image.")

            st.markdown('<div class="small">Tips: crop the disease patch, avoid humans/animals/food/objects, use natural light and fill the frame with the affected area.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small">Image saved for review: <code>{saved}</code></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Always allow image download
    st.download_button("🖼️ Download image", image_bytes(pil), file_name="input.png", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Plant Sight • Fast disease ID • Guidance only — consult local extension for chemicals & dosages</div>', unsafe_allow_html=True)

# End of file
