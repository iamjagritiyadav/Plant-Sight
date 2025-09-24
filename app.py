import sys, traceback
import streamlit as st
st.write("Python:", sys.version)
try:
    import ultralytics
    st.write("ultralytics:", ultralytics.__version__)
except Exception as e:
    st.write("ultralytics import error:", repr(e))
    st.write(traceback.format_exc())
# app.py - Plant Sight (Polished & Responsive UI)
import os, io, base64, traceback, subprocess, sys
import streamlit as st
from PIL import Image
import numpy as np

# ---------------- Ensure ultralytics available (runtime fallback) ----------------
YOLO = None
try:
    # try normal import first
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception:
    # try to install and import
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO
    except Exception:
        # leave YOLO as None; load_model_safe will raise a clear error
        YOLO = None

# ---------------- Config ----------------
MODEL_PATH = "best.pt"  # ensure this file exists next to app.py
TOP_K = 3

# ---------------- Page + CSS ----------------
st.set_page_config(page_title="Plant Sight", page_icon="🌿", layout="wide")
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800;900&display=swap');
:root{ --brand-green:#178f2d; --muted:#9aa4ad; --card:#0f1720; }
body { 
    background: #071016; 
    color: #E6EEF3; 
    font-family: 'Poppins', sans-serif; 
    animation: fadeIn 0.8s ease-in-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.header { 
    display:flex; 
    align-items:center; 
    gap:16px; 
    padding:18px; 
    background: var(--brand-green); 
    border-radius:12px; 
    box-shadow:0 8px 24px rgba(0,0,0,0.5); 
}
.logo-box { 
    width:64px; 
    height:64px; 
    border-radius:12px; 
    display:flex; 
    align-items:center; 
    justify-content:center; 
    background:rgba(255,255,255,0.06); 
}
.title { 
    font-size:30px; 
    font-weight:900; 
    color:white; 
    margin:0; 
}
.subtitle { 
    color:rgba(255,255,255,0.92); 
    margin-top:2px; 
    font-size:14px; 
}
.container { 
    margin-top:22px; 
    max-width:920px; 
    margin-left: auto;
    margin-right: auto;
}
.st-emotion-cache-18ni7ap {
    background-color: transparent !important;
}
.card { 
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); 
    border-radius:12px; 
    padding:24px; 
    box-shadow: 0 8px 30px rgba(2,6,12,0.6); 
    border: 1px solid rgba(255,255,255,0.05);
}
.hero { 
    padding:32px; 
    border-radius:12px; 
    margin-bottom:14px; 
    text-align:left; 
}
.hero-title { 
    font-size:38px; 
    font-weight:900; 
    margin:0; 
}
.hero-stat { 
    color:#dff7ea; 
    font-weight:700; 
    margin-top:8px; 
}
.hero-desc { 
    color:#cde6d9; 
    margin-top:10px; 
    font-size:16px; 
}
.uploader { 
    margin-top:22px; 
    padding:18px; 
    background:#0c1113; 
    border-radius:10px; 
    border:1px solid rgba(255,255,255,0.03); 
    text-align: center;
}
.pred-list { 
    display:flex; 
    flex-direction: column; 
    gap:12px; 
    margin-top:12px; 
}
.pred-item { 
    background: rgba(255,255,255,0.02); 
    padding:12px; 
    border-radius:10px; 
    flex:1; 
    min-width:0; 
}
.pred-name { 
    font-weight:700; 
    font-size:16px; 
    margin-bottom:6px; 
    overflow:hidden; 
    text-overflow:ellipsis; 
    white-space:nowrap; 
}
.conf-bar { 
    height:12px; 
    background: rgba(255,255,255,0.06); 
    border-radius:999px; 
    overflow:hidden; 
    animation: fillBar 1s ease-out;
}
.conf-fill { 
    height:100%; 
    background: linear-gradient(90deg,#178f2d,#2de36a); 
    transition: width 0.5s ease-in-out;
}
@keyframes fillBar {
  from { width: 0; }
}
.footer { 
    margin-top:30px; 
    color:var(--muted); 
    font-size:12px; 
    text-align:center; 
}
.small { color:var(--muted); font-size:13px; }
.stProgress > div > div {
    background-color: var(--brand-green) !important;
}
.stProgress > div {
    background-color: rgba(255,255,255,0.06) !important;
    border-radius: 999px;
}
.stButton>button {
    background-color: #178f2d;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    border: none;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #2de36a;
    color: #071016;
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}

/* Responsive CSS */
@media (max-width: 768px) {
    .header {
        flex-direction: column;
        text-align: center;
    }
    .container {
        padding: 0 10px;
    }
    .hero-title {
        font-size: 28px;
    }
    .hero-desc {
        font-size: 14px;
    }
    .st-emotion-cache-18ni7ap {
        width: 100% !important;
        padding: 0 !important;
    }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Helpers ----------------
def find_logo_file():
    # look for a file starting with 'logo' in cwd
    for fname in os.listdir("."):
        if fname.lower().startswith("logo") and os.path.isfile(fname):
            return fname
    return None

def load_model_safe(path):
    if YOLO is None:
        raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")
    # try normal load, fallback to CPU
    try:
        return YOLO(path)
    except Exception:
        try:
            return YOLO(path, device="cpu")
        except Exception as e:
            raise e

def extract_topk_from_result(result, k=TOP_K):
    # robustly extract top-k from result.probs or result.boxes
    probs_obj = getattr(result, "probs", None)
    if probs_obj is not None:
        try:
            arr = np.array(probs_obj).flatten()
            if arr.size:
                ids = np.argsort(-arr)[:k]
                return [(int(i), float(arr[i])) for i in ids]
        except Exception:
            pass
        try:
            if hasattr(probs_obj, "top1") and hasattr(probs_obj, "top1conf"):
                return [(int(probs_obj.top1), float(probs_obj.top1conf))]
        except Exception:
            pass
        try:
            if hasattr(probs_obj, "cpu"):
                arr = np.array(probs_obj.cpu()).flatten()
                ids = np.argsort(-arr)[:k]
                return [(int(i), float(arr[i])) for i in ids]
        except Exception:
            pass
    # fallback: boxes cls/conf
    try:
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
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

def image_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# ---------------- Class mapping & remedies ----------------
CLASS_MAPPING = {
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
REMEDIES = {
  0: "Monitor bolls, remove damaged bolls, use pheromone traps/targeted insecticide and Bt varieties; rotate crops.",
  1: "Remove infected debris, increase air circulation, apply appropriate fungicide and use resistant varieties.",
  2: "Handpick or use biocontrols/targeted insecticides; maintain healthy crop borders and timely scouting.",
  3: "Remove infected plant parts, avoid overhead irrigation, improve drainage; use approved bactericides.",
  4: "Use clean seed, balanced fertilizer, flood/dry management and registered bactericides; follow local extension advice.",
  5: "Use resistant varieties, balanced nutrients and foliar fungicides if needed; avoid prolonged leaf wetness.",
  6: "Remove alternate hosts, use rust-resistant varieties and fungicide sprays timed to infection risk.",
  7: "Use neem/soap sprays or biological controls; conserve predators (ladybugs); avoid excess nitrogen.",
  8: "Use certified seed, seed treatment and crop rotation; follow local fungicide recommendations.",
  9: "Remove infected leaves, improve airflow, apply fungicide when needed and avoid overhead watering.",
  10: "No disease detected — maintain good agronomic practices and regular scouting.",
  11: "Healthy — keep crop rotation, timely fungicide if risk appears, and good nutrition.",
  12: "Healthy — continue best-practices: balanced irrigation, nutrition and monitoring.",
  13: "Prune affected parts, use resistant varieties, remove heavily infested plants and use vector control.",
  14: "Use resistant varieties, seed treatment and fungicide where recommended; improve field hygiene.",
  15: "Remove infected stools, use resistant varieties, maintain balanced nutrition and follow extension guidance.",
  16: "Sanitation, remove infected canes, use resistant varieties and follow recommended fungicide schedule.",
  17: "Use resistant varieties, cultural sanitation, and fungicides as per local advice.",
  18: "Use resistant varieties, proper spacing, timely fungicide and water management; remove infected plants.",
  19: "Healthy — maintain good field hygiene, nutrition and pest monitoring.",
  20: "Viral disease — remove infected plants, control vectors (insects), use resistant varieties and healthy seed.",
  21: "Remove infected debris, apply rust control measures and use resistant varieties.",
  22: "See entry 21 — sanitation, resistant varieties and fungicide where applicable.",
  23: "Stem fly — early sowing, remove residues, use tolerant varieties and insecticide seed treatment if recommended.",
  24: "Wheat aphid — monitor, release/encourage natural enemies, use targeted insecticides only if threshold exceeded.",
  25: "Black rust — plant resistant varieties, monitor and apply fungicide at key growth stages.",
  26: "Leaf blight — crop rotation, avoid dense stands, treat with fungicide when needed.",
  27: "Mites — use miticides or oils, conserve predators and avoid excessive use of broad-spectrum insecticides.",
  28: "Powdery mildew — improve airflow, timely fungicides and grow resistant varieties.",
  29: "Scab (Fusarium) — crop rotation, resistant varieties, and fungicide seed treatment where advised.",
  30: "Yellow rust — plant resistant varieties, monitor and apply fungicide at early signs.",
  31: "Wilt — diagnose (fungal/bacterial/physiological), use resistant varieties and improve drainage; remove infected plants.",
  32: "Yellow Rust Sugarcane — use resistant varieties, fungicide sprays and sanitation.",
  33: "Bacterial blight (cotton) — remove infected material, control vectors, and use clean seed/approved treatments.",
  34: "Boll rot — improve air flow, avoid late-season wetness, timely insect control to prevent fruit damage.",
  35: "Bollworm — monitor, use pheromone traps/Bt or targeted insecticides and timely scouting.",
  36: "Mealy bug — release natural enemies, use insecticidal soaps/oils and remove heavily infested plants.",
  37: "Whitefly — use yellow sticky traps, natural enemies, and insecticidal soaps or targeted sprays if needed.",
  38: "Maize ear rot — manage moisture at harvest, use resistant hybrids and proper storage.",
  39: "Fall armyworm — early detection, biopesticides (Bt), targeted insecticides and field sanitation.",
  40: "Stem borer — use pheromone traps, resistant varieties and stem borer management practices.",
  41: "Pink bollworm — pheromone traps, cultural controls and timely insecticide targeting larvae.",
  42: "Red cotton bug — hand-pick high populations, use insecticide when threshold crossed and field sanitation.",
  43: "Thrips — monitor, use reflective mulch, conserve predators and apply insecticidal soap/targeted insecticide if needed."
}

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

# ---------------- Main container ----------------
st.markdown('<div class="container">', unsafe_allow_html=True)

# ---------------- Hero ----------------
st.markdown('<div class="hero card">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Plant Sight — Protect your crop, protect your livelihood</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-stat">Millions of smallholder farmers worldwide lose significant yield each year because of pests & diseases.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-desc">Upload a clear close-up photo of the affected part (leaf, stem, boll, ear). Plant Sight will identify likely pests/diseases and give concise action steps you can follow.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Uploader ----------------
st.markdown('<div class="uploader card uploader">', unsafe_allow_html=True)
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
    traceback.print_exc()
    st.stop()

# ---------------- When uploaded -> predict & show ----------------
if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("Unable to read image. Try another file.")
        st.stop()

    with st.spinner("Analyzing image..."):
        try:
            results = MODEL.predict(img, save=False, verbose=False)
            r = results[0]
            preds = extract_topk_from_result(r, k=TOP_K)
        except Exception as e:
            st.error("Prediction error: " + str(e)); traceback.print_exc(); st.stop()

    # prepare display preds
    display_preds = []
    for idx, conf in preds:
        display_preds.append((CLASS_MAPPING.get(idx, f"Class {idx}"), conf))

    if not display_preds and hasattr(r, "names"):
        try:
            best_idx = int(r.probs.top1)
            display_preds = [(r.names.get(best_idx, f"Class {best_idx}"), float(r.probs.top1conf))]
        except Exception:
            pass

    # visual layout
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if display_preds:
            top_name, top_conf = display_preds[0]
            st.markdown(f"#### 🔎 **Prediction:** {top_name}")
            st.markdown(f"**Confidence:** {int(top_conf*100)}%")
            st.markdown(f'<div class="conf-bar"><div class="conf-fill" style="width:{int(top_conf*100)}%"></div></div>', unsafe_allow_html=True)
        else:
            top_name, top_conf = ("Unknown", 0.0)
            st.markdown("**Prediction:** Unknown")

        st.markdown('---')
        remedy_text = REMEDIES.get(preds[0][0]) if preds else "No remedy available."
        st.markdown("#### 🌱 **Recommended Action**")
        st.write(remedy_text)
        st.caption("Remedies are guidance — consult local extension for chemicals & dosages.")
        st.markdown('</div>', unsafe_allow_html=True)

    # downloads
    st.markdown('<div class="stButton-row">', unsafe_allow_html=True)
    rep = f"Plant Sight result\nTop prediction: {top_name} ({top_conf:.2f})\nRemedy: {remedy_text}\n"
    st.download_button("📥 Download summary (.txt)", rep, file_name="plantsight_result.txt", use_container_width=True)
    st.download_button("🖼️ Download image", image_bytes(img), file_name="input.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Plant Sight • Fast disease ID • Guidance only — consult local extension for chemicals & dosages</div>', unsafe_allow_html=True)



