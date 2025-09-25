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
from fpdf import FPDF   # <-- added for PDF export

# ---------------- Config ----------------
MODEL_PATH = "best.pt"
TOP_K = 3
CONF_THRESHOLD = 0.70
REJECTED_DIR = "rejected"
CROP_KEYWORDS = ["cotton", "maize", "wheat", "rice", "sugarcane"]

os.makedirs(REJECTED_DIR, exist_ok=True)

# ... (baaki tumhara pura code jaisa ka taisa rahega)

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
            preds_raw = extract_topk_from_result(r, k=TOP_K)
        except Exception as e:
            st.error("Prediction error: " + str(e))
            st.write(traceback.format_exc())
            st.stop()

    display_preds = []
    for cid, conf in preds_raw:
        nconf = normalize_conf(conf)
        name = CLASS_MAPPING.get(int(cid), f"Class {cid}")
        display_preds.append((name, nconf, int(cid)))

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

            entry = REMEDIES_FULL.get(top_id, None)
            summary = entry.get("summary", "") if entry else ""
            details = entry.get("details", "") if entry else ""

            st.markdown("#### 🌱 **Recommended Action**")
            st.write(summary)
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
            st.download_button("📄 Download PDF Report", data=pdf_bytes,
                               file_name="plantsight_report.pdf", mime="application/pdf",
                               use_container_width=True)

        else:
            st.markdown("#### ⚠️ **No valid crop prediction**")
            if not confident:
                st.error("Model confidence is low. Try a clearer close-up.")
            elif not name_crop_flag:
                st.error("This image does not look like a crop part.")
            else:
                st.error("No valid prediction — try another image.")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Plant Sight • Fast disease ID • Guidance only — consult local extension for chemicals & dosages</div>', unsafe_allow_html=True)


