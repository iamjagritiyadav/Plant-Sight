# Replace the entire prediction display section (starting from "if confident and valid_class and name_crop_flag:")
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
        
        with st.expander("📖 Read detailed guidance", expanded=False):
            # Properly format the details with clear sections
            details_lines = details.split('\n')
            in_section = False
            current_section = ""
            
            for line in details_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Detect section headers (lines ending with colon)
                if line.endswith(':'):
                    if current_section:
                        st.markdown("")  # Add space between sections
                    
                    # Apply proper formatting for section headers
                    if line.lower() in ['symptoms:', 'cultural controls:', 'biological controls:', 
                                      'chemical controls:', 'monitoring:', 'notes:']:
                        st.markdown(f"**{line}**")
                        current_section = line
                        in_section = True
                    else:
                        st.write(line)
                        current_section = ""
                        in_section = False
                else:
                    # Content within sections
                    if in_section and line.startswith('- '):
                        st.markdown(f"• {line[2:]}")  # Convert to bullet points
                    elif in_section:
                        st.write(line)
                    else:
                        st.write(line)
    else:
        st.markdown("#### 🌱 **Recommended Action**")
        st.write(BUILTIN_REMEDIES_SHORT.get(top_id, "General guidance — consult local extension."))

    st.caption("Remedies are guidance only — consult local extension for chemicals & dosages.")
    
    # FIXED: Create proper PDF content (not just text file)
    import base64
    from fpdf import FPDF
    import tempfile
    import os
    
    # Create proper PDF summary
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Plant Sight - Disease Detection Report', 0, 1, 'C')
            self.ln(5)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    try:
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Plant Sight Detection Report", 0, 1, 'C')
        pdf.ln(10)
        
        # Prediction info
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Prediction: {top_name}", 0, 1)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Confidence: {top_conf*100:.1f}%", 0, 1)
        pdf.ln(5)
        
        # Recommended action
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Recommended Action:", 0, 1)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, summary)
        pdf.ln(5)
        
        # Detailed guidance
        if entry and details:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Detailed Guidance:", 0, 1)
            pdf.set_font("Arial", size=10)
            
            # Clean and format details for PDF
            clean_details = details.replace('    ', '  ').replace('  ', ' ')
            pdf.multi_cell(0, 6, clean_details)
        
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(0, 6, "Note: Remedies are guidance only — consult local extension for chemicals & dosages.")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf.output(tmp_file.name)
            
            # Read and encode for download
            with open(tmp_file.name, 'rb') as f:
                pdf_bytes = f.read()
            
            # Clean up
            os.unlink(tmp_file.name)
        
        # Download button for actual PDF
        st.download_button(
            "📥 Download summary (.pdf)",
            pdf_bytes,
            file_name=f"plantsight_{top_name.replace(' ', '_')}_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
    except Exception as e:
        # Fallback to text if PDF creation fails
        rep = f"Plant Sight Result\n\nPrediction: {top_name}\nConfidence: {top_conf*100:.1f}%\n\nRecommended Action:\n{summary}\n\nDetailed Guidance:\n{details if entry else ''}\n\nNote: Consult local extension for specific chemical recommendations."
        st.download_button(
            "📥 Download summary (.txt)",
            rep,
            file_name=f"plantsight_{top_name.replace(' ', '_')}.txt",
            use_container_width=True
        )

# REMOVE the separate image download button - it's redundant
# st.download_button("🖼️ Download image", image_bytes(pil), file_name="input.png", use_container_width=True)

