import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import json
import zstandard as zstd
from PIL import Image
import io
import base64
import math
import time

# Add backend to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.sbox_math import SBoxMath
from app.aes_engine import AES

# --- CUSTOM CSS WITH BOOTSTRAP INJECTION ---
st.set_page_config(
    page_title="AES S-Box Analyzer Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Bootstrap 5 & Custom Overrides
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
    /* Global Overrides to make Streamlit look less 'Streamlit' */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Card Styling */
    .card-box {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 4px solid #0d6efd;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
        margin-bottom: 5px;
    }
    
    /* Headers */
    h1 { color: #0d6efd; font-weight: 700; }
    h2 { color: #343a40; font-weight: 600; font-size: 1.5rem; margin-top: 1rem; }
    h3 { color: #495057; font-size: 1.2rem; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: white;
        border: 1px solid #dee2e6;
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        font-weight: 600;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d6efd !important;
        color: white !important;
        border-color: #0d6efd !important;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'sbox_math' not in st.session_state:
    st.session_state.sbox_math = SBoxMath()

if 'current_sbox' not in st.session_state:
    st.session_state.current_sbox = st.session_state.sbox_math.AES_SBOX
    st.session_state.sbox_name = "Standard AES"

# Helper Functions
def sbox_to_df(sbox):
    df = pd.DataFrame(np.array(sbox).reshape(16, 16))
    df.columns = [f"{i:X}" for i in range(16)]
    df.index = [f"{i:X}" for i in range(16)]
    return df.applymap(lambda x: f"{x:02X}")

def generate_random_affine():
    while True:
        mat = np.random.randint(0, 2, (8, 8), dtype=int)
        det = int(np.round(np.linalg.det(mat))) % 2
        if det == 1:
            break
    const_vec = np.random.randint(0, 2, 8, dtype=int).tolist()
    return mat.tolist(), const_vec

def calculate_metrics_pro(sbox):
    math = st.session_state.sbox_math
    if not math.check_bijective(sbox): return None
    return {
        "NL": math.calculate_nl(sbox),
        "SAC": math.calculate_sac(sbox),
        "BIC_NL": math.calculate_bic(sbox)[0],
        "BIC_SAC": math.calculate_bic(sbox)[1],
        "LAP": math.calculate_lap(sbox),
        "DAP": math.calculate_dap(sbox),
        "DU": math.calculate_du(sbox),
        "AD": math.calculate_ad(sbox)
    }

def encrypt_image_core(image_bytes, key, sbox, is_compact=False):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if is_compact:
        max_compact_size = 600
        if max(img.size) > max_compact_size:
             img.thumbnail((max_compact_size, max_compact_size), Image.Resampling.LANCZOS)
    else:
        max_dim = 4096
        if max(img.size) > max_dim:
             img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
             
    width, height = img.size
    dim_header = width.to_bytes(4, 'big') + height.to_bytes(4, 'big')
    pixel_data = img.tobytes()
    payload = dim_header + pixel_data
    
    cctx = zstd.ZstdCompressor(level=15)
    compressed = cctx.compress(payload)
    
    aes = AES(key, sbox)
    ciphertext = aes.encrypt_bytes_cbc(compressed)
    
    data_len = len(ciphertext)
    header = data_len.to_bytes(4, 'big')
    full_data = header + ciphertext
    
    pixels_needed = math.ceil(len(full_data) / 3)
    side = int(math.ceil(math.sqrt(pixels_needed)))
    padded_data = full_data + b'\x00' * ((side * side * 3) - len(full_data))
    
    enc_img = Image.frombytes('RGB', (side, side), padded_data)
    return img, enc_img, full_data

def decrypt_image_core(input_data, key, sbox, is_text_input=False):
    aes = AES(key, sbox)
    
    if is_text_input:
        # Input is Base64 string of the RAW AES OUTPUT (compressed), 
        # NOT the container image.
        # This matches backend '/decrypt-image-text' logic
        try:
            encrypted_bytes_compressed = base64.b64decode(input_data)
        except Exception as e:
            raise ValueError(f"Invalid Base64: {str(e)}")
            
        # Decrypt directly
        decrypted_compressed = aes.decrypt_bytes_cbc(encrypted_bytes_compressed)
        
    else:
        # Input is Container Image Bytes (PNG format)
        container_img = Image.open(io.BytesIO(input_data)).convert('RGB')
        container_bytes = container_img.tobytes()
        
        data_len = int.from_bytes(container_bytes[:4], 'big')
        if data_len > len(container_bytes) - 4:
            raise ValueError("Corrupted container header")
            
        encrypted_stream = container_bytes[4 : 4 + data_len]
        
        # Decrypt
        decrypted_compressed = aes.decrypt_bytes_cbc(encrypted_stream)
    
    # Decompress (Common path)
    dctx = zstd.ZstdDecompressor()
    plaintext = dctx.decompress(decrypted_compressed)
    
    # Reconstruct
    width = int.from_bytes(plaintext[:4], 'big')
    height = int.from_bytes(plaintext[4:8], 'big')
    pixel_data = plaintext[8:]
    
    if len(pixel_data) < width * height * 3:
        raise ValueError("Incomplete pixel data")
        
    dec_img = Image.frombytes('RGB', (width, height), pixel_data)
    return dec_img


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("AES Analyzer")
    st.caption("Advanced Cryptographic Tool")
    
    mode = st.radio("Select Module", ["S-Box Lab", "Text Vault", "Image Cipher Pro"])
    
    st.divider()
    st.markdown("""
    <div style='background-color:#e9ecef; padding:10px; border-radius:8px;'>
        <small><b>Status:</b> Ready<br>
        <b>Engine:</b> AES-128 CBC<br>
        <b>S-Box:</b> Active</small>
    </div>
    """, unsafe_allow_html=True)

# === TAB 1: S-BOX LAB ===
if mode == "S-Box Lab":
    st.markdown("<div class='card-box'>", unsafe_allow_html=True)
    st.title("🧪 S-Box Laboratory")
    
    col_sel, col_act = st.columns([3, 1])
    with col_sel:
        preset = st.selectbox("Load Preset", ["Standard AES", "S-Box 44 (Optimized)", "Random Generated"])
    
    if preset == "Standard AES":
        st.session_state.current_sbox = st.session_state.sbox_math.AES_SBOX
        st.session_state.sbox_name = "Standard AES"
    elif preset == "S-Box 44 (Optimized)":
        st.session_state.current_sbox = st.session_state.sbox_math.SBOX_44
        st.session_state.sbox_name = "S-Box 44"
    elif preset == "Random Generated":
        if st.button("🎲 Generate New", use_container_width=True):
            mat, vec = generate_random_affine()
            st.session_state.current_sbox = st.session_state.sbox_math.generate_sbox(mat, vec)
            st.session_state.sbox_name = f"Random-{int(time.time())}"
            st.success("New S-Box Generated!")
            
    st.markdown("</div>", unsafe_allow_html=True)

    # Visualization
    tab_vis, tab_met, tab_exp = st.tabs(["Visualization", "Cryptographic Metrics", "Export"])
    
    with tab_vis:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.dataframe(sbox_to_df(st.session_state.current_sbox), height=600, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_met:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        if st.button("🚀 Run Deep Analysis"):
            with st.spinner("Calculating non-linearity, SAC, and algebraic degree..."):
                metrics = calculate_metrics_pro(st.session_state.current_sbox)
                st.session_state.last_metrics = metrics
        
        if 'last_metrics' in st.session_state and st.session_state.last_metrics:
            m = st.session_state.last_metrics
            
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-card'><div class='metric-label'>Nonlinearity</div><div class='metric-value'>{m['NL']}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><div class='metric-label'>Diff. Uniformity</div><div class='metric-value'>{m['DU']}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-label'>Alg. Degree</div><div class='metric-value'>{m['AD']}</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><div class='metric-label'>LAP</div><div class='metric-value'>{m['LAP']:.4f}</div></div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-card'><div class='metric-label'>SAC</div><div class='metric-value'>{m['SAC']:.4f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><div class='metric-label'>BIC (NL)</div><div class='metric-value'>{m['BIC_NL']}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-label'>BIC (SAC)</div><div class='metric-value'>{m['BIC_SAC']:.4f}</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><div class='metric-label'>DAP</div><div class='metric-value'>{m['DAP']:.4f}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_exp:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        if st.button("Prepare Excel Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_sb = sbox_to_df(st.session_state.current_sbox)
                df_sb.to_excel(writer, sheet_name='S-Box')
                if 'last_metrics' in st.session_state and st.session_state.last_metrics:
                    df_m = pd.DataFrame(list(st.session_state.last_metrics.items()), columns=['Metric', 'Value'])
                    df_m.to_excel(writer, sheet_name='Metrics', index=False)
            
            st.download_button(
                label="📥 Download Excel (.xlsx)",
                data=output.getvalue(),
                file_name=f"sbox_analysis_{st.session_state.sbox_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.markdown("</div>", unsafe_allow_html=True)

# === TAB 2: TEXT VAULT ===
elif mode == "Text Vault":
    st.markdown("<div class='card-box'>", unsafe_allow_html=True)
    st.title("🔐 Text Vault")
    
    key = st.text_input("Encryption Key", type="password", help="Must be 16 bytes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Encrypt")
        txt_in = st.text_area("Plaintext", height=150)
        if st.button("Encrypt", key="btn_enc_txt"):
            if not key:
                st.error("Key is required!")
            else:
                aes = AES(key, st.session_state.current_sbox)
                try:
                    res = aes.encrypt_cbc(txt_in)
                    st.session_state.txt_res = res
                    st.success("Encrypted!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Decrypt")
        val = st.session_state.get('txt_res', '')
        txt_out = st.text_area("Ciphertext (Hex)", value=val, height=150)
        if st.button("Decrypt", key="btn_dec_txt"):
            if not key:
                st.error("Key is required!")
            else:
                aes = AES(key, st.session_state.current_sbox)
                try:
                    res = aes.decrypt_cbc(txt_out)
                    st.success("Decrypted!")
                    st.code(res, language='text')
                except Exception as e:
                    st.error(f"Decryption failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# === TAB 3: IMAGE CIPHER PRO ===
elif mode == "Image Cipher Pro":
    st.title("🖼️ Image Cipher Pro")
    st.info(f"Active S-Box: **{st.session_state.sbox_name}**")
    
    key_img = st.text_input("Secret Key", type="password")
    
    tab_enc_img, tab_dec_img = st.tabs(["Encrypt", "Decrypt"])
    
    with tab_enc_img:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.subheader("Encrypt Image")
        img_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        compact_mode = st.checkbox("Compact Mode (Resize to 600px)", value=True)
        
        if img_file and key_img:
            if st.button("Encrypt Image"):
                with st.spinner("Processing..."):
                    try:
                        orig, enc_img, raw_data = encrypt_image_core(
                            img_file.read(), 
                            key_img, 
                            st.session_state.current_sbox, 
                            compact_mode
                        )
                        
                        c1, c2 = st.columns(2)
                        with c1: st.image(orig, caption="Original", use_container_width=True)
                        with c2: st.image(enc_img, caption="Encrypted", use_container_width=True)
                            
                        b64_str = base64.b64encode(raw_data).decode('ascii')
                        
                        st.divider()
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            # Save container image
                            buf = io.BytesIO()
                            enc_img.save(buf, format="PNG")
                            st.download_button("📥 Download Encrypted Image", buf.getvalue(), "encrypted.png", "image/png")
                        with col_d2:
                            # Save text file (Fixing text/plain issue by offering download)
                            st.download_button("📋 Download Ciphertext (.txt)", b64_str, "ciphertext.txt", "text/plain")
                            
                    except Exception as e:
                        st.error(f"Encryption Error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
                        
    with tab_dec_img:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)
        st.subheader("Decrypt Image")
        
        # New: Tabbed input for Decryption
        dec_mode = st.radio("Input Source", ["Upload Image (.png)", "Upload Text File (.txt)"])
        
        dec_data = None
        is_text_input = False
        
        if dec_mode == "Upload Image (.png)":
            f = st.file_uploader("Upload Container Image", type=['png'])
            if f: dec_data = f.read()
            
        else: # Text File Upload
            f = st.file_uploader("Upload Ciphertext File", type=['txt'])
            if f:
                # Read text file content
                dec_data = f.read().decode('utf-8')
                is_text_input = True
            
            # Manual paste fallback
            txt_paste = st.text_area("Or Paste Base64 String (for small data)")
            if txt_paste and not dec_data:
                dec_data = txt_paste
                is_text_input = True
                
        if st.button("Decrypt Image") and dec_data and key_img:
            with st.spinner("Decrypting..."):
                try:
                    res_img = decrypt_image_core(dec_data, key_img, st.session_state.current_sbox, is_text_input)
                    st.success("Decryption Successful!")
                    st.image(res_img, caption="Recovered Image", use_container_width=True)
                    
                    buf = io.BytesIO()
                    res_img.save(buf, format="PNG")
                    st.download_button("📥 Download Result", buf.getvalue(), "decrypted.png", "image/png")
                except Exception as e:
                    st.error(f"Decryption failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

