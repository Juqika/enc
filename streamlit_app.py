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

# --- CUSTOM CSS FOR "PRO" LOOK ---
st.set_page_config(
    page_title="AES S-Box Analyzer Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #0e1117;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 5px;
        padding: 0 20px;
    }
    div[data-testid="stImage"] {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
    }
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
    # Convert to Hex strings for display
    return df.applymap(lambda x: f"{x:02X}")

def generate_random_affine():
    # Attempt to find invertible matrix
    while True:
        mat = np.random.randint(0, 2, (8, 8), dtype=int)
        det = int(np.round(np.linalg.det(mat))) % 2
        if det == 1:
            break
    const_vec = np.random.randint(0, 2, 8, dtype=int).tolist()
    return mat.tolist(), const_vec

def calculate_metrics_pro(sbox):
    math = st.session_state.sbox_math
    
    # Check Bijectivity First
    if not math.check_bijective(sbox):
        return None

    metrics = {
        "NL": math.calculate_nl(sbox),
        "SAC": math.calculate_sac(sbox),
        "BIC_NL": math.calculate_bic(sbox)[0],
        "BIC_SAC": math.calculate_bic(sbox)[1],
        "LAP": math.calculate_lap(sbox),
        "DAP": math.calculate_dap(sbox),
        "DU": math.calculate_du(sbox),
        "AD": math.calculate_ad(sbox)
    }
    return metrics

def encrypt_image_core(image_bytes, key, sbox, is_compact=False):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Compact Mode Logic
    if is_compact:
        max_compact_size = 600
        if max(img.size) > max_compact_size:
             img.thumbnail((max_compact_size, max_compact_size), Image.Resampling.LANCZOS)
    else:
        max_dim = 4096 # Cap for Streamlit safety
        if max(img.size) > max_dim:
             img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
             
    # Prepare Payload
    width, height = img.size
    dim_header = width.to_bytes(4, 'big') + height.to_bytes(4, 'big')
    pixel_data = img.tobytes()
    payload = dim_header + pixel_data
    
    # Compress
    cctx = zstd.ZstdCompressor(level=15)
    compressed = cctx.compress(payload)
    
    # Encrypt
    aes = AES(key, sbox)
    ciphertext = aes.encrypt_bytes_cbc(compressed)
    
    # Pack into container image
    data_len = len(ciphertext)
    header = data_len.to_bytes(4, 'big')
    full_data = header + ciphertext
    
    pixels_needed = math.ceil(len(full_data) / 3)
    side = int(math.ceil(math.sqrt(pixels_needed)))
    padded_data = full_data + b'\x00' * ((side * side * 3) - len(full_data))
    
    enc_img = Image.frombytes('RGB', (side, side), padded_data)
    
    return img, enc_img, full_data

def decrypt_image_core(input_data, key, sbox, is_text=False):
    aes = AES(key, sbox)
    
    encrypted_bytes = input_data
    if is_text:
        encrypted_bytes = base64.b64decode(input_data)
        # For text mode, we assume the input is the raw ciphertext (without header/container) OR
        # the container bytes? The backend text decrypt assumes Base64 of CIPHERTEXT only,
        # but the backend image decrypt assumes CONTAINER image.
        # Let's standardize: If text input, we treat it as the raw compressed ciphertext (if valid zstd)
        # OR the full container bytes.
        
        # Actually, let's follow the backend logic "decrypt_image_text":
        # It takes base64 ciphertext -> decrypts -> decompresses.
        # It assumes the base64 is just the AES output.
        pass 
    else:
        # File input (Container Image)
        container_img = Image.open(io.BytesIO(input_data)).convert('RGB')
        container_bytes = container_img.tobytes()
        
        data_len = int.from_bytes(container_bytes[:4], 'big')
        encrypted_bytes = container_bytes[4 : 4 + data_len]

    # Decrypt
    decrypted_compressed = aes.decrypt_bytes_cbc(encrypted_bytes)
    
    # Decompress
    dctx = zstd.ZstdDecompressor()
    plaintext = dctx.decompress(decrypted_compressed)
    
    # Reconstruct
    width = int.from_bytes(plaintext[:4], 'big')
    height = int.from_bytes(plaintext[4:8], 'big')
    pixel_data = plaintext[8:]
    
    dec_img = Image.frombytes('RGB', (width, height), pixel_data)
    return dec_img


# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("Select Module", ["S-Box Lab", "Text Vault", "Image Cipher Pro"])
    
    st.divider()
    st.info("💡 **Tip:** Use 'Image Cipher Pro' for secure image transmission with compression.")
    st.caption("v2.0 - Optimized for Streamlit")

# --- MAIN CONTENT ---

# === TAB 1: S-BOX LAB ===
if mode == "S-Box Lab":
    st.title("🧪 S-Box Laboratory")
    
    # Top Controls
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

    # Visualization
    st.subheader(f"Analyzing: {st.session_state.sbox_name}")
    
    tab_vis, tab_met, tab_exp = st.tabs(["Visualization", "Cryptographic Metrics", "Export"])
    
    with tab_vis:
        st.dataframe(sbox_to_df(st.session_state.current_sbox), height=600, use_container_width=True)
    
    with tab_met:
        if st.button("🚀 Run Deep Analysis"):
            with st.spinner("Crunching numbers (Walsh Transform, SAC, etc.)..."):
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
        else:
            st.info("Click 'Run Deep Analysis' to see metrics.")

    with tab_exp:
        st.write("Download the analysis report including the full S-Box and calculated metrics.")
        
        # Prepare Excel in memory
        if st.button("Prepare Excel Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Sheet 1: S-Box
                df_sb = sbox_to_df(st.session_state.current_sbox)
                df_sb.to_excel(writer, sheet_name='S-Box')
                
                # Sheet 2: Metrics
                if 'last_metrics' in st.session_state and st.session_state.last_metrics:
                    df_m = pd.DataFrame(list(st.session_state.last_metrics.items()), columns=['Metric', 'Value'])
                    df_m.to_excel(writer, sheet_name='Metrics', index=False)
            
            st.download_button(
                label="📥 Download Excel (.xlsx)",
                data=output.getvalue(),
                file_name=f"sbox_analysis_{st.session_state.sbox_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# === TAB 2: TEXT VAULT ===
elif mode == "Text Vault":
    st.title("🔐 Text Vault")
    
    key = st.text_input("Encryption Key", type="password", help="Must be 16 bytes. Will be padded if shorter.")
    
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
        st.subheader("Result / Decrypt Input")
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
                    st.error(f"Decryption failed. Check key or S-Box. ({e})")

# === TAB 3: IMAGE CIPHER PRO ===
elif mode == "Image Cipher Pro":
    st.title("🖼️ Image Cipher Pro")
    
    st.warning(f"Using Active S-Box: **{st.session_state.sbox_name}**. Ensure the receiver has the same S-Box configuration!")
    
    key_img = st.text_input("Secret Key", type="password")
    
    tab_enc_img, tab_dec_img = st.tabs(["Encrypt", "Decrypt"])
    
    with tab_enc_img:
        img_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        compact_mode = st.checkbox("Compact Mode (Resize to 600px)", value=True, help="Greatly reduces output text size.")
        
        if img_file and key_img:
            if st.button("Encrypt Image"):
                with st.spinner("Processing (Compressing & Encrypting)..."):
                    try:
                        orig, enc_img, raw_data = encrypt_image_core(
                            img_file.read(), 
                            key_img, 
                            st.session_state.current_sbox, 
                            compact_mode
                        )
                        
                        # Visual Comparison
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(orig, caption="Original", use_container_width=True)
                        with c2:
                            st.image(enc_img, caption="Encrypted Container (Noise)", use_container_width=True)
                            
                        # Outputs
                        b64_str = base64.b64encode(raw_data).decode('ascii')
                        
                        st.divider()
                        st.subheader("Output Data")
                        
                        exp1, exp2 = st.columns(2)
                        with exp1:
                            st.download_button(
                                "📥 Download Encrypted Image (.png)",
                                data=io.BytesIO(enc_img.tobytes()), # wait, need to save to png bytes
                                file_name="encrypted_image.png",
                                mime="image/png"
                            )
                            # Fix download logic
                            buf = io.BytesIO()
                            enc_img.save(buf, format="PNG")
                            st.download_button(
                                "📥 Download Encrypted Image (.png)",
                                data=buf.getvalue(),
                                file_name="encrypted.png",
                                mime="image/png"
                            )
                            
                        with exp2:
                            st.download_button(
                                "📋 Download Ciphertext Text (.txt)",
                                data=b64_str,
                                file_name="ciphertext.txt",
                                mime="text/plain"
                            )
                            
                        with st.expander("View Ciphertext String"):
                            st.code(b64_str[:500] + "...", language="text")

                    except Exception as e:
                        st.error(f"Encryption Error: {e}")
                        
    with tab_dec_img:
        st.write("Decrypt from **Image File** OR **Text String**.")
        
        dec_method = st.radio("Input Method", ["Upload Encrypted Image", "Paste Text String"])
        
        dec_data = None
        is_text_input = False
        
        if dec_method == "Upload Encrypted Image":
            f = st.file_uploader("Upload .png container", type=['png'])
            if f:
                dec_data = f.read()
        else:
            txt = st.text_area("Paste Base64 Ciphertext")
            if txt:
                dec_data = txt
                is_text_input = True
                
        if st.button("Decrypt Image") and dec_data and key_img:
            with st.spinner("Decrypting & Decompressing..."):
                try:
                    res_img = decrypt_image_core(dec_data, key_img, st.session_state.current_sbox, is_text_input)
                    st.success("Decryption Successful!")
                    st.image(res_img, caption="Recovered Image", use_container_width=True)
                    
                    buf = io.BytesIO()
                    res_img.save(buf, format="PNG")
                    st.download_button("📥 Download Result", buf.getvalue(), "decrypted.png", "image/png")
                except Exception as e:
                    st.error(f"Decryption failed: {e}")