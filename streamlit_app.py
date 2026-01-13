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

# Add backend to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.sbox_math import SBoxMath
from app.aes_engine import AES

st.set_page_config(
    page_title="AES S-Box Analyzer & Encryptor",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if 'sbox_math' not in st.session_state:
    st.session_state.sbox_math = SBoxMath()

if 'current_sbox' not in st.session_state:
    st.session_state.current_sbox = st.session_state.sbox_math.AES_SBOX
    st.session_state.sbox_name = "Standard AES"

if 'affine_matrix' not in st.session_state:
    st.session_state.affine_matrix = None
    st.session_state.constant_vector = None

# Sidebar
st.sidebar.title("Configuration")
mode = st.sidebar.radio("Mode", ["Analysis & Generation", "Text Encryption", "Image Encryption"])

def hex_string(sbox):
    return " ".join([f"{x:02x}" for x in sbox])

def sbox_to_grid(sbox):
    df = pd.DataFrame(np.array(sbox).reshape(16, 16))
    df.columns = [f"{i:x}" for i in range(16)]
    df.index = [f"{i:x}" for i in range(16)]
    return df

# --- TAB 1: Analysis & Generation ---
if mode == "Analysis & Generation":
    st.title("AES S-Box Analysis & Generation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("S-Box Selection")
        sbox_type = st.selectbox(
            "Choose S-Box Source",
            ["Standard AES", "S-Box 44 (Paper)", "Random Affine", "Custom Affine"]
        )
        
        if sbox_type == "Standard AES":
            st.session_state.current_sbox = st.session_state.sbox_math.AES_SBOX
            st.session_state.sbox_name = "Standard AES"
            
        elif sbox_type == "S-Box 44 (Paper)":
            st.session_state.current_sbox = st.session_state.sbox_math.SBOX_44
            st.session_state.sbox_name = "S-Box 44"
            
        elif sbox_type == "Random Affine":
            if st.button("Generate New Random S-Box"):
                # Simple random generation logic for demo (usually you'd want invertible matrices)
                # For now, let's try to generate until we get a bijective one or just random
                # Ideally, we implement the matrix generation logic here or in sbox_math
                # Since sbox_math doesn't have a 'generate_random_affine' public method exposed directly 
                # (it was likely in main.py or just math helpers), we will simulate or skip for now
                # typically you generate 8x8 matrix and check det != 0.
                
                # Placeholder for robust random generation:
                # In a real app, we'd loop until an invertible matrix is found.
                # For this simplified port, we might just stick to presets or implement a simple generator.
                
                # Let's try to implement a basic one:
                found = False
                while not found:
                    mat = np.random.randint(0, 2, (8, 8)).tolist()
                    # Check invertibility over GF(2) (det % 2 != 0)
                    if np.linalg.det(mat) % 2 != 0:
                        found = True
                        vec = np.random.randint(0, 2, 8).tolist()
                        st.session_state.affine_matrix = mat
                        st.session_state.constant_vector = vec
                        st.session_state.current_sbox = st.session_state.sbox_math.generate_sbox(mat, vec)
                        st.session_state.sbox_name = "Random Generated"
                        st.success("New S-Box Generated!")
        
        elif sbox_type == "Custom Affine":
            st.info("Custom Matrix input not fully implemented in this demo.")

        st.metric("S-Box Name", st.session_state.sbox_name)
        
        st.divider()
        st.subheader("Properties")
        is_bijective = st.session_state.sbox_math.check_bijective(st.session_state.current_sbox)
        st.write(f"**Bijective:** {'✅ Yes' if is_bijective else '❌ No'}")
        
    with col2:
        st.subheader("Cryptographic Metrics")
        
        if st.button("Analyze Current S-Box"):
            with st.spinner("Calculating metrics..."):
                sb = st.session_state.current_sbox
                math = st.session_state.sbox_math
                
                nl = math.calculate_nl(sb)
                sac = math.calculate_sac(sb)
                bic_nl, bic_sac = math.calculate_bic(sb)
                lap = math.calculate_lap(sb)
                dap = math.calculate_dap(sb)
                du = math.calculate_du(sb)
                ad = math.calculate_ad(sb)
                
                # Metrics Display
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Nonlinearity (NL)", nl)
                m_col2.metric("SAC", f"{sac:.4f}")
                m_col3.metric("BIC (NL)", f"{bic_nl}")
                
                m_col1.metric("BIC (SAC)", f"{bic_sac:.4f}")
                m_col2.metric("LAP", f"{lap:.4f}")
                m_col3.metric("DAP", f"{dap:.4f}")
                
                m_col1.metric("Diff. Uniformity", du)
                m_col2.metric("Alg. Degree", ad)

        st.subheader("S-Box Visualization")
        st.dataframe(sbox_to_grid(st.session_state.current_sbox), height=600)

# --- TAB 2: Text Encryption ---
elif mode == "Text Encryption":
    st.title("Text Encryption (AES-CBC)")
    
    st.info(f"Using S-Box: **{st.session_state.sbox_name}**")
    
    key_input = st.text_input("Encryption Key (16 chars recommended)", "secret_key_12345")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Encrypt")
        plaintext = st.text_area("Plaintext", "Hello, World!")
        if st.button("Encrypt Text"):
            try:
                engine = AES(key_input, st.session_state.current_sbox)
                ciphertext = engine.encrypt_cbc(plaintext)
                st.session_state.last_ciphertext = ciphertext
                st.success("Encryption Successful")
                st.code(ciphertext, language="text")
            except Exception as e:
                st.error(f"Error: {e}")
                
    with col2:
        st.subheader("Decrypt")
        ciphertext_input = st.text_area("Ciphertext (Hex)", st.session_state.get('last_ciphertext', ''))
        if st.button("Decrypt Text"):
            try:
                engine = AES(key_input, st.session_state.current_sbox)
                decrypted = engine.decrypt_cbc(ciphertext_input)
                st.success("Decryption Successful")
                st.code(decrypted, language="text")
            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 3: Image Encryption ---
elif mode == "Image Encryption":
    st.title("Image Encryption")
    st.info(f"Using S-Box: **{st.session_state.sbox_name}**")
    
    key_input_img = st.text_input("Encryption Key", "image_secret_key")
    
    tab_enc, tab_dec = st.tabs(["Encrypt Image", "Decrypt Image"])
    
    with tab_enc:
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        compress = st.checkbox("Compress Output (Zstd Level 15)", value=True)
        
        if uploaded_file and st.button("Encrypt Image"):
            with st.spinner("Processing..."):
                try:
                    # Load Image
                    image = Image.open(uploaded_file).convert('RGB')
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG') # Standardize to PNG
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Encrypt
                    engine = AES(key_input_img, st.session_state.current_sbox)
                    encrypted_data = engine.encrypt_bytes_cbc(img_bytes)
                    
                    final_data = encrypted_data
                    if compress:
                        cctx = zstd.ZstdCompressor(level=15)
                        final_data = cctx.compress(encrypted_data)
                    
                    # Encode for display/download
                    b64_str = base64.b64encode(final_data).decode('utf-8')
                    
                    st.success("Image Encrypted!")
                    st.text_area("Encrypted Base64 String", b64_str, height=150)
                    st.download_button("Download Encrypted Data", final_data, file_name="encrypted.bin")
                    
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_dec:
        st.write("Upload encrypted binary file or paste Base64 string")
        
        enc_file = st.file_uploader("Upload Encrypted File", type=['bin'])
        enc_text = st.text_area("Or Paste Base64 String")
        is_compressed = st.checkbox("Input is Compressed (Zstd)", value=True)
        
        if st.button("Decrypt Image"):
            with st.spinner("Decrypting..."):
                try:
                    data_to_decrypt = None
                    if enc_file:
                        data_to_decrypt = enc_file.read()
                    elif enc_text:
                        data_to_decrypt = base64.b64decode(enc_text)
                    
                    if data_to_decrypt:
                        if is_compressed:
                            dctx = zstd.ZstdDecompressor()
                            data_to_decrypt = dctx.decompress(data_to_decrypt)
                        
                        engine = AES(key_input_img, st.session_state.current_sbox)
                        decrypted_bytes = engine.decrypt_bytes_cbc(data_to_decrypt)
                        
                        # Load bytes back to image
                        dec_image = Image.open(io.BytesIO(decrypted_bytes))
                        st.image(dec_image, caption="Decrypted Image")
                        
                        # Download button for result
                        buf = io.BytesIO()
                        dec_image.save(buf, format="PNG")
                        st.download_button("Download Image", buf.getvalue(), file_name="decrypted.png", mime="image/png")
                    else:
                        st.warning("Please provide input data.")
                except Exception as e:
                    st.error(f"Decryption Failed: {e}")

