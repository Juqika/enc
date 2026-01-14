from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, Response, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .sbox_math import sbox_math
from .schemas import AffineMatrixInput, AnalysisResult, EncryptionRequest, DecryptionRequest
from .aes_engine import AES
from PIL import Image
import numpy as np
import pandas as pd
import io
import base64
import json
import time
import math
import zstandard as zstd
from typing import Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import os

# Configure logging
logger = logging.getLogger("uvicorn")

# Global resources for heavy image operations
MAX_CONCURRENT_IMAGE_OPS = 2
image_processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_OPS)
thread_pool = ThreadPoolExecutor(max_workers=4)

app = FastAPI()

# Absolute Path Setup for Static Files (Fix for Render)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Analysis", "Content-Disposition", "X-Sbox-Type", "X-Encryption-Time", "X-Histogram-Data"]
)

# --- Helper Functions for Image Processing ---

async def _read_upload_file_limited(file: UploadFile, limit_mb: int = 100):
    contents = await file.read()
    if len(contents) > limit_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {limit_mb}MB.")
    return contents

def _process_image_histogram(img_array):
    return sbox_math.get_histogram(img_array)

def _prepare_key(key_str):
    key_bytes = key_str.encode('utf-8')
    if len(key_bytes) < 16:
        key_bytes = key_bytes + b'\0' * (16 - len(key_bytes))
    return key_bytes[:16]

def _encrypt_image_data(image_data, key_bytes, sbox_list, poly=0x11B):
    aes = AES(key_bytes.decode('utf-8', errors='ignore'), sbox_list, poly)
    return aes.encrypt_bytes_cbc(image_data)

def analyze_image_encryption(orig_arr, enc_arr):
    entropy_orig = sbox_math.calculate_entropy(orig_arr)
    entropy_enc = sbox_math.calculate_entropy(enc_arr)
    npcr = sbox_math.calculate_npcr(orig_arr, enc_arr)
    uaci = sbox_math.calculate_uaci(orig_arr, enc_arr)
    corr_orig = sbox_math.calculate_correlation(orig_arr)
    corr_enc = sbox_math.calculate_correlation(enc_arr)
    
    return {
        "original_entropy": entropy_orig,
        "encrypted_entropy": entropy_enc,
        "npcr": npcr,
        "uaci": uaci,
        "corr_orig": corr_orig,
        "corr_enc": corr_enc
    }

# Mount static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/encrypt")
def encrypt_text(req: EncryptionRequest):
    try:
        aes = AES(req.key, req.sbox)
        ciphertext = aes.encrypt_cbc(req.plaintext)
        return {"ciphertext": ciphertext}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/decrypt")
def decrypt_text(req: DecryptionRequest):
    try:
        aes = AES(req.key, req.sbox)
        plaintext = aes.decrypt_cbc(req.ciphertext)
        return {"plaintext": plaintext}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/paper-presets")
def get_paper_presets():
    return {
        "A0": sbox_math.PAPER_A0,
        "A1": sbox_math.PAPER_A1,
        "A2": sbox_math.PAPER_A2,
        "K44": sbox_math.PAPER2_K44,
        "K81": sbox_math.PAPER2_K81,
        "K111": sbox_math.PAPER2_K111,
        "constant": sbox_math.AES_CONSTANT,
        "poly": sbox_math.PAPER_POLY
    }

@app.post("/analyze", response_model=AnalysisResult)
def analyze_sbox(input_data: AffineMatrixInput):
    sbox = []
    if input_data.sbox:
        if len(input_data.sbox) != 256:
            raise HTTPException(status_code=400, detail="Imported S-Box must contain exactly 256 integers.")
        sbox = input_data.sbox
    elif input_data.matrix and input_data.constant:
        sbox_math.set_irreducible_poly(input_data.poly)
        sbox = sbox_math.generate_sbox(input_data.matrix, input_data.constant)
    else:
        raise HTTPException(status_code=400, detail="Invalid input: Provide either an S-Box OR an Affine Matrix + Constant.")
    
    is_bijective = sbox_math.check_bijective(sbox)
    is_balanced = sbox_math.check_balance_bits(sbox)
    
    metrics = {}
    if is_bijective:
        metrics["NL"] = sbox_math.calculate_nl(sbox)
        metrics["SAC"] = sbox_math.calculate_sac(sbox)
        bic_nl, bic_sac = sbox_math.calculate_bic(sbox)
        metrics["BIC_NL"] = bic_nl
        metrics["BIC_SAC"] = bic_sac
        metrics["LAP"] = sbox_math.calculate_lap(sbox)
        metrics["DAP"] = sbox_math.calculate_dap(sbox)
        metrics["DU"] = sbox_math.calculate_du(sbox)
        metrics["AD"] = sbox_math.calculate_ad(sbox)
        metrics["TO"] = sbox_math.calculate_to(sbox)
        metrics["CI"] = sbox_math.calculate_ci(sbox)
    
    return {
        "sbox": sbox,
        "is_bijective": is_bijective,
        "is_balanced": is_balanced,
        "metrics": metrics,
        "comparison": {} 
    }

@app.post("/encrypt-image")
async def encrypt_image(
    file: UploadFile = File(...),
    key: str = Form(...),
    sbox: str = Form(...),
    poly: int = Form(0x11B),
    compact: str = Form("false"),
    use_compression: str = Form("true")
):
    async with image_processing_semaphore:
        try:
            is_compact = compact.lower() == 'true'
            do_compress = use_compression.lower() == 'true'
            
            logger.info(f"🎨 Starting image encryption (Compact: {is_compact}, Compress: {do_compress})")
            
            sbox_list = json.loads(sbox)
            if len(sbox_list) != 256:
                raise HTTPException(status_code=400, detail="S-Box must be 256 elements")
            
            start_time = time.time()
            logger.info(f"📥 Reading uploaded file: {file.filename}")
            image_bytes = await _read_upload_file_limited(file)
            
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            if is_compact:
                max_compact_size = 600
                if max(img.size) > max_compact_size:
                    img.thumbnail((max_compact_size, max_compact_size), Image.Resampling.LANCZOS)
            else:
                max_dimension = 8192
                if max(img.size) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            img_array = np.array(img)
            
            try:
                original_histogram = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, 
                    _process_image_histogram, 
                    img_array
                )
            except Exception as e:
                logger.error(f"❌ Histogram error: {e}")
                raise
            
            # Ensure metrics comparison uses RGB (drop alpha if present)
            if img.mode == 'RGBA':
                original_array_for_metrics = np.array(img.convert('RGB'))
            else:
                original_array_for_metrics = img_array.copy()

            del img_array
            gc.collect()
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.close()
            
            width, height = img.size
            dim_header = width.to_bytes(4, 'big') + height.to_bytes(4, 'big')
            pixel_data = img.tobytes()
            payload = dim_header + pixel_data
            
            final_payload = payload
            if do_compress:
                logger.info("🗜️ Compressing payload with Zstandard (Level 3)...")
                cctx = zstd.ZstdCompressor(level=3)
                final_payload = cctx.compress(payload)
            else:
                logger.info("⏩ Skipping compression")

            key_bytes = _prepare_key(key)
            
            ciphertext_bytes = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                _encrypt_image_data,
                final_payload, key_bytes, sbox_list, poly
            )
            
            data_len = len(ciphertext_bytes)
            header = data_len.to_bytes(4, byteorder='big')
            full_encrypted_data = header + ciphertext_bytes
            
            pixels_needed = math.ceil(len(full_encrypted_data) / 3)
            side = int(math.ceil(math.sqrt(pixels_needed)))
            total_bytes_needed = side * side * 3
            padded_data = full_encrypted_data + b'\x00' * (total_bytes_needed - len(full_encrypted_data))
            
            enc_img = Image.frombytes('RGB', (side, side), padded_data)
            
            enc_array = np.array(enc_img)
            
            def _get_analysis_sync(orig_arr, enc_arr):
                hist = sbox_math.get_histogram(enc_arr)
                metrics = analyze_image_encryption(orig_arr, enc_arr)
                return hist, metrics
            
            encrypted_histogram, security_metrics = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                _get_analysis_sync,
                original_array_for_metrics,
                enc_array
            )
            
            analysis = {
                "original_entropy": security_metrics["original_entropy"],
                "encrypted_entropy": security_metrics["encrypted_entropy"],
                "npcr": security_metrics["npcr"],
                "uaci": security_metrics["uaci"],
                "corr_orig": security_metrics["corr_orig"],
                "corr_enc": security_metrics["corr_enc"],
                "original_histogram": original_histogram,
                "encrypted_histogram": encrypted_histogram
            }
            
            del original_array_for_metrics
            del enc_array
            gc.collect()
            
            output = io.BytesIO()
            enc_img.save(output, format="PNG")
            output.seek(0)
            
            encryption_time = (time.time() - start_time) * 1000
            
            return {
                "image_data": base64.b64encode(output.getvalue()).decode('ascii'),
                "ciphertext": base64.b64encode(ciphertext_bytes).decode('ascii'),
                "analysis": analysis,
                "encryption_time": encryption_time
            }

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"❌ FATAL ERROR in encrypt_image: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)

@app.post("/decrypt-image")
async def decrypt_image(
    file: UploadFile = File(...),
    key: str = Form(...),
    sbox: str = Form(...),
    poly: int = Form(0x11B)
):
    async with image_processing_semaphore:
        try:
            logger.info(f"🔓 Starting image decryption (File: {file.filename})...")
            sbox_list = json.loads(sbox)
            aes = AES(key, sbox_list, poly)
            
            # Read file content
            file_content = await _read_upload_file_limited(file)
            
            encrypted_stream = None
            
            # Detect file type based on extension or content
            is_text_file = file.filename.lower().endswith('.txt')
            
            if is_text_file:
                logger.info("📄 Processing as Ciphertext Text File")
                try:
                    # Text file contains Base64 string
                    b64_str = file_content.decode('utf-8').strip()
                    
                    # Remove Data URI header if present (e.g. data:text/plain;base64,...)
                    if "," in b64_str and (b64_str.startswith("data:") or "base64" in b64_str[:30]):
                        logger.info("🧹 Removing Data URI header...")
                        b64_str = b64_str.split(",", 1)[1]
                    
                    # Clean whitespaces/newlines
                    b64_str = "".join(b64_str.split())
                    
                    encrypted_stream = base64.b64decode(b64_str)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid Base64 in text file: {str(e)}")
            else:
                logger.info("🖼️ Processing as Container Image")
                try:
                    container_img = Image.open(io.BytesIO(file_content))
                    if container_img.mode != 'RGB':
                        container_img = container_img.convert('RGB')
                    
                    container_bytes = container_img.tobytes()
                    
                    if len(container_bytes) < 4:
                        raise HTTPException(status_code=400, detail="Invalid image format")
                        
                    data_len = int.from_bytes(container_bytes[:4], 'big')
                    if data_len > len(container_bytes) - 4:
                        raise HTTPException(status_code=400, detail="Corrupted data length header")
                        
                    encrypted_stream = container_bytes[4 : 4 + data_len]
                except Exception as e:
                     raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

            # Decrypt Logic (Common)
            def _decrypt_sync(data, aes_obj):
                return aes_obj.decrypt_bytes_cbc(data)
            
            plaintext_compressed = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                _decrypt_sync,
                encrypted_stream, aes
            )
            
            # Try Decompression
            try:
                dctx = zstd.ZstdDecompressor()
                plaintext = dctx.decompress(plaintext_compressed)
            except zstd.ZstdError:
                logger.info("⚠️ Decompression failed, assuming raw data...")
                plaintext = plaintext_compressed
                
            if len(plaintext) < 8:
                raise HTTPException(status_code=400, detail="Invalid decrypted data structure")
                
            width = int.from_bytes(plaintext[:4], 'big')
            height = int.from_bytes(plaintext[4:8], 'big')
            pixel_data = plaintext[8:]
            
            expected_bytes = width * height * 3
            if len(pixel_data) < expected_bytes:
                 raise HTTPException(status_code=400, detail=f"Pixel data mismatch. Expected {expected_bytes}, got {len(pixel_data)}")

            dec_img = Image.frombytes('RGB', (width, height), pixel_data[:expected_bytes])
            
            output = io.BytesIO()
            dec_img.save(output, format="PNG")
            output.seek(0)
            
            return Response(
                content=output.getvalue(),
                media_type="image/png",
                headers={
                    "Content-Disposition": 'attachment; filename="decrypted_image.png"'
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"❌ FATAL ERROR in decrypt_image: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)

@app.get("/random-affine")
def get_random_affine():
    while True:
        mat = np.random.randint(0, 2, (8, 8), dtype=int)
        det = int(np.round(np.linalg.det(mat))) % 2
        if det == 1:
            break
            
    const_vec = np.random.randint(0, 2, 8, dtype=int).tolist()
    return {"matrix": mat.tolist(), "constant": const_vec}

@app.post("/export-excel")
async def export_excel(data: dict):
    sbox = data.get('sbox')
    metrics = data.get('metrics')
    
    df_sbox = pd.DataFrame([sbox[i:i+16] for i in range(0, 256, 16)])
    df_sbox = df_sbox.applymap(lambda x: f"{x:02X}")
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_sbox.to_excel(writer, sheet_name='S-Box', header=False, index=False)
        if metrics:
            df_metrics = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            
    output.seek(0)
    
    from fastapi.responses import StreamingResponse
    headers = {
        'Content-Disposition': 'attachment; filename="sbox_analysis.xlsx"'
    }
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')