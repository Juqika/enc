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

# Configure logging
logger = logging.getLogger("uvicorn")

app = FastAPI()

# Global resources for heavy image operations
MAX_CONCURRENT_IMAGE_OPS = 2
image_processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_OPS)
thread_pool = ThreadPoolExecutor(max_workers=4)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Analysis", "Content-Disposition", "X-Sbox-Type", "X-Encryption-Time", "X-Histogram-Data"]
)

# --- Helper Functions for Image Processing (from contoh.txt) ---

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
    return {
        "original_entropy": entropy_orig,
        "encrypted_entropy": entropy_enc,
        "npcr": npcr
    }

# Serve static files for frontend (Vue.js via CDN)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

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
    # Set the polynomial for inversion
    sbox_math.set_irreducible_poly(input_data.poly)
    
    # 1. Generate S-Box
    sbox = sbox_math.generate_sbox(input_data.matrix, input_data.constant)
    
    is_bijective = sbox_math.check_bijective(sbox)
    is_balanced = sbox_math.check_balance_bits(sbox)
    
    metrics = {}
    
    if is_bijective:
        # Calculate Metrics
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
    sbox: str = Form(...), # JSON string of list of 256 ints
    poly: int = Form(0x11B),
    compact: str = Form("false") # Received as string from FormData
):
    # Use semaphore to limit concurrent heavy operations
    async with image_processing_semaphore:
        try:
            is_compact = compact.lower() == 'true'
            logger.info(f"🎨 Starting image encryption (Compact Mode: {is_compact})")
            
            sbox_list = json.loads(sbox)
            if len(sbox_list) != 256:
                raise HTTPException(status_code=400, detail="S-Box must be 256 elements")
            
            start_time = time.time()
            # read upload with limit to avoid OOM/timeouts
            logger.info(f"📥 Reading uploaded file: {file.filename}")
            image_bytes = await _read_upload_file_limited(file)
            
            logger.info("🖼️  Opening image...")
            img = Image.open(io.BytesIO(image_bytes))
            
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # --- COMPACT MODE LOGIC ---
            if is_compact:
                # Resize to max 600px dimension
                max_compact_size = 600
                if max(img.size) > max_compact_size:
                    logger.info(f"📉 Compact Mode: Resizing from {img.size} to max {max_compact_size}px")
                    img.thumbnail((max_compact_size, max_compact_size), Image.Resampling.LANCZOS)
                else:
                    logger.info(f"ℹ️ Compact Mode: Image size {img.size} is already small enough.")
            else:
                # Normal mode logic (just safety cap)
                max_dimension = 8192
                if max(img.size) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            # Calculate histogram with memory-efficient method
            logger.info("📊 Converting to NumPy array...")
            img_array = np.array(img)
            array_size_mb = img_array.nbytes / 1024 / 1024
            logger.info(f"📊 Array shape: {img_array.shape}, size: {array_size_mb:.2f}MB")
            
            if array_size_mb > 200:
                raise HTTPException(status_code=413, detail=f"Image too large to process ({array_size_mb:.1f}MB).")
            
            try:
                logger.info("📈 Calculating original histogram...")
                original_histogram = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, 
                    _process_image_histogram, 
                    img_array
                )
                logger.info("✅ Original histogram calculated")
            except Exception as e:
                logger.error(f"❌ Histogram error: {e}")
                raise
            
            # Store original array for security analysis
            original_array_for_metrics = img_array.copy()
            
            # Clean up array to free memory
            del img_array
            if array_size_mb > 50:
                gc.collect()
            
            logger.info("💾 Converting image to PNG bytes for encryption...")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            raw_image_data = img_buffer.getvalue()
            img_buffer.close()
            
            # Prepare payload for AES (matching existing logic: [Width(4)][Height(4)][Pixels...])
            # Wait, the existing logic used img.tobytes(), not the PNG encoded bytes.
            # Using PNG encoded bytes as the payload is actually more common for "image encryption" 
            # as it preserves the format but encrypts the content.
            # BUT, the current decrypt logic expects [Width][Height][Pixels].
            # Let's stick to the current project's payload format to avoid breaking decryption.
            
            width, height = img.size
            dim_header = width.to_bytes(4, 'big') + height.to_bytes(4, 'big')
            pixel_data = img.tobytes()
            payload = dim_header + pixel_data
            
            # Compress payload to reduce ciphertext length using Zstandard (Zstd)
            # Level 15 provides a strong balance of high compression ratio and speed.
            logger.info("🗜️ Compressing payload with Zstandard (Level 15)...")
            cctx = zstd.ZstdCompressor(level=15)
            compressed_payload = cctx.compress(payload)
            logger.info(f"✅ Compression: {len(payload)} -> {len(compressed_payload)} bytes")
            
            logger.info("🔐 Preparing encryption...")
            key_bytes = _prepare_key(key)
            
            # Run CPU-intensive encryption in thread pool
            ciphertext_bytes = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                _encrypt_image_data,
                compressed_payload, key_bytes, sbox_list, poly
            )
            logger.info(f"✅ Encryption complete: {len(ciphertext_bytes) / 1024:.2f}KB")
            
            # 3. Create Encrypted Image Container
            data_len = len(ciphertext_bytes)
            header = data_len.to_bytes(4, byteorder='big')
            full_encrypted_data = header + ciphertext_bytes
            
            # Calculate size for square container
            pixels_needed = math.ceil(len(full_encrypted_data) / 3)
            side = int(math.ceil(math.sqrt(pixels_needed)))
            total_bytes_needed = side * side * 3
            padded_data = full_encrypted_data + b'\x00' * (total_bytes_needed - len(full_encrypted_data))
            
            enc_img = Image.frombytes('RGB', (side, side), padded_data)
            logger.info(f"📐 Created encrypted image container: {side}x{side}")
            
            # Analysis
            logger.info("📈 Calculating encrypted analysis...")
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
            
            # Re-construct the analysis object expected by the frontend
            analysis = {
                "original_entropy": security_metrics["original_entropy"],
                "encrypted_entropy": security_metrics["encrypted_entropy"],
                "npcr": security_metrics["npcr"],
                "original_histogram": original_histogram,
                "encrypted_histogram": encrypted_histogram
            }
            
            # Cleanup
            del original_array_for_metrics
            del enc_array
            gc.collect()
            
            # 5. Return
            output = io.BytesIO()
            enc_img.save(output, format="PNG")
            output.seek(0)
            
            encryption_time = (time.time() - start_time) * 1000
            
            logger.info("🎉 Image encryption successful!")
            
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
            logger.info("🔓 Starting image decryption...")
            sbox_list = json.loads(sbox)
            aes = AES(key, sbox_list, poly)
            
            image_bytes = await _read_upload_file_limited(file)
            container_img = Image.open(io.BytesIO(image_bytes))
            if container_img.mode != 'RGB':
                container_img = container_img.convert('RGB')
                
            container_bytes = container_img.tobytes()
            
            # Extract Length
            if len(container_bytes) < 4:
                raise HTTPException(status_code=400, detail="Invalid image format")
                
            data_len = int.from_bytes(container_bytes[:4], 'big')
            logger.info(f"📦 Encrypted data length: {data_len} bytes")
            
            if data_len > len(container_bytes) - 4:
                raise HTTPException(status_code=400, detail="Corrupted data length header")
                
            encrypted_stream = container_bytes[4 : 4 + data_len]
            
            # Decrypt
            logger.info("🔑 Decrypting data (async in thread pool)...")
            try:
                def _decrypt_sync(data, aes_obj):
                    return aes_obj.decrypt_bytes_cbc(data)
                
                plaintext_compressed = await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    _decrypt_sync,
                    encrypted_stream, aes
                )
                
                # Decompress
                logger.info("🔓 Decompressing data (Zstandard)...")
                dctx = zstd.ZstdDecompressor()
                plaintext = dctx.decompress(plaintext_compressed)
                
            except Exception as e:
                logger.error(f"❌ Decryption/Decompression failed: {e}")
                raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")
                
            # Parse Plaintext: [Width(4)][Height(4)][Pixels...]
            if len(plaintext) < 8:
                raise HTTPException(status_code=400, detail="Invalid decrypted data structure")
                
            width = int.from_bytes(plaintext[:4], 'big')
            height = int.from_bytes(plaintext[4:8], 'big')
            pixel_data = plaintext[8:]
            
            logger.info(f"📐 Reconstructing image: {width}x{height}")
            expected_bytes = width * height * 3
            if len(pixel_data) < expected_bytes:
                 raise HTTPException(status_code=400, detail=f"Pixel data mismatch. Expected {expected_bytes}, got {len(pixel_data)}")

            # Reconstruct
            dec_img = Image.frombytes('RGB', (width, height), pixel_data[:expected_bytes])
            
            output = io.BytesIO()
            dec_img.save(output, format="PNG")
            output.seek(0)
            
            logger.info("🎉 Image decryption successful!")
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

@app.post("/decrypt-image-text")
async def decrypt_image_text(
    ciphertext: str = Form(...),
    key: str = Form(...),
    sbox: str = Form(...),
    poly: int = Form(0x11B)
):
    async with image_processing_semaphore:
        try:
            logger.info("🔓 Starting image decryption from TEXT...")
            sbox_list = json.loads(sbox)
            aes = AES(key, sbox_list, poly)
            
            # 1. Decode Base64 to get encrypted bytes
            try:
                encrypted_bytes = base64.b64decode(ciphertext)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid Base64 string")

            # 2. Decrypt
            logger.info("🔑 Decrypting data (async)...")
            try:
                def _decrypt_sync(data, aes_obj):
                    return aes_obj.decrypt_bytes_cbc(data)
                
                plaintext_compressed = await asyncio.get_event_loop().run_in_executor(
                    thread_pool,
                    _decrypt_sync,
                    encrypted_bytes, aes
                )
                
                # 3. Decompress
                logger.info("🔓 Decompressing data (Zstandard)...")
                dctx = zstd.ZstdDecompressor()
                plaintext = dctx.decompress(plaintext_compressed)
                
            except Exception as e:
                logger.error(f"❌ Decryption/Decompression failed: {e}")
                raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")
                
            # 4. Parse Plaintext: [Width(4)][Height(4)][Pixels...]
            if len(plaintext) < 8:
                raise HTTPException(status_code=400, detail="Invalid decrypted data structure")
                
            width = int.from_bytes(plaintext[:4], 'big')
            height = int.from_bytes(plaintext[4:8], 'big')
            pixel_data = plaintext[8:]
            
            logger.info(f"📐 Reconstructing image: {width}x{height}")
            expected_bytes = width * height * 3
            if len(pixel_data) < expected_bytes:
                 raise HTTPException(status_code=400, detail=f"Pixel data mismatch. Expected {expected_bytes}, got {len(pixel_data)}")

            # 5. Reconstruct
            dec_img = Image.frombytes('RGB', (width, height), pixel_data[:expected_bytes])
            
            output = io.BytesIO()
            dec_img.save(output, format="PNG")
            output.seek(0)
            
            logger.info("🎉 Image text-decryption successful!")
            return Response(
                content=output.getvalue(),
                media_type="image/png",
                headers={
                    "Content-Disposition": 'attachment; filename="decrypted_image_from_text.png"'
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"❌ FATAL ERROR in decrypt_image_text: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)

@app.get("/random-affine")
def get_random_affine():
    # Loop until we find an invertible matrix (bijective affine transform)
    # The paper implies we explore affine matrices. Only invertible ones create valid S-Boxes (bijective).
    
    while True:
        # Generate random 8x8 matrix
        mat = np.random.randint(0, 2, (8, 8), dtype=int)
        # Check determinant over GF(2)
        det = int(np.round(np.linalg.det(mat))) % 2
        if det == 1:
            break
            
    const_vec = np.random.randint(0, 2, 8, dtype=int).tolist()
    return {"matrix": mat.tolist(), "constant": const_vec}

@app.post("/export-excel")
async def export_excel(data: dict):
    # Data expected: {"sbox": [...], "metrics": {...}}
    sbox = data.get('sbox')
    metrics = data.get('metrics')
    
    df_sbox = pd.DataFrame([sbox[i:i+16] for i in range(0, 256, 16)])
    # Hex formatting
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

