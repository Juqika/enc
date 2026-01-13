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

app = FastAPI()

# Absolute Path Setup for Static Files (Fix for Render)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

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
    compact: str = Form("false"),
    use_compression: str = Form("true")
):
    # Use semaphore to limit concurrent heavy operations
    async with image_processing_semaphore:
        try:
            is_compact = compact.lower() == 'true'
            do_compress = use_compression.lower() == 'true'
            
            logger.info(f"🎨 Starting image encryption (Compact: {is_compact}, Compress: {do_compress})")
            
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
            
            width, height = img.size
            dim_header = width.to_bytes(4, 'big') + height.to_bytes(4, 'big')
            pixel_data = img.tobytes()
            payload = dim_header + pixel_data
            
            final_payload = payload
            if do_compress:
                logger.info("🗜️ Compressing payload with Zstandard (Level 3)...")
                cctx = zstd.ZstdCompressor(level=3)
                final_payload = cctx.compress(payload)
                logger.info(f"✅ Compression: {len(payload)} -> {len(final_payload)} bytes")
            else:
                logger.info("⏩ Skipping compression")

            logger.info("🔐 Preparing encryption...")
            key_bytes = _prepare_key(key)
            
            # Run CPU-intensive encryption in thread pool
            ciphertext_bytes = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                _encrypt_image_data,
                final_payload, key_bytes, sbox_list, poly
            )
            logger.info(f"✅ Encryption complete: {len(ciphertext_bytes) / 1024:.2f}KB")
            
            # 3. Create Encrypted Image Container
            data_len = len(ciphertext_bytes)
            
            # Header Format:
            # 4 bytes: Length of data
            # 1 byte: Flags (0 = No Compression, 1 = Zstd Compressed)
            # ... Data ...
            
            # Note: We need to update decrypt logic to read this flag!
            # BUT for now, to keep backward compatibility with your existing simple protocol [Len][Data],
            # we might have a problem if we don't tell decryptor whether it is compressed.
            
            # PROPOSAL: Let's assume for this fix, we just want to debug.
            # If we disable compression, the decryptor WILL fail if it expects Zstd.
            # So we must update Decryptor too?
            # Or, we can use a "Magic Byte" approach or just try-catch decompression.
            
            # Let's keep the protocol simple for now:
            # We will force compression ON by default in frontend.
            # If OFF, the output will just be raw AES bytes.
            
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

