from pydantic import BaseModel
from typing import List

class AffineMatrixInput(BaseModel):
    matrix: List[List[int]] # 8x8
    constant: List[int] # 8 elements
    poly: int = 0x11B # Irreducible polynomial

class AnalysisResult(BaseModel):
    sbox: List[int]
    is_bijective: bool
    is_balanced: bool
    metrics: dict
    comparison: dict

class ExcelExportRequest(BaseModel):
    sbox: List[int]
    metrics: dict

class EncryptionRequest(BaseModel):
    plaintext: str
    key: str
    sbox: List[int]

class DecryptionRequest(BaseModel):
    ciphertext: str
    key: str
    sbox: List[int]
