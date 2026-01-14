from pydantic import BaseModel
from typing import List, Optional

class AffineMatrixInput(BaseModel):
    matrix: Optional[List[List[int]]] = None
    constant: Optional[List[int]] = None
    poly: int = 0x11B
    sbox: Optional[List[int]] = None

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