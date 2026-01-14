import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
from collections import Counter

class SBoxMath:
    def __init__(self):
        # Default Irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)
        self.DEFAULT_POLY = 0x11B
        # Paper Irreducible polynomial: x^8 + x^7 + x^6 + x^5 + x^4 + x + 1 (0x1F3)
        self.PAPER_POLY = 0x1F3
        self.IRREDUCIBLE_POLY = self.DEFAULT_POLY
        
        self.AES_SBOX = self._get_aes_sbox()
        self.SBOX_44 = self._get_sbox_44() # From PDF
        
        # Paper Affine Matrices (from 42305-112406-1-PB.pdf)
        self.PAPER_A0 = [
            [1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 1]
        ]
        
        self.PAPER_A1 = [
            [0, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 0]
        ]
        
        self.PAPER_A2 = [
            [1, 1, 1, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 0, 0, 1, 1]
        ]
        
        # New Paper Matrices (s11071-024-10414-3.pdf)
        # K44 - For S-box44 (Optimized)
        self.PAPER2_K44 = [
            [0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 0]
        ]
        
        # K81 - For S-box81
        self.PAPER2_K81 = [
            [1, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 1]
        ]
        
        # K111 - For S-box111
        self.PAPER2_K111 = [
            [1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 0, 0, 1]
        ]
        
        self.AES_CONSTANT = [1, 1, 0, 0, 0, 1, 1, 0] # 0x63

    def set_irreducible_poly(self, poly_hex: int):
        self.IRREDUCIBLE_POLY = poly_hex

    def _get_aes_sbox(self) -> List[int]:
        return [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        ]

    def _get_sbox_44(self) -> List[int]:
        # Values extracted from Table 16 of the PDF
        return [
            99, 205, 85, 71, 25, 127, 113, 219, 63, 244, 109, 159, 11, 228, 94, 214,
            77, 177, 201, 78, 5, 48, 29, 30, 87, 96, 193, 80, 156, 200, 216, 86,
            116, 143, 10, 14, 54, 169, 148, 68, 49, 75, 171, 157, 92, 114, 188, 194,
            121, 220, 131, 210, 83, 135, 250, 149, 253, 72, 182, 33, 190, 141, 249, 82,
            232, 50, 21, 84, 215, 242, 180, 198, 168, 167, 103, 122, 152, 162, 145, 184,
            43, 237, 119, 183, 7, 12, 125, 55, 252, 206, 235, 160, 140, 133, 179, 192,
            110, 176, 221, 134, 19, 6, 187, 59, 26, 129, 112, 73, 175, 45, 24, 218,
            44, 66, 151, 32, 137, 31, 35, 147, 236, 247, 117, 132, 79, 136, 154, 105,
            199, 101, 203, 52, 57, 4, 153, 197, 88, 76, 202, 174, 233, 62, 208, 91,
            231, 53, 1, 124, 0, 28, 142, 170, 158, 51, 226, 65, 123, 186, 239, 246,
            38, 56, 36, 108, 8, 126, 9, 189, 81, 234, 212, 224, 13, 3, 40, 64,
            172, 74, 181, 118, 39, 227, 130, 89, 245, 166, 16, 61, 106, 196, 211, 107,
            229, 195, 138, 18, 93, 207, 240, 95, 58, 255, 209, 217, 15, 111, 46, 173,
            223, 42, 115, 238, 139, 243, 23, 98, 100, 178, 37, 97, 191, 213, 222, 155,
            165, 2, 146, 204, 120, 241, 163, 128, 22, 90, 60, 185, 67, 34, 27, 248,
            164, 69, 41, 230, 104, 47, 144, 251, 20, 17, 150, 225, 254, 161, 102, 70
        ]

    # --- IMAGE ANALYSIS METRICS ---
    def calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate image entropy (randomness)"""
        counts = Counter(data.flatten())
        total = data.size
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        return float(entropy)

    def calculate_npcr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Number of Pixels Change Rate"""
        # Resize img2 to match img1 if shapes differ (Standardization for metric)
        if img1.shape != img2.shape:
            try:
                from PIL import Image
                i2 = Image.fromarray(img2)
                i2 = i2.resize((img1.shape[1], img1.shape[0]), resample=Image.NEAREST)
                img2 = np.array(i2)
            except:
                return 0.0

        diff = np.where(img1 != img2, 1, 0)
        npcr = np.sum(diff) / img1.size * 100
        return float(npcr)

    def calculate_uaci(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Unified Average Changing Intensity"""
        if img1.shape != img2.shape:
            try:
                from PIL import Image
                i2 = Image.fromarray(img2)
                i2 = i2.resize((img1.shape[1], img1.shape[0]), resample=Image.NEAREST)
                img2 = np.array(i2)
            except:
                return 0.0
        
        diff = np.abs(img1.astype(int) - img2.astype(int))
        uaci = np.mean(diff) / 255.0 * 100
        return float(uaci)

    def calculate_correlation(self, img: np.ndarray) -> Dict[str, float]:
        """Correlation Coefficient (Horizontal, Vertical, Diagonal)"""
        # Uses random sampling of 3000 pairs for performance
        N_SAMPLES = 3000
        results = {}
        
        if len(img.shape) == 3:
            flat = np.mean(img, axis=2)
        else:
            flat = img.astype(float)
            
        H, W = flat.shape
        
        directions = {
            'Horizontal': (0, 1),
            'Vertical': (1, 0),
            'Diagonal': (1, 1)
        }
        
        for name, (dy, dx) in directions.items():
            if H <= dy or W <= dx:
                results[name] = 0.0
                continue
            
            x_vals = []
            y_vals = []
            
            for _ in range(N_SAMPLES):
                r = np.random.randint(0, H - dy)
                c = np.random.randint(0, W - dx)
                x_vals.append(flat[r, c])
                y_vals.append(flat[r+dy, c+dx])
                
            try:
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
                results[name] = float(corr) if not np.isnan(corr) else 0.0
            except:
                results[name] = 0.0
                
        return results

    def get_histogram(self, img_array: np.ndarray) -> Dict[str, List[int]]:
        """Calculate histogram for each channel"""
        if len(img_array.shape) == 2: # Grayscale
            hist = np.histogram(img_array, bins=256, range=(0, 255))[0]
            return {"gray": hist.tolist()}
        else: # RGB
            channels = ["red", "green", "blue"]
            result = {}
            for i, char in enumerate(channels):
                hist = np.histogram(img_array[:, :, i], bins=256, range=(0, 255))[0]
                result[char] = hist.tolist()
            return result

    def gf_mult(self, a, b):
        """Galois Field multiplication in GF(2^8)"""
        p = 0
        for i in range(8):
            if (b & 1) == 1:
                p ^= a
            hi_bit_set = (a & 0x80)
            a <<= 1
            if hi_bit_set:
                a ^= self.IRREDUCIBLE_POLY
            b >>= 1
        return p & 0xFF

    def gf_inverse(self, a):
        """Multiplicative inverse in GF(2^8)."""
        if a == 0:
            return 0
        # Extended Euclidean Algorithm could be used, 
        # but brute force for 256 values is essentially instant.
        for i in range(256):
            if self.gf_mult(a, i) == 1:
                return i
        return 0

    def generate_sbox(self, affine_matrix: List[List[int]], constant_vector: List[int]) -> List[int]:
        """
        Generates S-Box using S(x) = M * x^(-1) + C
        affine_matrix: 8x8 list of 0/1
        constant_vector: list of 8 0/1 (LSB at index 0 or MSB? Usually vector is column. 
                        Let's assume index 0 is LSB, index 7 is MSB to match bit operations)
        """
        sbox = []
        
        # Precompute inverses
        inverses = [self.gf_inverse(x) for x in range(256)]
        
        # Constant to integer
        c_val = 0
        for i, bit in enumerate(constant_vector):
            if bit:
                c_val |= (1 << i)

        for i in range(256):
            inv = inverses[i]
            # Matrix Multiplication over GF(2)
            # transform inv (byte) to vector (bits)
            inv_bits = [(inv >> k) & 1 for k in range(8)]
            
            # M * inv
            res_bits = [0] * 8
            for row in range(8):
                val = 0
                for col in range(8):
                    val ^= (affine_matrix[row][col] * inv_bits[col])
                res_bits[row] = val
            
            # Bits to int
            res_val = 0
            for k in range(8):
                if res_bits[k]:
                    res_val |= (1 << k)
            
            # XOR Constant
            sbox_val = res_val ^ c_val
            sbox.append(sbox_val)
            
        return sbox

    def check_bijective(self, sbox: List[int]) -> bool:
        return len(set(sbox)) == 256 and max(sbox) == 255 and min(sbox) == 0

    def check_balance_bits(self, sbox: List[int]) -> bool:
        """
        Check if each output bit is balanced (has 128 zeros and 128 ones).
        """
        for i in range(8):
            count_ones = sum((val >> i) & 1 for val in sbox)
            if count_ones != 128:
                return False
        return True

    # --- METRICS IMPLEMENTATION ---
    
    def _walsh_transform(self, truth_table):
        fwht = np.array(truth_table)
        h = 1
        while h < 256:
            for j in range(0, 256, h * 2):
                for k in range(j, j + h):
                    x = fwht[k]
                    y = fwht[k + h]
                    fwht[k] = x + y
                    fwht[k + h] = x - y
            h *= 2
        return fwht

    def calculate_nl(self, sbox):
        """Nonlinearity"""
        nl_values = []
        for i in range(8): # For each output bit function
            truth_table = []
            for x in range(256):
                val = (sbox[x] >> i) & 1
                truth_table.append(1 if val == 0 else -1)
            
            fwht = self._walsh_transform(truth_table)
            max_walsh = np.max(np.abs(fwht))
            nl_values.append(128 - max_walsh / 2)
        
        return int(min(nl_values))

    def calculate_sac(self, sbox):
        """Strict Avalanche Criterion"""
        sac_matrix = np.zeros((8, 8))
        for i in range(8): # input bit flipped
            for j in range(8): # output bit changed
                count = 0
                for x in range(256):
                    x_flip = x ^ (1 << i)
                    y1 = sbox[x]
                    y2 = sbox[x_flip]
                    if ((y1 >> j) & 1) != ((y2 >> j) & 1):
                        count += 1
                sac_matrix[i][j] = count / 256.0
        return float(np.mean(sac_matrix))

    def calculate_bic(self, sbox):
        """Bit Independence Criterion (BIC-NL and BIC-SAC)"""
        # BIC-NL: Min NL of XOR sum of all pairs of output bits
        bic_nl_vals = []
        # BIC-SAC: Mean SAC of XOR sum of all pairs of output bits
        bic_sac_vals = []

        for j in range(8):
            for k in range(j + 1, 8):
                # Function f = bit_j XOR bit_k
                truth_table_nl = []
                # For SAC of (bit_j XOR bit_k)
                sac_sum = 0
                
                # Precompute XOR sum bits
                xor_bits = [( (sbox[x]>>j)&1 ) ^ ( (sbox[x]>>k)&1 ) for x in range(256)]

                # BIC-NL part
                truth_table_nl = [1 if b == 0 else -1 for b in xor_bits]
                fwht = self._walsh_transform(truth_table_nl)
                max_walsh = np.max(np.abs(fwht))
                bic_nl_vals.append(128 - max_walsh / 2)

                # BIC-SAC part
                # Check avalanche of the combined function 'xor_bits'
                for i_in in range(8): # Input bit flip
                    flip_diff_count = 0
                    for x in range(256):
                        x_flip = x ^ (1 << i_in)
                        val_orig = xor_bits[x]
                        val_flip = xor_bits[x_flip]
                        if val_orig != val_flip:
                            flip_diff_count += 1
                    sac_sum += (flip_diff_count / 256.0)
                bic_sac_vals.append(sac_sum / 8.0) # Average over 8 input bits

        return float(min(bic_nl_vals)), float(np.mean(bic_sac_vals))

    def calculate_lap(self, sbox):
        """Linear Approximation Probability"""
        max_bias = 0
        for b in range(1, 256):
            # F_b(x) = b dot S(x)
            truth_table = []
            for x in range(256):
                # Dot product parity
                y = sbox[x]
                dot = 0
                temp = y & b
                while temp:
                    dot ^= (temp & 1)
                    temp >>= 1
                truth_table.append(1 if dot == 0 else -1)
            
            fwht = self._walsh_transform(truth_table)
            
            # exclude a=0 for b!=0
            lat_row = np.abs(fwht)
            current_max = np.max(lat_row[1:]) 
            if current_max > max_bias:
                max_bias = current_max
        
        # LAP = max_bias / 2^n ? No, prob deviation.
        # Paper S-box LAP is 0.0625 = 1/16.
        # Max bias (Walsh) for AES is 32. 32/256 = 0.125.
        # 0.0625 is 0.125 / 2.
        return max_bias / 512.0

    def calculate_dap(self, sbox):
        """Differential Approximation Probability"""
        ddt = np.zeros((256, 256), dtype=int)
        for x in range(256):
            for dx in range(1, 256): # input diff
                dy = sbox[x] ^ sbox[x ^ dx]
                ddt[dx][dy] += 1
        
        max_diff = np.max(ddt)
        return max_diff / 256.0

    def calculate_du(self, sbox):
        """Differential Uniformity"""
        ddt = np.zeros((256, 256), dtype=int)
        for x in range(256):
            for dx in range(1, 256):
                dy = sbox[x] ^ sbox[x ^ dx]
                ddt[dx][dy] += 1
        return int(np.max(ddt))

    def calculate_ad(self, sbox):
        """Algebraic Degree"""
        degrees = []
        for i in range(8):
            # Truth table for i-th output bit
            tt = [(sbox[x] >> i) & 1 for x in range(256)]
            
            # Mobius Transform over GF(2)
            h = 1
            while h < 256:
                for j in range(0, 256, h * 2):
                    for k in range(j, j + h):
                        tt[k + h] = tt[k + h] ^ tt[k]
                h *= 2
            
            max_deg = 0
            for x in range(256):
                if tt[x] == 1:
                    deg = bin(x).count('1')
                    if deg > max_deg:
                        max_deg = deg
            degrees.append(max_deg)
            
        return max(degrees)

    def calculate_to(self, sbox):
        """Transparency Order (Standard Definition)"""
        # TO = Max over beta!=0 of ( | n - 2*wt(beta) - 1/(2^(2n-1)) * sum_{a!=0} (-1)^wt(a) |Walsh(a, beta)| | )
        # n=8.
        # This is complex. Using a simplified interpretation often found in lightweight tools:
        # TO = (1/256) * (Max(|Walsh(a,b)|) - Avg(|Walsh(a,b)|)) ? No.
        # Let's implementation the formal definition roughly.
        # Cost is O(2^n * 2^n) = 65k ops. Fast enough.
        
        n = 8
        max_val = 0
        
        # Precompute Walsh spectrum for all beta
        # We need Walsh(a, beta) for all a, beta.
        # W(a, beta) = sum (-1)^(a.x + beta.S(x))
        # This is exactly LAT[beta][a] (or transposed).
        # We can reuse logic from LAP calculation but we need ALL values.
        
        # This loop is 256 * 256 * 256 ops (too slow, ~16M).
        # We must use FWHT.
        
        # Calculate full LAT table using FWHT: 256 * (256 log 256) ~ 256 * 2048 = 0.5M ops. Fast.
        lat_abs = np.zeros((256, 256))
        
        for beta in range(1, 256): # beta != 0
            truth_table = []
            beta_weight = bin(beta).count('1')
            
            for x in range(256):
                # beta dot S(x)
                y = sbox[x]
                dot = 0
                temp = y & beta
                while temp:
                    dot ^= (temp & 1)
                    temp >>= 1
                truth_table.append(1 if dot == 0 else -1)
            
            fwht = self._walsh_transform(truth_table)
            # fwht[alpha] is Walsh(alpha, beta)
            
            # Formula part: sum_{alpha!=0} (-1)^wt(alpha) * |Walsh(alpha, beta)|
            sum_term = 0
            for alpha in range(1, 256): # alpha != 0
                w_val = abs(fwht[alpha])
                a_weight = bin(alpha).count('1')
                if a_weight % 2 == 1:
                    sum_term -= w_val
                else:
                    sum_term += w_val
            
            # | n - 2*wt(beta) - (1 / 2^(2n-1)) * sum_term |
            # 2^(2n-1) = 2^15 = 32768
            term = abs(n - 2*beta_weight - (sum_term / 32768.0))
            if term > max_val:
                max_val = term
                
        return max_val

    def calculate_ci(self, sbox):
        """Correlation Immunity"""
        ci_values = []
        for i in range(8):
            truth_table = []
            for x in range(256):
                val = (sbox[x] >> i) & 1
                truth_table.append(1 if val == 0 else -1)
            
            fwht = self._walsh_transform(truth_table)
            
            current_ci = 0
            for k in range(1, 9):
                is_zero = True
                for w in range(1, 256):
                    if bin(w).count('1') == k:
                        if fwht[w] != 0:
                            is_zero = False
                            break
                if is_zero:
                    current_ci = k
                else:
                    break
            ci_values.append(current_ci)
        return min(ci_values)

sbox_math = SBoxMath()
