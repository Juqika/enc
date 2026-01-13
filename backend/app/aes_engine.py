import os

class AES:
    def __init__(self, key_str: str, sbox: list, poly: int = 0x11B):
        # Handle key padding/truncation (UTF-8 string -> 16 bytes)
        key_bytes = key_str.encode('utf-8')
        if len(key_bytes) < 16:
            # Pad with zeros
            key_bytes = key_bytes + b'\0' * (16 - len(key_bytes))
        elif len(key_bytes) > 16:
            # Truncate
            key_bytes = key_bytes[:16]
            
        self.key = list(key_bytes)
        
        self.sbox = sbox
        self.inv_sbox = self._generate_inv_sbox(sbox)
        self.poly = poly # Irreducible polynomial for MixColumns (usually 0x11B)
        
        self.nb = 4
        self.nk = 4
        self.nr = 10
        
        self.w = self._key_expansion(self.key)

    def _generate_inv_sbox(self, sbox):
        inv_sbox = [0] * 256
        for i, val in enumerate(sbox):
            inv_sbox[val] = i
        return inv_sbox

    def _rot_word(self, word):
        return word[1:] + word[:1]

    def _sub_word(self, word):
        return [self.sbox[b] for b in word]

    def _key_expansion(self, key):
        w = [0] * (self.nb * (self.nr + 1))
        for i in range(self.nk):
            w[i] = [key[4*i], key[4*i+1], key[4*i+2], key[4*i+3]]
            
        rcon = [
            [0x01, 0, 0, 0], [0x02, 0, 0, 0], [0x04, 0, 0, 0], [0x08, 0, 0, 0],
            [0x10, 0, 0, 0], [0x20, 0, 0, 0], [0x40, 0, 0, 0], [0x80, 0, 0, 0],
            [0x1b, 0, 0, 0], [0x36, 0, 0, 0]
        ]
        
        for i in range(self.nk, self.nb * (self.nr + 1)):
            temp = w[i-1]
            if i % self.nk == 0:
                temp = [a ^ b for a, b in zip(self._sub_word(self._rot_word(temp)), rcon[i//self.nk - 1])]
            w[i] = [a ^ b for a, b in zip(w[i-self.nk], temp)]
        return w

    def _add_round_key(self, state, round_idx):
        for c in range(self.nb):
            for r in range(4):
                state[r][c] ^= self.w[round_idx * self.nb + c][r]
        return state

    def _sub_bytes(self, state):
        for r in range(4):
            for c in range(self.nb):
                state[r][c] = self.sbox[state[r][c]]
        return state

    def _inv_sub_bytes(self, state):
        for r in range(4):
            for c in range(self.nb):
                state[r][c] = self.inv_sbox[state[r][c]]
        return state

    def _shift_rows(self, state):
        state[1] = state[1][1:] + state[1][:1]
        state[2] = state[2][2:] + state[2][:2]
        state[3] = state[3][3:] + state[3][:3]
        return state

    def _inv_shift_rows(self, state):
        state[1] = state[1][-1:] + state[1][:-1]
        state[2] = state[2][-2:] + state[2][:-2]
        state[3] = state[3][-3:] + state[3][:-3]
        return state

    def _mix_columns(self, state):
        for c in range(self.nb):
            col = [state[r][c] for r in range(4)]
            state[0][c] = self._galois_mult(col[0], 2) ^ self._galois_mult(col[1], 3) ^ col[2] ^ col[3]
            state[1][c] = col[0] ^ self._galois_mult(col[1], 2) ^ self._galois_mult(col[2], 3) ^ col[3]
            state[2][c] = col[0] ^ col[1] ^ self._galois_mult(col[2], 2) ^ self._galois_mult(col[3], 3)
            state[3][c] = self._galois_mult(col[0], 3) ^ col[1] ^ col[2] ^ self._galois_mult(col[3], 2)
        return state

    def _inv_mix_columns(self, state):
        for c in range(self.nb):
            col = [state[r][c] for r in range(4)]
            state[0][c] = self._galois_mult(col[0], 0x0e) ^ self._galois_mult(col[1], 0x0b) ^ self._galois_mult(col[2], 0x0d) ^ self._galois_mult(col[3], 0x09)
            state[1][c] = self._galois_mult(col[0], 0x09) ^ self._galois_mult(col[1], 0x0e) ^ self._galois_mult(col[2], 0x0b) ^ self._galois_mult(col[3], 0x0d)
            state[2][c] = self._galois_mult(col[0], 0x0d) ^ self._galois_mult(col[1], 0x09) ^ self._galois_mult(col[2], 0x0e) ^ self._galois_mult(col[3], 0x0b)
            state[3][c] = self._galois_mult(col[0], 0x0b) ^ self._galois_mult(col[1], 0x0d) ^ self._galois_mult(col[2], 0x09) ^ self._galois_mult(col[3], 0x0e)
        return state

    def _encrypt_block_core(self, data: list) -> list:
        # Expects 16 bytes list
        state = [[0]*4 for _ in range(4)]
        for i in range(16):
            state[i % 4][i // 4] = data[i]

        self._add_round_key(state, 0)
        
        for round in range(1, self.nr):
            self._sub_bytes(state)
            self._shift_rows(state)
            self._mix_columns(state)
            self._add_round_key(state, round)
            
        self._sub_bytes(state)
        self._shift_rows(state)
        self._add_round_key(state, self.nr)
        
        out_bytes = []
        for i in range(16):
            out_bytes.append(state[i % 4][i // 4])
        return out_bytes

    def _decrypt_block_core(self, data: list) -> list:
        state = [[0]*4 for _ in range(4)]
        for i in range(16):
            state[i % 4][i // 4] = data[i]

        self._add_round_key(state, self.nr)
        
        for round in range(self.nr - 1, 0, -1):
            self._inv_shift_rows(state)
            self._inv_sub_bytes(state)
            self._add_round_key(state, round)
            self._inv_mix_columns(state)
            
        self._inv_shift_rows(state)
        self._inv_sub_bytes(state)
        self._add_round_key(state, 0)
        
        out_bytes = []
        for i in range(16):
            out_bytes.append(state[i % 4][i // 4])
        return out_bytes

    def _pkcs7_pad(self, data: bytes) -> bytes:
        pad_len = 16 - (len(data) % 16)
        return data + bytes([pad_len] * pad_len)

    def _pkcs7_unpad(self, data: bytes) -> bytes:
        if not data:
            return b""
        pad_len = data[-1]
        if pad_len > 16 or pad_len == 0:
            raise ValueError("Invalid PKCS7 padding")
        return data[:-pad_len]

    def _galois_mult(self, a, b):
        p = 0
        for i in range(8):
            if b & 1:
                p ^= a
            hi_bit = a & 0x80
            a <<= 1
            if hi_bit:
                a ^= self.poly
            a &= 0xFF # Safety mask to ensure 8-bit range
            b >>= 1
        return p & 0xFF

    def encrypt_bytes_cbc(self, data: bytes) -> bytes:
        data = self._pkcs7_pad(data)
        
        # Generate random IV
        iv = [int(b) for b in os.urandom(16)]
        
        ciphertext_blocks = []
        prev_block = iv
        
        for i in range(0, len(data), 16):
            chunk = list(data[i:i+16])
            xored = [a ^ b for a, b in zip(chunk, prev_block)]
            encrypted = self._encrypt_block_core(xored)
            ciphertext_blocks.extend(encrypted)
            prev_block = encrypted
            
        return bytes(iv + ciphertext_blocks)

    def decrypt_bytes_cbc(self, full_data: bytes) -> bytes:
        if len(full_data) < 32:
            raise ValueError("Data too short")
            
        iv = list(full_data[:16])
        ciphertext = full_data[16:]
        
        if len(ciphertext) % 16 != 0:
            raise ValueError("Ciphertext length error")
            
        plaintext_bytes = []
        prev_block = iv
        
        for i in range(0, len(ciphertext), 16):
            chunk = list(ciphertext[i:i+16])
            decrypted = self._decrypt_block_core(chunk)
            xored = [a ^ b for a, b in zip(decrypted, prev_block)]
            plaintext_bytes.extend(xored)
            prev_block = chunk
            
        return self._pkcs7_unpad(bytes(plaintext_bytes))

    def encrypt_cbc(self, plaintext: str) -> str:
        data = plaintext.encode('utf-8')
        return self.encrypt_bytes_cbc(data).hex()

    def decrypt_cbc(self, ciphertext_hex: str) -> str:
        try:
            full_data = bytes.fromhex(ciphertext_hex)
        except ValueError:
             raise ValueError("Invalid Hex String")
        
        decrypted = self.decrypt_bytes_cbc(full_data)
        try:
            return decrypted.decode('utf-8')
        except:
            return decrypted.hex()