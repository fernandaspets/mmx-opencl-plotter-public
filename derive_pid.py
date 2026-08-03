#!/usr/bin/env python3
"""Derive MMX plot ID from seed using the reference method.
PID = SHA256("MMX/PLOTID/OG" || ksize_byte || seed(32) || farmer_key(48))
"""
import hashlib
import sys
import os
import secrets

FARMER_KEY_HEX = "02292cd11aa18e5f64344cbe6c580249364dfe5a3683adc25446aadcc1b38555d7"

def derive_pid(seed: bytes, ksize: int, farmer_key_hex: str = FARMER_KEY_HEX) -> str:
    farmer_key = bytes.fromhex(farmer_key_hex)
    tag = b"MMX/PLOTID/OG"
    buf = tag + bytes([ksize]) + seed + farmer_key
    return hashlib.sha256(buf).hexdigest().upper()

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for i in range(n):
        seed = secrets.token_bytes(32)
        k18_pid = derive_pid(seed, 18)
        k25_pid = derive_pid(seed, 25)
        print(f"{seed.hex()},{k18_pid},{k25_pid}")

if __name__ == "__main__":
    main()
