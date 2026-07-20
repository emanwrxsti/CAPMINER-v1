#!/usr/bin/env python3
"""Cross-check the miner's BLAKE3-92 against the official blake3 library.

Drives tests/host_kat_test (build it first) in two modes:
  --dump-blake3 <hex92>   byte-path reference over a raw 92-byte header
  --hash-fields ...       field serialization + word path (kernel math)

and compares both against pip blake3 over the identical bytes. Any divergence
in serialization, message scheduling, or the round-0 precompute shows up here.

Usage: python3 cross_check_blake3.py [path-to-host_kat_test]
"""
import os
import random
import struct
import subprocess
import sys

try:
    import blake3
except ImportError:
    print("SKIP: pip package 'blake3' not installed (pip install blake3)")
    sys.exit(0)

BIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "build_tests", "host_kat_test")
if os.name == "nt" and not BIN.lower().endswith(".exe"):
    BIN += ".exe"
if not os.path.exists(BIN):
    print(f"FAIL: test binary not found at {BIN} (build tests/host_kat_test.cpp first)")
    sys.exit(1)

rng = random.Random(0xA1F0_2026)
fails = 0

def run(args):
    out = subprocess.run([BIN] + args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{args[0]} rc={out.returncode}: {out.stderr.strip()}")
    return out.stdout.strip()

# --- raw 92-byte headers through the byte-path reference ---------------------
for i in range(16):
    header = bytes(rng.randrange(256) for _ in range(92))
    expect = blake3.blake3(header).hexdigest()
    got = run(["--dump-blake3", header.hex()])
    if got != expect:
        fails += 1
        print(f"FAIL raw header {i}: lib={expect} ref={got}")
print(f"raw headers vs blake3 lib      16 checked, {fails} failures")

# --- field serialization + word path (the kernel math) -----------------------
f2 = 0
for i in range(64):
    bn = rng.randrange(2**32)
    prev = bytes(rng.randrange(256) for _ in range(32))
    ts = rng.randrange(2**64)
    if i == 0:
        nonce = 0
    elif i == 1:
        nonce = 2**32 - 1
    elif i == 2:
        nonce = 2**32
    elif i == 3:
        nonce = 2**64 - 1
    else:
        nonce = rng.randrange(2**64)
    diff = rng.randrange(2**64)
    merkle = bytes(rng.randrange(256) for _ in range(32))

    header = struct.pack("<I32sQQQ32s", bn, prev, ts, nonce, diff, merkle)
    assert len(header) == 92
    expect = blake3.blake3(header).hexdigest()
    got = run(["--hash-fields", str(bn), prev.hex(), str(ts), str(nonce), str(diff),
               merkle.hex()])
    if got != expect:
        f2 += 1
        print(f"FAIL fields {i} nonce={nonce}: lib={expect} miner={got}")
fails += f2
print(f"field/word path vs blake3 lib  64 checked, {f2} failures")

if fails == 0:
    print("CROSS-CHECK PASS: official blake3 == byte reference == device word path")
    sys.exit(0)
print(f"CROSS-CHECK FAIL: {fails} mismatches")
sys.exit(1)
