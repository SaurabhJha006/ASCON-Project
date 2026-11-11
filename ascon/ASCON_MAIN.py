from __future__ import annotations
import argparse
import os
from typing import Callable, Dict, List, Sequence, Tuple

MASK64 = 0xFFFFFFFFFFFFFFFF

KEY_BITS = 128
RATE_BYTES = 8
ROUNDS_A = 12
ROUNDS_B = 6

IV_ASCON_128 = (KEY_BITS << 56) | ((RATE_BYTES * 8) << 48) | (ROUNDS_A << 40) | (ROUNDS_B << 32)

ROUND_CONSTANTS = [0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5,
                   0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B]

KAT_KEY = bytes(range(16))
KAT_NONCE = bytes(range(16))
KAT_TAG_EMPTY = bytes.fromhex("E355159F292911F794CB1432A0103A8A")

State = List[int]

def banner(title: str) -> None:
    print()
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)


def sub(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 68 - len(title)))


def show_state(state: Sequence[int], label: str = "") -> None:
    if label:
        print(f"  {label}")
    for i, lane in enumerate(state):
        print(f"    x{i} = {lane:016x}")


def bits_of(x: int) -> str:
    return format(x & MASK64, "064b")


def rotr(x: int, n: int) -> int:
    """Rotate a 64-bit word right by n bits."""
    return ((x >> n) | (x << (64 - n))) & MASK64


def hamming(x: int) -> int:
    return bin(x & MASK64).count("1")

def p_C(state: State, rc: int) -> None:
    """Constant addition: XOR the round constant into x2."""
    state[2] ^= rc


def p_S(state: State) -> None:

    x0, x1, x2, x3, x4 = state
    x0 ^= x4; x4 ^= x3; x2 ^= x1
    t0 = ~x0 & x1; t1 = ~x1 & x2; t2 = ~x2 & x3; t3 = ~x3 & x4; t4 = ~x4 & x0
    x0 ^= t1; x1 ^= t2; x2 ^= t3; x3 ^= t4; x4 ^= t0
    x1 ^= x0; x0 ^= x4; x3 ^= x2; x2 = ~x2
    state[:] = [v & MASK64 for v in (x0, x1, x2, x3, x4)]


def p_L(state: State) -> None:
    """Linear diffusion layer: each lane mixed with two rotations of itself."""
    for i, (r1, r2) in enumerate(ROTATIONS):
        state[i] ^= rotr(state[i], r1) ^ rotr(state[i], r2)


def ascon_round(state: State, rc: int) -> None:
    p_C(state, rc)
    p_S(state)
    p_L(state)


def permutation(state: State, rounds: int) -> None:
    """p^rounds, using the last `rounds` constants of the 12-round schedule."""
    for rc in ROUND_CONSTANTS[12 - rounds:]:
        ascon_round(state, rc)

def get_column(state: Sequence[int], col: int) -> int:
    """Read the 5-bit column at bit position `col` (x0 is the MSB)."""
    v = 0
    for lane in state:
        v = (v << 1) | ((lane >> col) & 1)
    return v


def put_column(state: State, col: int, value: int) -> None:
    for i in range(5):
        bit = (value >> (4 - i)) & 1
        state[i] = (state[i] & ~(1 << col)) | (bit << col)


def derive_sbox_table() -> List[int]:
    """Evaluate the bitsliced p_S on every 5-bit input to recover the table."""
    table = []
    for v in range(32):
        s = [0] * 5
        put_column(s, 0, v)
        p_S(s)
        table.append(get_column(s, 0))
    return table

def to_lanes(b: bytes) -> List[int]:
    return [int.from_bytes(b[i:i + 8], "big") for i in range(0, len(b), 8)]


def pad(data: bytes) -> bytes:
    """Append 0x80 then zeros up to a multiple of the rate (8 bytes)."""
    n = RATE_BYTES - (len(data) % RATE_BYTES) - 1
    return data + b"\x80" + b"\x00" * n


def initialise(key: bytes, nonce: bytes, trace: bool = False) -> State:
    k0, k1 = to_lanes(key)
    n0, n1 = to_lanes(nonce)
    state = [IV_ASCON_128, k0, k1, n0, n1]
    if trace:
        show_state(state, "S = IV || K || N")
    permutation(state, ROUNDS_A)
    if trace:
        show_state(state, f"after p^{ROUNDS_A}")
    state[3] ^= k0
    state[4] ^= k1
    if trace:
        show_state(state, "after XOR of K into (x3, x4)")
    return state


def absorb_ad(state: State, ad: bytes, trace: bool = False) -> None:
    if ad:
        blocks = pad(ad)
        for i in range(0, len(blocks), RATE_BYTES):
            state[0] ^= int.from_bytes(blocks[i:i + RATE_BYTES], "big")
            permutation(state, ROUNDS_B)
            if trace:
                print(f"    AD block {i // RATE_BYTES}: x0 ^= block, then p^{ROUNDS_B}")
    state[4] ^= 1  # domain separation between AD and message
    if trace:
        show_state(state, "after domain separation (x4 ^= 1)")


def encrypt_blocks(state: State, pt: bytes, trace: bool = False) -> bytes:
    padded = pad(pt)
    ct = bytearray()
    nblocks = len(padded) // RATE_BYTES
    for i in range(nblocks):
        block = int.from_bytes(padded[i * 8:(i + 1) * 8], "big")
        state[0] ^= block
        out = state[0].to_bytes(8, "big")
        if i < nblocks - 1:
            ct += out
            permutation(state, ROUNDS_B)
            if trace:
                print(f"    PT block {i}: C = x0 ^ P = {out.hex()}, then p^{ROUNDS_B}")
        else:
            last = len(pt) % RATE_BYTES
            ct += out[:last]
            if trace:
                print(f"    last PT block ({last} bytes): C = {out[:last].hex() or '(empty)'}")
    return bytes(ct)


def decrypt_blocks(state: State, ct: bytes) -> bytes:
    pt = bytearray()
    full = len(ct) // RATE_BYTES
    for i in range(full):
        c = int.from_bytes(ct[i * 8:(i + 1) * 8], "big")
        pt += (state[0] ^ c).to_bytes(8, "big")
        state[0] = c
        permutation(state, ROUNDS_B)
    rest = ct[full * 8:]
    x0 = state[0].to_bytes(8, "big")
    p_last = bytes(a ^ b for a, b in zip(x0, rest))
    pt += p_last
    # Replace the leading bytes of x0 with the ciphertext and apply padding.
    padded_last = pad(p_last)
    state[0] ^= int.from_bytes(padded_last, "big")
    return bytes(pt)


def finalise(state: State, key: bytes, trace: bool = False) -> bytes:
    k0, k1 = to_lanes(key)
    state[1] ^= k0
    state[2] ^= k1
    if trace:
        print("    K XORed into (x1, x2) -- the lanes right after the rate")
    permutation(state, ROUNDS_A)
    state[3] ^= k0
    state[4] ^= k1
    tag = state[3].to_bytes(8, "big") + state[4].to_bytes(8, "big")
    if trace:
        print(f"    p^{ROUNDS_A}, then T = (x3, x4) ^ K = {tag.hex()}")
    return tag


def ascon128_encrypt(key: bytes, nonce: bytes, ad: bytes, pt: bytes,
                     trace: bool = False) -> Tuple[bytes, bytes]:
    if trace:
        sub("Phase 1: Initialisation")
    s = initialise(key, nonce, trace)
    if trace:
        sub("Phase 2: Associated data")
    absorb_ad(s, ad, trace)
    if trace:
        sub("Phase 3: Plaintext")
    ct = encrypt_blocks(s, pt, trace)
    if trace:
        sub("Phase 4: Finalisation")
    tag = finalise(s, key, trace)
    return ct, tag


def ascon128_decrypt(key: bytes, nonce: bytes, ad: bytes, ct: bytes,
                     tag: bytes) -> bytes | None:
    s = initialise(key, nonce)
    absorb_ad(s, ad)
    pt = decrypt_blocks(s, ct)
    expected = finalise(s, key)
    diff = 0
    for a, b in zip(expected, tag):  # constant-time style comparison
        diff |= a ^ b
    return pt if diff == 0 and len(tag) == 16 else None

def section_round() -> None:
    s = [0] * 5
    show_state(s, "input (all zero)")
    p_C(s, ROUND_CONSTANTS[0])
    show_state(s, f"after p_C (c_0 = {ROUND_CONSTANTS[0]:#04x} into x2)")
    p_S(s)
    show_state(s, "after p_S")
    p_L(s)
    show_state(s, "after p_L")
    print("\n  Round constants c_0..c_11:")
    print("    " + " ".join(f"{c:02x}" for c in ROUND_CONSTANTS))


def section_sbox() -> None:
    table = derive_sbox_table()
    for row in range(0, 32, 8):
        ins = " ".join(f"{v:02x}" for v in range(row, row + 8))
        outs = " ".join(f"{table[v]:02x}" for v in range(row, row + 8))
        print(f"    x    : {ins}")
        print(f"    S(x) : {outs}")
    print(f"\n  Bijective (a permutation of 0..31): {sorted(table) == list(range(32))}")

    # Differential uniformity: max entries of the difference distribution table.
    best = 0
    for dx in range(1, 32):
        counts = [0] * 32
        for x in range(32):
            counts[table[x] ^ table[x ^ dx]] += 1
        best = max(best, max(counts))
    print(f"  Differential uniformity: {best}  (best differential prob. {best}/32)")


def section_linear() -> None:
    for i, (r1, r2) in enumerate(ROTATIONS):
        print(f"    x{i}:  rotations ({r1:2d}, {r2:2d})")
      s = [1, 0, 0, 0, 0]
    print(f"    before: x0 = {bits_of(s[0])}  (weight {hamming(s[0])})")
    p_L(s)
    print(f"    after : x0 = {bits_of(s[0])}  (weight {hamming(s[0])})")
    print("\n  Note p_L never moves bits between lanes; only p_S mixes lanes.")


def section_permutation() -> None:
    s = [IV_ASCON_128, 0, 0, 0, 0]
    show_state(s, "input: IV in x0, rest zero")
    permutation(s, ROUNDS_A)
    show_state(s, f"after p^{ROUNDS_A}")


def section_aead() -> None:

    ad = b"ASCON"
    pt = b"lightweight AEAD"
    print(f"  AD = {ad!r}   PT = {pt!r}")
    ct, tag = ascon128_encrypt(KAT_KEY, KAT_NONCE, ad, pt, trace=True)
    sub("Result")
    print(f"    ciphertext = {ct.hex()}")
    print(f"    tag        = {tag.hex()}")

    sub("Check against the official ASCON-128 v1.2 test vector")
    _, kat_tag = ascon128_encrypt(KAT_KEY, KAT_NONCE, b"", b"")
    ok = kat_tag == KAT_TAG_EMPTY
    print(f"    empty AD, empty PT -> tag {kat_tag.hex()}")
    print(f"    expected              -> {KAT_TAG_EMPTY.hex()}")
    print(f"    match: {ok}")


def section_decrypt() -> None:
    key = os.urandom(16)
    nonce = os.urandom(16)
    ad, pt = b"header", b"attack at dawn!!!"
    ct, tag = ascon128_encrypt(key, nonce, ad, pt)
    print(f"    plaintext       : {pt!r}")
    print(f"    ciphertext      : {ct.hex()}")
    print(f"    decrypt (valid) : {ascon128_decrypt(key, nonce, ad, ct, tag)!r}")
    bad = bytes([ct[0] ^ 1]) + ct[1:]
    print(f"    decrypt (1 bit of C flipped): {ascon128_decrypt(key, nonce, ad, bad, tag)!r}")
    bad_ad = ascon128_decrypt(key, nonce, b"Header", ct, tag)
    print(f"    decrypt (AD changed)        : {bad_ad!r}")


def section_diffusion() -> None:
    trials = 200
    print("    rounds   avg. flipped bits   (ideal ~160)")
    for r in range(1, 7):
        total = 0
        for _ in range(trials):
            a = to_lanes(os.urandom(40))
            b = list(a)
            b[4] ^= 1  # single-bit difference in the nonce lane x4
            sa, sb = list(a), list(b)
            for rc in ROUND_CONSTANTS[12 - r:]:
                ascon_round(sa, rc)
                ascon_round(sb, rc)
            total += sum(hamming(x ^ y) for x, y in zip(sa, sb))
        avg = total / trials
        bar = "#" * int(avg / 4)
        print(f"    {r:>6}   {avg:17.1f}   {bar}")


SECTIONS: Dict[int, Tuple[str, Callable[[], None]]] = {
    1: ("The 320-bit state", section_state),
    2: ("The round function", section_round),
    3: ("The 5-bit S-box", section_sbox),
    4: ("The linear layer", section_linear),
    5: ("The permutation p^a / p^b", section_permutation),
    6: ("ASCON-128 AEAD (traced)", section_aead),
    7: ("Decryption & verification", section_decrypt),
    8: ("Diffusion over rounds", section_diffusion),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided tour of the ASCON cipher.")
    parser.add_argument("--section", type=int, choices=sorted(SECTIONS),
                        help="run only this section")
    parser.add_argument("--list", action="store_true", help="list sections and exit")
    args = parser.parse_args()

    if args.list:
        for n, (title, _) in SECTIONS.items():
            print(f"  {n}. {title}")
        return
    if args.section:
        SECTIONS[args.section][1]()
        return
    for _, fn in SECTIONS.values():
        fn()
    print()


if __name__ == "__main__":
    main()