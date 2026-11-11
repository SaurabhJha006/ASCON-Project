from __future__ import annotations

import argparse
import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

MASK64 = 0xFFFFFFFFFFFFFFFF
ROUND_CONSTANTS = [0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5,
                   0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B]
ROTATIONS = [(19, 28), (61, 39), (1, 6), (10, 17), (7, 41)]
STATE_BITS = 320

State = List[int]


def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & MASK64


def ascon_permutation(state: Sequence[int], rounds: int) -> State:
    """p^rounds on a copy of `state`, using the last `rounds` constants."""
    x0, x1, x2, x3, x4 = state
    for rc in ROUND_CONSTANTS[12 - rounds:]:
        x2 ^= rc
        x0 ^= x4; x4 ^= x3; x2 ^= x1
        t0 = ~x0 & x1; t1 = ~x1 & x2; t2 = ~x2 & x3; t3 = ~x3 & x4; t4 = ~x4 & x0
        x0 ^= t1; x1 ^= t2; x2 ^= t3; x3 ^= t4; x4 ^= t0
        x1 ^= x0; x0 ^= x4; x3 ^= x2; x2 = ~x2
        x0 &= MASK64; x1 &= MASK64; x2 &= MASK64; x3 &= MASK64; x4 &= MASK64
        x0 ^= rotr(x0, 19) ^ rotr(x0, 28)
        x1 ^= rotr(x1, 61) ^ rotr(x1, 39)
        x2 ^= rotr(x2, 1) ^ rotr(x2, 6)
        x3 ^= rotr(x3, 10) ^ rotr(x3, 17)
        x4 ^= rotr(x4, 7) ^ rotr(x4, 41)
    return [x0, x1, x2, x3, x4]


def random_state(rng: random.Random) -> State:
    return [rng.getrandbits(64) for _ in range(5)]


def state_bits(state: Sequence[int]) -> List[int]:
    """Flatten 5 lanes into 320 bits, x0 first, MSB first."""
    return [(lane >> (63 - b)) & 1 for lane in state for b in range(64)]


def bit_name(index: int) -> str:
    return f"x{index // 64}[{63 - index % 64}]"

def banner(title: str) -> None:
    print()
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)


def bar(value: float, scale: float = 40.0, maximum: float = 1.0) -> str:
    return "#" * max(0, int(round(scale * value / maximum)))

DEFAULT_DELTA: Tuple[int, ...] = (0, 0, 0, 1 << 63, 1 << 63)


def real_oracle(rng: random.Random, rounds: int, delta: Sequence[int]) -> State:
    """
    REAL world: pick random P, return the output difference
        p^r(P) xor p^r(P xor delta).
    """
    p0 = random_state(rng)
    p1 = [a ^ d for a, d in zip(p0, delta)]
    c0 = ascon_permutation(p0, rounds)
    c1 = ascon_permutation(p1, rounds)
    return [a ^ b for a, b in zip(c0, c1)]


def random_oracle(rng: random.Random, rounds: int, delta: Sequence[int]) -> State:
    return random_state(rng)


Oracle = Callable[[random.Random, int, Sequence[int]], State]


def profile_biases(rounds: int, delta: Sequence[int], samples: int,
                   rng: random.Random) -> List[float]:

    ones = [0] * STATE_BITS
    for _ in range(samples):
        for i, b in enumerate(state_bits(real_oracle(rng, rounds, delta))):
            ones[i] += b
    return [o / samples for o in ones]


def select_bits(probs: Sequence[float], k: int, min_bias: float) -> List[int]:
    ranked = sorted(range(STATE_BITS), key=lambda i: abs(probs[i] - 0.5), reverse=True)
    return [i for i in ranked[:k] if abs(probs[i] - 0.5) >= min_bias]


def llr_score(diffs: Sequence[State], probs: Sequence[float],
              bits: Sequence[int]) -> float:
    eps = 1e-6
    w1 = {i: math.log(max(probs[i], eps) / 0.5) for i in bits}
    w0 = {i: math.log(max(1 - probs[i], eps) / 0.5) for i in bits}
    total = 0.0
    for d in diffs:
        for i in bits:
            bit = (d[i // 64] >> (63 - i % 64)) & 1
            total += w1[i] if bit else w0[i]
    return total


def play_game(oracle: Oracle, rounds: int, delta: Sequence[int], queries: int,
              probs: Sequence[float], bits: Sequence[int],
              rng: random.Random) -> int:
    """Query the oracle `queries` times and guess 1 (real) or 0 (random)."""
    diffs = [oracle(rng, rounds, delta) for _ in range(queries)]
    return 1 if llr_score(diffs, probs, bits) > 0 else 0


def measure(rounds: int, delta: Sequence[int], queries: int, games: int,
            probs: Sequence[float], bits: Sequence[int],
            rng: random.Random) -> Dict[str, float]:
    """Play `games` rounds of the game in each world; report accuracy."""
    hits_real = sum(play_game(real_oracle, rounds, delta, queries, probs, bits, rng)
                    for _ in range(games))
    hits_rand = sum(1 - play_game(random_oracle, rounds, delta, queries, probs, bits, rng)
                    for _ in range(games))
    tpr = hits_real / games
    tnr = hits_rand / games
    return {"tpr": tpr, "tnr": tnr, "accuracy": (tpr + tnr) / 2,
            "advantage": abs(tpr + tnr - 1)}


class Settings:
    profile_samples = 4000
    games = 200
    seed = 2026


def section_toy() -> None:
      rng = random.Random(Settings.seed)
    for n in (10, 100, 400, 1600):
        correct = 0
        for _ in range(500):
            heads = sum(rng.random() < 0.5 for _ in range(n))
            correct += heads <= 0.525 * n
            heads = sum(rng.random() < 0.55 for _ in range(n))
            correct += heads > 0.525 * n
        acc = correct / 1000
        print(f"    {n:7d}   {acc:.3f}  {bar(acc - 0.5, maximum=0.5)}")



def section_delta() -> None:

    candidates = {
        "x0[63] only":           (1 << 63, 0, 0, 0, 0),
        "x4[0] only":            (0, 0, 0, 0, 1),
        "x3[63] + x4[63] (col)": DEFAULT_DELTA,
        "x0..x4 all at bit 63":  tuple([1 << 63] * 5),
        "two columns (x3,x4)":   (0, 0, 0, (1 << 63) | 1, (1 << 63) | 1),
        "random 320-bit delta":  tuple(random_state(random.Random(7))),
    }
    rng = random.Random(Settings.seed)
    print("    candidate                  biased  deterministic")
    for name, delta in candidates.items():
        probs = profile_biases(3, delta, 500, rng)
        strong = sum(abs(p - 0.5) > 0.2 for p in probs)
        fixed = sum(p in (0.0, 1.0) for p in probs)
        print(f"    {name:24s}   {strong:5d}   {fixed:5d}   {bar(strong, maximum=100)}")




def section_offline() -> None:

    rng = random.Random(Settings.seed)
    probs = profile_biases(3, DEFAULT_DELTA, Settings.profile_samples, rng)
    ranked = sorted(range(STATE_BITS), key=lambda i: abs(probs[i] - 0.5), reverse=True)
    print("    bit        Pr[=1]   bias |p-1/2|")
    for i in ranked[:12]:
        b = abs(probs[i] - 0.5)
        print(f"    {bit_name(i):9s}  {probs[i]:.3f}    {b:.3f}  {bar(b, maximum=0.5)}")
    noise = 1 / math.sqrt(Settings.profile_samples)
    over = sum(abs(p - 0.5) > 3 * noise for p in probs)


def section_online() -> None:
    banner("5. Online phase: the decision rule")
    rng = random.Random(Settings.seed + 1)
    probs = profile_biases(3, DEFAULT_DELTA, Settings.profile_samples, rng)
    bits = select_bits(probs, 32, 0.05)
    for name, oracle in (("REAL  ", real_oracle), ("RANDOM", random_oracle)):
        diffs = [oracle(rng, 3, DEFAULT_DELTA) for _ in range(4)]
        s = llr_score(diffs, probs, bits)
        guess = "REAL" if s > 0 else "RANDOM"
        print(f"    box is {name} -> LLR = {s:+9.2f} -> guess {guess}")


def section_play() -> None:
    banner("6. Playing the game: accuracy and advantage")
    rng = random.Random(Settings.seed + 2)
    probs = profile_biases(3, DEFAULT_DELTA, Settings.profile_samples, rng)
    bits = select_bits(probs, 32, 0.05)
    print(f"""
  Each world is played {Settings.games} times with 3-round ASCON,
  K = {len(bits)} bits. N = number of chosen-difference queries per game.
""")
    print("    N    TPR     TNR     accuracy   advantage")
    for n in (1, 2, 4, 8):
        m = measure(3, DEFAULT_DELTA, n, Settings.games, probs, bits, rng)
        print(f"    {n:<3d}  {m['tpr']:.3f}   {m['tnr']:.3f}   {m['accuracy']:.3f}"
              f"      {m['advantage']:.3f}  {bar(m['advantage'])}")



def section_complexity() -> None:
    banner("7. How many samples are needed?")
    rng = random.Random(Settings.seed + 3)
    samples = 5 * Settings.profile_samples
    noise = 1 / math.sqrt(samples)
    print(f"    ({samples} profiling samples; sampling noise ~ {noise:.4f})\n")
    print("    rounds   best bias   real bits*   ~N with best bit")
    for r in (3, 4, 5):
        probs = profile_biases(r, DEFAULT_DELTA, samples, rng)
        biases = sorted((abs(p - 0.5) for p in probs), reverse=True)
        best = biases[0]
        real = sum(b > 4 * noise for b in biases)
        n1 = f"{1 / best ** 2:,.0f}" if real else "n/a (only noise)"
        print(f"    {r:6d}   {best:9.4f}   {real:10d}   {n1}")

def section_rounds() -> None:
    banner("8. Rounds vs distinguishability")
    rng = random.Random(Settings.seed + 4)
    print("    rounds   accuracy   advantage   verdict")
    for r in (1, 2, 3, 4, 5):
        probs = profile_biases(r, DEFAULT_DELTA, Settings.profile_samples, rng)
        bits = select_bits(probs, 32, 0.0)
        m = measure(r, DEFAULT_DELTA, 16, Settings.games, probs, bits, rng)
        # 3-sigma threshold for a coin-flip guesser on 2*games trials
        threshold = 0.5 + 3 * 0.5 / math.sqrt(2 * Settings.games)
        verdict = "DISTINGUISHED" if m["accuracy"] > threshold else "looks random"
        print(f"    {r:6d}   {m['accuracy']:8.3f}   {m['advantage']:9.3f}   {verdict}"
              f"  {bar(m['advantage'], scale=20)}")

    probs = profile_biases(4, DEFAULT_DELTA, 5 * Settings.profile_samples, rng)
    bits = select_bits(probs, 1, 0.0)
    games = max(20, Settings.games // 4)
    m = measure(4, DEFAULT_DELTA, 1500, games, probs, bits, rng)
    threshold = 0.5 + 3 * 0.5 / math.sqrt(2 * games)
    verdict = "DISTINGUISHED" if m["accuracy"] > threshold else "looks random"
    print(f"    bit {bit_name(bits[0])}, bias {abs(probs[bits[0]] - 0.5):.3f}, "
          f"{games} games per world")
    print(f"    4 (N=1500)  accuracy {m['accuracy']:.3f}   advantage {m['advantage']:.3f}"
          f"   {verdict}")


SECTIONS: Dict[int, Tuple[str, Callable[[], None]]] = {
    1: ("The distinguishing game", section_game),
    2: ("A toy distinguisher (biased coin)", section_toy),
    3: ("Choosing an input difference", section_delta),
    4: ("Offline phase: learning biases", section_offline),
    5: ("Online phase: the decision rule", section_online),
    6: ("Playing the game", section_play),
    7: ("How many samples are needed?", section_complexity),
    8: ("Rounds vs distinguishability", section_rounds),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="How a simple distinguisher works.")
    parser.add_argument("--section", type=int, choices=sorted(SECTIONS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="fewer samples (faster, noisier numbers)")
    args = parser.parse_args()

    if args.quick:
        Settings.profile_samples = 1000
        Settings.games = 60
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
