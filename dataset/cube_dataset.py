
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from .random_dataset import (
        IV_ASCON_128,
        LANE_BITS,
        MASK64,
        DeterministicRNG,
        ascon_permutation,
        state_to_bits,
    )
except ImportError:
    from dataset.random_dataset import (  # type: ignore
        IV_ASCON_128,
        LANE_BITS,
        MASK64,
        DeterministicRNG,
        ascon_permutation,
        state_to_bits,
    )

KEY_BITS: int = 128
NONCE_BITS: int = 128
MAX_CUBE_DIM: int = 16
DEFAULT_CUBE_DIM: int = 6
CUBE_ROUND_SCHEDULE: Tuple[int, ...] = (4, 5, 6)
KEYSTREAM_LANE: int = 0
CUBE_TESTS: Tuple[str, ...] = (
    "constant",
    "linear_blr",
    "neutral_key_bits",
    "balance",
)

@dataclass(frozen=True)
class CubeIndex:
    index: int

    @property
    def lane(self) -> int:
        return 3 if self.index < LANE_BITS else 4

    @property
    def bit(self) -> int:
        return LANE_BITS - 1 - (self.index % LANE_BITS)


@dataclass
class Cube:

    variables: Tuple[CubeIndex, ...] = ()
    name: str = "empty"

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def size(self) -> int:
        return 1 << self.dim

    def validate(self) -> None:
        if self.dim == 0:
            raise ValueError("cube must contain at least one variable")
        if self.dim > MAX_CUBE_DIM:
            raise ValueError("cube dimension too large")
        idx = [v.index for v in self.variables]
        if len(set(idx)) != len(idx):
            raise ValueError("duplicate cube variables")
        if any(not 0 <= i < NONCE_BITS for i in idx):
            raise ValueError("cube variable outside nonce range")


@dataclass
class CubeDatasetConfig:
    rounds: int = 5
    num_samples: int = 1 << 12
    cubes: Tuple[Cube, ...] = ()
    real_fraction: float = 0.5
    seed: Optional[int] = None
    output_lanes: Tuple[int, ...] = (KEYSTREAM_LANE,)
    blr_queries: int = 8
    neutral_probe_bits: int = 16
    output_dir: str = "data/cube"
    shard_size: int = 1 << 10


@dataclass
class CubeSample:

    features: Tuple[int, ...]
    label: int
    key: Tuple[int, int] = (0, 0)

def split128(value: int) -> Tuple[int, int]:
    """Split a 128-bit integer into (high 64, low 64)."""
    return (value >> 64) & MASK64, value & MASK64


def initial_state(key: int, nonce: int) -> List[int]:
    """
    Build the ASCON-128 initialisation state IV || K || N.

        x0 = IV
        x1, x2 = K (high, low)
        x3, x4 = N (high, low)
    """
    k_hi, k_lo = split128(key)
    n_hi, n_lo = split128(nonce)
    return [IV_ASCON_128, k_hi, k_lo, n_hi, n_lo]


def reduced_init(key: int, nonce: int, rounds: int) -> List[int]:
    """
    Round-reduced initialisation: p^r(IV||K||N) followed by the final
    key XOR into (x3, x4), as in the specification.
    """
    state = ascon_permutation(initial_state(key, nonce), rounds)
    k_hi, k_lo = split128(key)
    state[3] ^= k_hi
    state[4] ^= k_lo
    return state


def observe(state: Sequence[int], lanes: Sequence[int]) -> List[int]:
    """Bits visible to the attacker after initialisation."""
    return state_to_bits(state, lanes)


OracleFn = Callable[[int], List[int]]


def ascon_oracle(key: int, rounds: int, lanes: Sequence[int]) -> OracleFn:
    """Return nonce -> observed-bits for a fixed secret key."""
    def oracle(nonce: int) -> List[int]:
        return observe(reduced_init(key, nonce, rounds), lanes)
    return oracle


def random_oracle(rng: DeterministicRNG, lanes: Sequence[int]) -> OracleFn:
    """
    Lazily-sampled random function with the same interface as
    ``ascon_oracle``. Responses are memoised so repeated nonces agree.
    """
    table: Dict[int, List[int]] = {}
    width = len(lanes) * LANE_BITS

    def oracle(nonce: int) -> List[int]:
        if nonce not in table:
            raw = rng.random_bytes((width + 7) // 8)
            table[nonce] = [(raw[i // 8] >> (7 - i % 8)) & 1 for i in range(width)]
        return table[nonce]

    return oracle

def cube_from_indices(indices: Sequence[int], name: str = "") -> Cube:
    """Build a cube from raw nonce-bit indices."""
    vars_ = tuple(CubeIndex(i) for i in indices)
    return Cube(vars_, name or f"cube{list(indices)}")


def column_aligned_cube(columns: Sequence[int]) -> Cube:
    """
    Put cube variables in both nonce lanes at the same S-box column, so
    that x3[c] and x4[c] enter one S-box together (a common heuristic for
    keeping the superpoly degree low in round 1).
    """
    idx: List[int] = []
    for c in columns:
        idx.append(LANE_BITS - 1 - c)
        idx.append(2 * LANE_BITS - 1 - c)
    return cube_from_indices(idx, f"colcube{list(columns)}")


def single_lane_cube(lane: int, bits: Sequence[int]) -> Cube:
    """Cube variables restricted to x3 (lane=3) or x4 (lane=4)."""
    offset = 0 if lane == 3 else LANE_BITS
    idx = [offset + (LANE_BITS - 1 - b) for b in bits]
    return cube_from_indices(idx, f"x{lane}cube{list(bits)}")


def random_cube(rng: DeterministicRNG, dim: int) -> Cube:
    """Uniformly random cube of dimension ``dim``."""
    chosen: List[int] = []
    while len(chosen) < dim:
        i = rng.random_below(NONCE_BITS)
        if i not in chosen:
            chosen.append(i)
    return cube_from_indices(sorted(chosen), f"rnd{dim}")


def default_cube_catalogue() -> List[Cube]:
    """Placeholder set of cubes used by the prototype dataset."""
    return [
        column_aligned_cube(range(DEFAULT_CUBE_DIM // 2)),
        column_aligned_cube((0, 21, 42)),
        single_lane_cube(3, range(DEFAULT_CUBE_DIM)),
        single_lane_cube(4, range(DEFAULT_CUBE_DIM)),
    ]
def nonce_with_cube(base_nonce: int, cube: Cube, assignment: int) -> int:
    """
    Overwrite the cube variables of ``base_nonce`` with the bits of
    ``assignment`` (bit i of assignment -> i-th cube variable).
    """
    nonce = base_nonce
    for i, var in enumerate(cube.variables):
        pos = NONCE_BITS - 1 - var.index
        if (assignment >> i) & 1:
            nonce |= 1 << pos
        else:
            nonce &= ~(1 << pos)
    return nonce


def cube_sum(oracle: OracleFn, cube: Cube, base_nonce: int) -> List[int]:
    acc: Optional[List[int]] = None
    for a in range(cube.size):
        out = oracle(nonce_with_cube(base_nonce, cube, a))
        acc = list(out) if acc is None else [x ^ y for x, y in zip(acc, out)]
    return acc or []

def test_constant(
    oracle: OracleFn, cube: Cube, rng: DeterministicRNG, queries: int
) -> List[int]:
    """
    Per output bit: 1 if the superpoly looked constant over ``queries``
    random base nonces, else 0.
    """
    first = cube_sum(oracle, cube, rng.random_below(1 << NONCE_BITS))
    stable = [1] * len(first)
    for _ in range(queries - 1):
        s = cube_sum(oracle, cube, rng.random_below(1 << NONCE_BITS))
        for i, (a, b) in enumerate(zip(first, s)):
            if a != b:
                stable[i] = 0
    return stable

def test_linear_blr(
    oracle: OracleFn, cube: Cube, rng: DeterministicRNG, queries: int
) -> List[int]:
    zero = cube_sum(oracle, cube, 0)
    passed = [1] * len(zero)
    for _ in range(queries):
        x = rng.random_below(1 << NONCE_BITS)
        y = rng.random_below(1 << NONCE_BITS)
        px = cube_sum(oracle, cube, x)
        py = cube_sum(oracle, cube, y)
        pxy = cube_sum(oracle, cube, x ^ y)
        for i in range(len(zero)):
            if px[i] ^ py[i] ^ pxy[i] ^ zero[i]:
                passed[i] = 0
    return passed


def test_neutral_bits(
    oracle: OracleFn,
    cube: Cube,
    rng: DeterministicRNG,
    probe_bits: int,
) -> List[int]:

    cube_idx = {v.index for v in cube.variables}
    base = rng.random_below(1 << NONCE_BITS)
    ref = cube_sum(oracle, cube, base)
    neutral = [0] * len(ref)
    tried = 0
    while tried < probe_bits:
        j = rng.random_below(NONCE_BITS)
        if j in cube_idx:
            continue
        flipped = cube_sum(oracle, cube, base ^ (1 << (NONCE_BITS - 1 - j)))
        for i, (a, b) in enumerate(zip(ref, flipped)):
            if a == b:
                neutral[i] += 1
        tried += 1
    return neutral


def test_balance(
    oracle: OracleFn, cube: Cube, rng: DeterministicRNG, queries: int
) -> List[int]:

    ones: Optional[List[int]] = None
    for _ in range(queries):
        s = cube_sum(oracle, cube, rng.random_below(1 << NONCE_BITS))
        ones = list(s) if ones is None else [o + b for o, b in zip(ones, s)]
    return ones or []

def binarise_counts(counts: Sequence[int], total: int) -> List[int]:
    """
    Encode each count as two bits: (count == 0 or count == total) and
    (count > total / 2). Keeps feature vectors binary for every model.
    """
    out: List[int] = []
    for c in counts:
        out.append(1 if c in (0, total) else 0)
        out.append(1 if 2 * c > total else 0)
    return out


def cube_feature_block(
    oracle: OracleFn,
    cube: Cube,
    rng: DeterministicRNG,
    cfg: CubeDatasetConfig,
) -> List[int]:
    """Run every cube tester for one cube and concatenate the bits."""
    q = cfg.blr_queries
    block: List[int] = []
    block += test_constant(oracle, cube, rng, q)
    block += test_linear_blr(oracle, cube, rng, q)
    block += binarise_counts(
        test_neutral_bits(oracle, cube, rng, cfg.neutral_probe_bits),
        cfg.neutral_probe_bits,
    )
    block += binarise_counts(test_balance(oracle, cube, rng, q), q)
    return block


def cube_feature_width(cfg: CubeDatasetConfig) -> int:
    """
    Width of one row: per cube, per output bit we emit
    1 (constant) + 1 (linear) + 2 (neutral) + 2 (balance) = 6 bits.
    """
    out_bits = len(cfg.output_lanes) * LANE_BITS
    return len(cfg.cubes) * out_bits * 6

class CubeDatasetGenerator:
    def __init__(self, config: CubeDatasetConfig) -> None:
        if not config.cubes:
            config.cubes = tuple(default_cube_catalogue())
        for c in config.cubes:
            c.validate()
        self.config = config
        self.rng = DeterministicRNG(config.seed)
        self._width = cube_feature_width(config)
        self._counts = {"real": 0, "random": 0}

    def _features_for(self, oracle: OracleFn) -> List[int]:
        row: List[int] = []
        for cube in self.config.cubes:
            row += cube_feature_block(oracle, cube, self.rng, self.config)
        return row

    def _real_sample(self) -> CubeSample:
        key = self.rng.random_below(1 << KEY_BITS)
        oracle = ascon_oracle(key, self.config.rounds, self.config.output_lanes)
        self._counts["real"] += 1
        return CubeSample(tuple(self._features_for(oracle)), 1, split128(key))

    def _random_sample(self) -> CubeSample:
        oracle = random_oracle(self.rng, self.config.output_lanes)
        self._counts["random"] += 1
        return CubeSample(tuple(self._features_for(oracle)), 0)

    def draw_sample(self) -> CubeSample:
        threshold = int(self.config.real_fraction * 1_000_000)
        if self.rng.random_below(1_000_000) < threshold:
            return self._real_sample()
        return self._random_sample()

    def build_batch(self, n: int) -> Tuple[List[List[int]], List[int]]:
        xs: List[List[int]] = []
        ys: List[int] = []
        for _ in range(n):
            s = self.draw_sample()
            xs.append(list(s.features))
            ys.append(s.label)
        return xs, ys

    def iter_shards(self) -> Iterator[Tuple[int, List[List[int]], List[int]]]:
        remaining = self.config.num_samples
        shard_id = 0
        while remaining > 0:
            n = min(remaining, self.config.shard_size)
            xs, ys = self.build_batch(n)
            yield shard_id, xs, ys
            remaining -= n
            shard_id += 1

    def shard_path(self, shard_id: int) -> str:
        name = f"cube_r{self.config.rounds}_c{len(self.config.cubes)}_{shard_id:05d}.bin"
        return os.path.join(self.config.output_dir, name)

    def write_shard(self, shard_id: int, xs: List[List[int]], ys: List[int]) -> None:
        """Layout: [u32 rows][u32 width] then per row [u8 label][packed bits]."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        row_bytes = (self._width + 7) // 8
        with open(self.shard_path(shard_id), "wb") as fh:
            fh.write(struct.pack(">II", len(xs), self._width))
            for feats, label in zip(xs, ys):
                packed = bytearray(row_bytes)
                for i, b in enumerate(feats):
                    if b:
                        packed[i // 8] |= 0x80 >> (i % 8)
                fh.write(bytes([label]))
                fh.write(bytes(packed))

    def generate(self, persist: bool = False) -> Tuple[List[List[int]], List[int]]:
        all_x: List[List[int]] = []
        all_y: List[int] = []
        for shard_id, xs, ys in self.iter_shards():
            if persist:
                self.write_shard(shard_id, xs, ys)
            else:
                all_x.extend(xs)
                all_y.extend(ys)
        return all_x, all_y

    @property
    def width(self) -> int:
        return self._width

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def oracle_queries_per_sample(self) -> int:
        q = self.config.blr_queries
        p = self.config.neutral_probe_bits
        per_cube_sums = q + (3 * q + 1) + (p + 1) + q
        return sum(per_cube_sums * c.size for c in self.config.cubes)


def cube_strength(
    cube: Cube,
    rounds: int,
    keys: int = 4,
    queries: int = 8,
    seed: Optional[int] = None,
) -> float:
    rng = DeterministicRNG(seed)
    total = 0.0
    for _ in range(keys):
        key = rng.random_below(1 << KEY_BITS)
        oracle = ascon_oracle(key, rounds, (KEYSTREAM_LANE,))
        bits = test_constant(oracle, cube, rng, queries)
        total += sum(bits) / len(bits)
    return total / keys


def random_cube_search(
    dim: int,
    rounds: int,
    tries: int,
    seed: Optional[int] = None,
) -> List[Tuple[Cube, float]]:
    rng = DeterministicRNG(seed)
    results = []
    for _ in range(tries):
        c = random_cube(rng, dim)
        results.append((c, cube_strength(c, rounds, seed=seed)))
    results.sort(key=lambda t: t[1], reverse=True)
    return results

CUBE_GRID: List[Dict[str, int]] = [
    {"rounds": r, "dim": d, "samples": 1 << n}
    for r in CUBE_ROUND_SCHEDULE
    for d in (4, 6, 8)
    for n in (10, 12)
]

def plan_cube_experiments(seed: int = 2026) -> List[CubeDatasetConfig]:
    """Expand ``CUBE_GRID`` into configs. Nothing is generated."""
    plans: List[CubeDatasetConfig] = []
    for i, e in enumerate(CUBE_GRID):
        cubes = (column_aligned_cube(range(e["dim"] // 2)),)
        plans.append(
            CubeDatasetConfig(
                rounds=e["rounds"], num_samples=e["samples"], cubes=cubes, seed=seed + i
            )
        )
    return plans


def check_nonce_assignment_roundtrip() -> bool:
    cube = single_lane_cube(3, range(4))
    base = 0
    on = nonce_with_cube(base, cube, cube.size - 1)
    off = nonce_with_cube(on, cube, 0)
    return off == base and on != base



if __name__ == "__main__":

    pass
