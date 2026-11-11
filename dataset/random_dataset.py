from __future__ import annotations

import os
import struct
import secrets
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


MASK64: int = 0xFFFFFFFFFFFFFFFF
NUM_LANES: int = 5
LANE_BITS: int = 64
STATE_BITS: int = NUM_LANES * LANE_BITS
MAX_ROUNDS: int = 12
ROUND_CONSTANTS: Tuple[int, ...] = (
    0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5,
    0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B,
)
LINEAR_ROTATIONS: Tuple[Tuple[int, int], ...] = (
    (19, 28),
    (61, 39),
    (1, 6),
    (10, 17),
    (7, 41),
)
IV_ASCON_128: int = 0x80400C0600000000
IV_ASCON_128A: int = 0x80800C0800000000
IV_ASCON_HASH: int = 0x00400C0000000100

SBOX_TABLE: Tuple[int, ...] = (
    0x04, 0x0B, 0x1F, 0x14, 0x1A, 0x15, 0x09, 0x02,
    0x1B, 0x05, 0x08, 0x12, 0x1D, 0x03, 0x06, 0x1C,
    0x1E, 0x13, 0x07, 0x0E, 0x00, 0x0D, 0x11, 0x18,
    0x10, 0x0C, 0x01, 0x19, 0x16, 0x0A, 0x0F, 0x17,
)

DEFAULT_ROUND_SCHEDULE: Tuple[int, ...] = (3, 4, 5)

@dataclass
class DifferenceSpec:


    lanes: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    name: str = "zero"

    def is_zero(self) -> bool:
        return all(lane == 0 for lane in self.lanes)

    def hamming_weight(self) -> int:
        return sum(bin(lane & MASK64).count("1") for lane in self.lanes)


@dataclass
class RandomDatasetConfig:
  
    rounds: int = 4
    num_samples: int = 1 << 20
    real_fraction: float = 0.5
    difference: DifferenceSpec = field(default_factory=DifferenceSpec)
    seed: Optional[int] = None
    feature_mode: str = "concat"          # "concat" | "xor" | "both"
    observed_lanes: Tuple[int, ...] = (0, 1, 2, 3, 4)
    output_dir: str = "data/random"
    shard_size: int = 1 << 16
    dtype: str = "uint8"


@dataclass
class SampleRecord:

    left: Tuple[int, ...]
    right: Tuple[int, ...]
    label: int
    meta: Dict[str, int] = field(default_factory=dict)


def rotr64(value: int, shift: int) -> int:
    shift %= LANE_BITS
    value &= MASK64
    return ((value >> shift) | (value << (LANE_BITS - shift))) & MASK64


def rotl64(value: int, shift: int) -> int:
    return rotr64(value, LANE_BITS - (shift % LANE_BITS))


def bytes_to_state(data: bytes) -> List[int]:

    if len(data) != STATE_BITS // 8:
        raise ValueError("ASCON state must be exactly 40 bytes")
    return list(struct.unpack(">5Q", data))


def state_to_bytes(state: Sequence[int]) -> bytes:
    return struct.pack(">5Q", *(lane & MASK64 for lane in state))


def xor_states(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [(x ^ y) & MASK64 for x, y in zip(a, b)]


def copy_state(state: Sequence[int]) -> List[int]:
    return [lane & MASK64 for lane in state]


def state_hamming_weight(state: Sequence[int]) -> int:
    return sum(bin(lane & MASK64).count("1") for lane in state)


def lane_bit(lane: int, index: int) -> int:
    return (lane >> index) & 1


def set_lane_bit(lane: int, index: int, bit: int) -> int:
    if bit:
        return (lane | (1 << index)) & MASK64
    return lane & ~(1 << index) & MASK64

def add_round_constant(state: List[int], round_index: int) -> None:

    state[2] ^= ROUND_CONSTANTS[round_index]
    state[2] &= MASK64


def substitution_layer(state: List[int]) -> None:

    x0, x1, x2, x3, x4 = state

    x0 ^= x4
    x4 ^= x3
    x2 ^= x1

    t0 = (~x0) & x1
    t1 = (~x1) & x2
    t2 = (~x2) & x3
    t3 = (~x3) & x4
    t4 = (~x4) & x0

    x2 = ~x2

    state[0] = x0 & MASK64
    state[1] = x1 & MASK64
    state[2] = x2 & MASK64
    state[3] = x3 & MASK64
    state[4] = x4 & MASK64


def linear_layer(state: List[int]) -> None:

    for i, (r1, r2) in enumerate(LINEAR_ROTATIONS):
        lane = state[i]
        state[i] = (lane ^ rotr64(lane, r1) ^ rotr64(lane, r2)) & MASK64


def ascon_round(state: List[int], round_index: int) -> None:
    """Apply one full round p = p_L o p_S o p_C in place."""
    add_round_constant(state, round_index)
    substitution_layer(state)
    linear_layer(state)


def ascon_permutation(state: Sequence[int], rounds: int) -> List[int]:

    if not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError("rounds must be between 1 and 12")
    work = copy_state(state)
    start = MAX_ROUNDS - rounds
    for r in range(start, MAX_ROUNDS):
        ascon_round(work, r)
    return work


def sbox_lookup(value: int) -> int:
    return SBOX_TABLE[value & 0x1F]


def extract_column(state: Sequence[int], column: int) -> int:
    value = 0
    for lane in state:
        value = (value << 1) | lane_bit(lane, column)
    return value


def inject_column(state: List[int], column: int, value: int) -> None:
    """Write a 5-bit column value back into the state in place."""
    for i in range(NUM_LANES):
        bit = (value >> (NUM_LANES - 1 - i)) & 1
        state[i] = set_lane_bit(state[i], column, bit)


class DeterministicRNG:

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._counter = 0
        self._buffer = b""

    def _refill(self, n: int) -> None:
        if self._seed is None:
            self._buffer += secrets.token_bytes(max(n, 4096))
            return
        material = struct.pack(">QQ", self._seed & MASK64, self._counter)
        self._counter += 1
        self._buffer += hashlib.shake_128(material).digest(max(n, 4096))

    def random_bytes(self, n: int) -> bytes:
        """Return ``n`` pseudo-random bytes."""
        if len(self._buffer) < n:
            self._refill(n - len(self._buffer))
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def random_word(self) -> int:
        """Return one uniformly random 64-bit word."""
        return struct.unpack(">Q", self.random_bytes(8))[0]

    def random_state(self) -> List[int]:
        """Return a uniformly random 320-bit state."""
        return bytes_to_state(self.random_bytes(STATE_BITS // 8))

    def random_bit(self) -> int:
        """Return a single random bit."""
        return self.random_bytes(1)[0] & 1

    def random_below(self, bound: int) -> int:
        """Uniform integer in [0, bound) via rejection sampling."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        width = max(1, (bound - 1).bit_length())
        nbytes = (width + 7) // 8
        mask = (1 << width) - 1
        while True:
            candidate = int.from_bytes(self.random_bytes(nbytes), "big") & mask
            if candidate < bound:
                return candidate


def single_bit_difference(lane: int, bit: int) -> DifferenceSpec:
    """Create a difference with exactly one active bit."""
    lanes = [0] * NUM_LANES
    lanes[lane] = 1 << bit
    return DifferenceSpec(tuple(lanes), f"x{lane}[{bit}]")


def column_difference(column: int, pattern: int) -> DifferenceSpec:

    lanes = [0] * NUM_LANES
    for i in range(NUM_LANES):
        if (pattern >> (NUM_LANES - 1 - i)) & 1:
            lanes[i] = 1 << column
    return DifferenceSpec(tuple(lanes), f"col{column}:{pattern:05b}")


def nonce_only_difference(bit: int) -> DifferenceSpec:

    lane = 3 if bit < LANE_BITS else 4
    return single_bit_difference(lane, bit % LANE_BITS)


def default_difference_catalogue() -> List[DifferenceSpec]:

    catalogue: List[DifferenceSpec] = []
    for bit in (0, 1, 31, 63):
        catalogue.append(single_bit_difference(3, bit))
        catalogue.append(single_bit_difference(4, bit))
    for pattern in (0b00011, 0b00110, 0b11000):
        catalogue.append(column_difference(63, pattern))
    return catalogue

def state_to_bits(state: Sequence[int], lanes: Iterable[int]) -> List[int]:

    bits: List[int] = []
    for i in lanes:
        lane = state[i]
        for b in range(LANE_BITS - 1, -1, -1):
            bits.append((lane >> b) & 1)
    return bits


def pair_to_features(
    left: Sequence[int],
    right: Sequence[int],
    mode: str,
    lanes: Sequence[int],
) -> List[int]:

    if mode == "concat":
        return state_to_bits(left, lanes) + state_to_bits(right, lanes)
    if mode == "xor":
        return state_to_bits(xor_states(left, right), lanes)
    if mode == "both":
        return (
            state_to_bits(left, lanes)
            + state_to_bits(right, lanes)
            + state_to_bits(xor_states(left, right), lanes)
        )
    raise ValueError(f"unknown feature mode: {mode}")


def feature_width(mode: str, lanes: Sequence[int]) -> int:
    """Width of the feature vector produced by ``pair_to_features``."""
    per_state = len(lanes) * LANE_BITS
    return {"concat": 2 * per_state, "xor": per_state, "both": 3 * per_state}[mode]


def pack_bits(bits: Sequence[int]) -> bytes:
    """Pack a bit list into bytes (MSB first) for compact storage."""
    out = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= 0x80 >> (i % 8)
    return bytes(out)


def unpack_bits(data: bytes, width: int) -> List[int]:
    """Inverse of ``pack_bits``."""
    return [(data[i // 8] >> (7 - i % 8)) & 1 for i in range(width)]

class RandomDatasetGenerator:
 
    def __init__(self, config: RandomDatasetConfig) -> None:
        self.config = config
        self.rng = DeterministicRNG(config.seed)
        self._validate_config()
        self._width = feature_width(config.feature_mode, config.observed_lanes)
        self._stats: Dict[str, int] = {"real": 0, "random": 0}


    def _validate_config(self) -> None:
        cfg = self.config
        if not 1 <= cfg.rounds <= MAX_ROUNDS:
            raise ValueError("rounds out of range")
        if cfg.difference.is_zero():
            # A zero difference makes real and random trivially separable
            # in the wrong direction, so we default to a 1-bit nonce delta.
            cfg.difference = nonce_only_difference(63)
        if not 0.0 < cfg.real_fraction < 1.0:
            raise ValueError("real_fraction must be in (0, 1)")
        if any(not 0 <= lane < NUM_LANES for lane in cfg.observed_lanes):
            raise ValueError("observed lane index out of range")


    def _real_pair(self) -> SampleRecord:
        p0 = self.rng.random_state()
        p1 = xor_states(p0, self.config.difference.lanes)
        c0 = ascon_permutation(p0, self.config.rounds)
        c1 = ascon_permutation(p1, self.config.rounds)
        self._stats["real"] += 1
        return SampleRecord(tuple(c0), tuple(c1), 1)

    def _random_pair(self) -> SampleRecord:
        p0 = self.rng.random_state()
        p1 = self.rng.random_state()
        c0 = ascon_permutation(p0, self.config.rounds)
        c1 = ascon_permutation(p1, self.config.rounds)
        self._stats["random"] += 1
        return SampleRecord(tuple(c0), tuple(c1), 0)

    def draw_sample(self) -> SampleRecord:
        """Draw one sample, choosing real vs random by ``real_fraction``."""
        threshold = int(self.config.real_fraction * 1_000_000)
        if self.rng.random_below(1_000_000) < threshold:
            return self._real_pair()
        return self._random_pair()

    def sample_to_row(self, sample: SampleRecord) -> Tuple[List[int], int]:
        """Convert a record into (feature_bits, label)."""
        feats = pair_to_features(
            sample.left,
            sample.right,
            self.config.feature_mode,
            self.config.observed_lanes,
        )
        return feats, sample.label


    def iter_samples(self, n: int) -> Iterator[SampleRecord]:
        """Lazily yield ``n`` samples."""
        for _ in range(n):
            yield self.draw_sample()

    def build_batch(self, n: int) -> Tuple[List[List[int]], List[int]]:
        """Materialise ``n`` rows as Python lists (X, y)."""
        xs: List[List[int]] = []
        ys: List[int] = []
        for sample in self.iter_samples(n):
            feats, label = self.sample_to_row(sample)
            xs.append(feats)
            ys.append(label)
        return xs, ys

    def iter_shards(self) -> Iterator[Tuple[int, List[List[int]], List[int]]]:
        """Yield (shard_id, X, y) shards of ``shard_size`` rows."""
        remaining = self.config.num_samples
        shard_id = 0
        while remaining > 0:
            n = min(remaining, self.config.shard_size)
            xs, ys = self.build_batch(n)
            yield shard_id, xs, ys
            remaining -= n
            shard_id += 1


    def shard_path(self, shard_id: int) -> str:
        """Filesystem path for a given shard."""
        name = f"random_r{self.config.rounds}_{shard_id:05d}.bin"
        return os.path.join(self.config.output_dir, name)

    def write_shard(self, shard_id: int, xs: List[List[int]], ys: List[int]) -> None:
         os.makedirs(self.config.output_dir, exist_ok=True)
        with open(self.shard_path(shard_id), "wb") as fh:
            fh.write(struct.pack(">II", len(xs), self._width))
            for feats, label in zip(xs, ys):
                fh.write(bytes([label]))
                fh.write(pack_bits(feats))

    def read_shard(self, shard_id: int) -> Tuple[List[List[int]], List[int]]:
        """Load a shard written by ``write_shard``."""
        with open(self.shard_path(shard_id), "rb") as fh:
            rows, width = struct.unpack(">II", fh.read(8))
            row_bytes = (width + 7) // 8
            xs: List[List[int]] = []
            ys: List[int] = []
            for _ in range(rows):
                ys.append(fh.read(1)[0])
                xs.append(unpack_bits(fh.read(row_bytes), width))
        return xs, ys

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
        """Feature width in bits."""
        return self._width

    @property
    def stats(self) -> Dict[str, int]:
        """Running count of real / random samples drawn so far."""
        return dict(self._stats)

def train_val_test_split(
    xs: List[List[int]],
    ys: List[int],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: Optional[int] = None,
) -> Dict[str, Tuple[List[List[int]], List[int]]]:
 
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    rng = DeterministicRNG(seed)
    order = list(range(len(xs)))
    for i in range(len(order) - 1, 0, -1):
        j = rng.random_below(i + 1)
        order[i], order[j] = order[j], order[i]
    n = len(order)
    a = int(ratios[0] * n)
    b = a + int(ratios[1] * n)
    parts = {"train": order[:a], "val": order[a:b], "test": order[b:]}
    return {
        key: ([xs[i] for i in idx], [ys[i] for i in idx])
        for key, idx in parts.items()
    }


def class_balance(ys: Sequence[int]) -> Dict[int, int]:
    """Return a label -> count histogram."""
    hist: Dict[int, int] = {}
    for y in ys:
        hist[y] = hist.get(y, 0) + 1
    return hist

def output_bit_bias(
    difference: DifferenceSpec,
    rounds: int,
    trials: int,
    seed: Optional[int] = None,
) -> List[float]:

    rng = DeterministicRNG(seed)
    zeros = [0] * STATE_BITS
    for _ in range(trials):
        p0 = rng.random_state()
        p1 = xor_states(p0, difference.lanes)
        d = xor_states(ascon_permutation(p0, rounds), ascon_permutation(p1, rounds))
        for idx, bit in enumerate(state_to_bits(d, range(NUM_LANES))):
            if bit == 0:
                zeros[idx] += 1
    return [abs(z / trials - 0.5) for z in zeros]


def screen_differences(
    catalogue: Sequence[DifferenceSpec],
    rounds: int,
    trials: int = 1 << 12,
    seed: Optional[int] = None,
) -> List[Tuple[DifferenceSpec, float]]:

    scored: List[Tuple[DifferenceSpec, float]] = []
    for diff in catalogue:
        biases = output_bit_bias(diff, rounds, trials, seed)
        scored.append((diff, max(biases)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def biased_bit_positions(biases: Sequence[float], threshold: float) -> List[int]:
    """Indices of output bits whose bias exceeds ``threshold``."""
    return [i for i, b in enumerate(biases) if b > threshold]


__all__ = [
    "DifferenceSpec", "RandomDatasetConfig", "SampleRecord", "DeterministicRNG",
    "ascon_round", "ascon_permutation", "xor_states", "state_to_bits",
    "pair_to_features", "feature_width", "RandomDatasetGenerator",
    "train_val_test_split", "screen_differences",
]

if __name__ == "__main__":

    pass
