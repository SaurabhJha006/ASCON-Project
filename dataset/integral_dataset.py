from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:  # package-style import
    from .random_dataset import (
        LANE_BITS,
        MASK64,
        NUM_LANES,
        STATE_BITS,
        DeterministicRNG,
        ascon_permutation,
        copy_state,
        state_to_bits,
    )
except ImportError:  # script-style import (running from inside dataset/)
    from random_dataset import (  # type: ignore
        LANE_BITS,
        MASK64,
        NUM_LANES,
        STATE_BITS,
        DeterministicRNG,
        ascon_permutation,
        copy_state,
        state_to_bits,
    )

MAX_ACTIVE_BITS: int = 20

DEFAULT_ACTIVE_BITS: int = 8


NONCE_LANES: Tuple[int, int] = (3, 4)

KEY_LANES: Tuple[int, int] = (1, 2)

INTEGRAL_ROUND_SCHEDULE: Tuple[int, ...] = (3, 4, 5, 6)

PROPERTY_CONSTANT = "C"
PROPERTY_ACTIVE = "A"
PROPERTY_BALANCED = "B"
PROPERTY_UNKNOWN = "U"


@dataclass(frozen=True)
class BitPosition:

    lane: int
    bit: int

    def flat_index(self) -> int:
        return self.lane * LANE_BITS + (LANE_BITS - 1 - self.bit)

    def label(self) -> str:
        return f"x{self.lane}[{self.bit}]"


@dataclass
class ActiveSet:


    positions: Tuple[BitPosition, ...] = ()
    name: str = "empty"

    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def cardinality(self) -> int:
        return 1 << self.size

    def lane_masks(self) -> List[int]:
        masks = [0] * NUM_LANES
        for pos in self.positions:
            masks[pos.lane] |= 1 << pos.bit
        return masks

    def validate(self) -> None:
        if self.size == 0:
            raise ValueError("active set must not be empty")
        if self.size > MAX_ACTIVE_BITS:
            raise ValueError("active set too large for dataset generation")
        if len(set(self.positions)) != self.size:
            raise ValueError("duplicate positions in active set")
        for pos in self.positions:
            if not (0 <= pos.lane < NUM_LANES and 0 <= pos.bit < LANE_BITS):
                raise ValueError(f"bad position {pos}")


@dataclass
class IntegralDatasetConfig:
    """Tunable parameters for the integral dataset."""

    rounds: int = 4
    num_samples: int = 1 << 14
    active_set: ActiveSet = field(default_factory=ActiveSet)
    real_fraction: float = 0.5
    seed: Optional[int] = None
    observed_lanes: Tuple[int, ...] = (0, 1, 2, 3, 4)
    randomise_constant: bool = True
    output_dir: str = "data/integral"
    shard_size: int = 1 << 12


@dataclass
class IntegralSample:
    """One labelled sample: a 320-bit XOR-sum and its label."""

    xor_sum: Tuple[int, ...]
    label: int
    constant: Tuple[int, ...] = ()


def contiguous_active_set(lane: int, start_bit: int, size: int) -> ActiveSet:
    """Active bits ``start_bit .. start_bit+size-1`` of one lane."""
    positions = tuple(
        BitPosition(lane, (start_bit + i) % LANE_BITS) for i in range(size)
    )
    return ActiveSet(positions, f"x{lane}[{start_bit}:+{size}]")


def column_active_set(columns: Sequence[int], lanes: Sequence[int]) -> ActiveSet:

    positions = tuple(BitPosition(l, c) for c in columns for l in lanes)
    return ActiveSet(positions, f"cols{list(columns)}@lanes{list(lanes)}")


def nonce_active_set(size: int, stride: int = 1) -> ActiveSet:

    positions: List[BitPosition] = []
    for i in range(size):
        lane = NONCE_LANES[i % 2]
        bit = ((i // 2) * stride) % LANE_BITS
        positions.append(BitPosition(lane, bit))
    return ActiveSet(tuple(positions), f"nonce{size}s{stride}")


def paired_column_active_set(columns: Sequence[int]) -> ActiveSet:
    """
    Activate (x3, x4) jointly at each listed column, the classical choice
    from the cube-like / integral literature on ASCON initialisation.
    """
    return column_active_set(columns, NONCE_LANES)


def default_active_set_catalogue() -> List[ActiveSet]:
    """Placeholder catalogue of candidate active sets to screen."""
    return [
        contiguous_active_set(3, 0, DEFAULT_ACTIVE_BITS),
        contiguous_active_set(4, 0, DEFAULT_ACTIVE_BITS),
        nonce_active_set(DEFAULT_ACTIVE_BITS, stride=1),
        nonce_active_set(DEFAULT_ACTIVE_BITS, stride=8),
        paired_column_active_set(range(DEFAULT_ACTIVE_BITS // 2)),
        paired_column_active_set((0, 13, 26, 39)),
    ]


def assign_active_bits(
    base: Sequence[int], active: ActiveSet, counter: int
) -> List[int]:
    """
    Return ``base`` with the active bits overwritten by the binary digits
    of ``counter`` (bit i of counter -> i-th active position).
    """
    state = copy_state(base)
    for i, pos in enumerate(active.positions):
        bit = (counter >> i) & 1
        if bit:
            state[pos.lane] |= 1 << pos.bit
        else:
            state[pos.lane] &= ~(1 << pos.bit) & MASK64
    return state


def clear_active_bits(state: Sequence[int], active: ActiveSet) -> List[int]:
    """Zero out every active position (used to build the coset constant)."""
    masks = active.lane_masks()
    return [(lane & ~m) & MASK64 for lane, m in zip(state, masks)]


def iter_integral_set(
    constant: Sequence[int], active: ActiveSet
) -> Iterator[List[int]]:

    state = clear_active_bits(constant, active)
    yield copy_state(state)
    for k in range(1, active.cardinality):
        flip = (k & -k).bit_length() - 1
        pos = active.positions[flip]
        state[pos.lane] ^= 1 << pos.bit
        yield copy_state(state)


def integral_xor_sum(
    constant: Sequence[int], active: ActiveSet, rounds: int
) -> List[int]:

    acc = [0] * NUM_LANES
    for x in iter_integral_set(constant, active):
        y = ascon_permutation(x, rounds)
        for i in range(NUM_LANES):
            acc[i] ^= y[i]
    return acc


def random_xor_sum(rng: DeterministicRNG, count: int, rounds: int) -> List[int]:

    acc = [0] * NUM_LANES
    for _ in range(count):
        y = ascon_permutation(rng.random_state(), rounds)
        for i in range(NUM_LANES):
            acc[i] ^= y[i]
    return acc


def empirical_balanced_bits(
    active: ActiveSet,
    rounds: int,
    trials: int,
    seed: Optional[int] = None,
) -> List[int]:

    rng = DeterministicRNG(seed)
    always_zero = [True] * STATE_BITS
    for _ in range(trials):
        s = integral_xor_sum(rng.random_state(), active, rounds)
        for idx, bit in enumerate(state_to_bits(s, range(NUM_LANES))):
            if bit:
                always_zero[idx] = False
    return [i for i, z in enumerate(always_zero) if z]


def sum_bit_bias(
    active: ActiveSet,
    rounds: int,
    trials: int,
    seed: Optional[int] = None,
) -> List[float]:

    rng = DeterministicRNG(seed)
    zeros = [0] * STATE_BITS
    for _ in range(trials):
        s = integral_xor_sum(rng.random_state(), active, rounds)
        for idx, bit in enumerate(state_to_bits(s, range(NUM_LANES))):
            if bit == 0:
                zeros[idx] += 1
    return [abs(z / trials - 0.5) for z in zeros]


def property_string(balanced: Sequence[int]) -> str:
 
    flags = [PROPERTY_UNKNOWN] * STATE_BITS
    for idx in balanced:
        flags[idx] = PROPERTY_BALANCED
    lanes = [
        "".join(flags[i * LANE_BITS:(i + 1) * LANE_BITS]) for i in range(NUM_LANES)
    ]
    return "|".join(lanes)


def input_property_string(active: ActiveSet) -> str:
    """Render the input property (A for active, C for constant)."""
    flags = [PROPERTY_CONSTANT] * STATE_BITS
    for pos in active.positions:
        flags[pos.flat_index()] = PROPERTY_ACTIVE
    lanes = [
        "".join(flags[i * LANE_BITS:(i + 1) * LANE_BITS]) for i in range(NUM_LANES)
    ]
    return "|".join(lanes)

SBOX_DEGREE: int = 2


def degree_upper_bound(rounds: int) -> int:

    return min(SBOX_DEGREE ** rounds, STATE_BITS)


def minimum_active_bits_for_trivial_balance(rounds: int) -> int:
    """Smallest |A| for which the trivial degree bound guarantees balance."""
    return degree_upper_bound(rounds) + 1


def is_trivially_balanced(active: ActiveSet, rounds: int) -> bool:
    """True if every output bit is balanced by the degree bound alone."""
    return active.size > degree_upper_bound(rounds)


class DivisionPropertyModel:
    def __init__(self, rounds: int, active: ActiveSet) -> None:
        self.rounds = rounds
        self.active = active
        self._cache: Dict[int, bool] = {}

    def build_constraints(self) -> List[str]:
        """Return a list of symbolic constraint names (placeholder)."""
        names: List[str] = []
        for r in range(self.rounds):
            names.append(f"const_r{r}")
            names.extend(f"sbox_r{r}_c{c}" for c in range(LANE_BITS))
            names.extend(f"lin_r{r}_x{i}" for i in range(NUM_LANES))
        return names

    def is_balanced(self, flat_index: int, trials: int = 16) -> bool:
        """Query whether output bit ``flat_index`` is balanced."""
        if flat_index not in self._cache:
            balanced = empirical_balanced_bits(self.active, self.rounds, trials)
            for idx in range(STATE_BITS):
                self._cache[idx] = idx in balanced
        return self._cache[flat_index]

    def balanced_bits(self, trials: int = 16) -> List[int]:
        """All balanced output bits according to the model."""
        return [i for i in range(STATE_BITS) if self.is_balanced(i, trials)]


def sum_to_features(xor_sum: Sequence[int], lanes: Sequence[int]) -> List[int]:
    """Flatten the selected lanes of the XOR-sum into a bit vector."""
    return state_to_bits(xor_sum, lanes)


def masked_features(
    xor_sum: Sequence[int], keep: Sequence[int]
) -> List[int]:
    bits = state_to_bits(xor_sum, range(NUM_LANES))
    return [bits[i] for i in keep]


def integral_feature_width(lanes: Sequence[int]) -> int:
    """Feature width for full-lane features."""
    return len(lanes) * LANE_BITS


class IntegralDatasetGenerator:
 
    def __init__(self, config: IntegralDatasetConfig) -> None:
        self.config = config
        if config.active_set.size == 0:
            config.active_set = nonce_active_set(DEFAULT_ACTIVE_BITS)
        config.active_set.validate()
        self.rng = DeterministicRNG(config.seed)
        self._fixed_constant = self.rng.random_state()
        self._width = integral_feature_width(config.observed_lanes)
        self._counts = {"real": 0, "random": 0}


    def _coset_constant(self) -> List[int]:
        if self.config.randomise_constant:
            return self.rng.random_state()
        return copy_state(self._fixed_constant)

    def _real_sample(self) -> IntegralSample:
        constant = self._coset_constant()
        s = integral_xor_sum(constant, self.config.active_set, self.config.rounds)
        self._counts["real"] += 1
        return IntegralSample(tuple(s), 1, tuple(constant))

    def _random_sample(self) -> IntegralSample:
        s = random_xor_sum(
            self.rng, self.config.active_set.cardinality, self.config.rounds
        )
        self._counts["random"] += 1
        return IntegralSample(tuple(s), 0)

    def draw_sample(self) -> IntegralSample:
        threshold = int(self.config.real_fraction * 1_000_000)
        if self.rng.random_below(1_000_000) < threshold:
            return self._real_sample()
        return self._random_sample()

    def sample_to_row(self, sample: IntegralSample) -> Tuple[List[int], int]:
        return sum_to_features(sample.xor_sum, self.config.observed_lanes), sample.label


    def build_batch(self, n: int) -> Tuple[List[List[int]], List[int]]:
        xs: List[List[int]] = []
        ys: List[int] = []
        for _ in range(n):
            feats, label = self.sample_to_row(self.draw_sample())
            xs.append(feats)
            ys.append(label)
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
        a = self.config.active_set
        name = f"integral_r{self.config.rounds}_a{a.size}_{shard_id:05d}.bin"
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

    def permutation_calls_per_sample(self) -> int:
        return self.config.active_set.cardinality

    def total_permutation_calls(self) -> int:
        return self.config.num_samples * self.permutation_calls_per_sample()


def score_active_set(
    active: ActiveSet,
    rounds: int,
    trials: int = 32,
    seed: Optional[int] = None,
) -> Dict[str, float]:

    biases = sum_bit_bias(active, rounds, trials, seed)
    balanced = sum(1 for b in biases if b >= 0.5)
    return {
        "balanced": float(balanced),
        "max_bias": max(biases),
        "mean_bias": sum(biases) / len(biases),
    }


def rank_active_sets(
    catalogue: Sequence[ActiveSet],
    rounds: int,
    trials: int = 32,
    seed: Optional[int] = None,
) -> List[Tuple[ActiveSet, Dict[str, float]]]:
    scored = [(a, score_active_set(a, rounds, trials, seed)) for a in catalogue]
    scored.sort(key=lambda t: (t[1]["balanced"], t[1]["mean_bias"]), reverse=True)
    return scored


def greedy_extend_active_set(
    base: ActiveSet,
    candidates: Sequence[BitPosition],
    rounds: int,
    steps: int,
    trials: int = 16,
    seed: Optional[int] = None,
) -> ActiveSet:

    current = base
    for _ in range(steps):
        best: Optional[ActiveSet] = None
        best_score = -1.0
        for pos in candidates:
            if pos in current.positions:
                continue
            trial = ActiveSet(current.positions + (pos,), current.name + "+")
            if trial.size > MAX_ACTIVE_BITS:
                continue
            score = score_active_set(trial, rounds, trials, seed)["balanced"]
            if score > best_score:
                best, best_score = trial, score
        if best is None:
            break
        current = best
    return current


def enumerate_small_active_sets(
    pool: Sequence[BitPosition], size: int
) -> Iterator[ActiveSet]:
    """All size-``size`` subsets of ``pool`` (only for tiny pools)."""
    for combo in combinations(pool, size):
        yield ActiveSet(tuple(combo), "combo")


def estimate_cost(config: IntegralDatasetConfig) -> Dict[str, int]:
    """Permutation calls and approximate disk bytes for a config."""
    width = integral_feature_width(config.observed_lanes)
    calls = config.num_samples * config.active_set.cardinality
    disk = 8 + config.num_samples * (1 + (width + 7) // 8)
    return {"permutation_calls": calls, "disk_bytes": disk}


INTEGRAL_GRID: List[Dict[str, int]] = [
    {"rounds": r, "active_bits": a, "samples": 1 << n}
    for r in INTEGRAL_ROUND_SCHEDULE
    for a in (4, 8, 12)
    for n in (12, 14)
]


def config_from_grid(entry: Dict[str, int], seed: int) -> IntegralDatasetConfig:
    return IntegralDatasetConfig(
        rounds=entry["rounds"],
        num_samples=entry["samples"],
        active_set=nonce_active_set(entry["active_bits"]),
        seed=seed,
    )


def plan_integral_experiments(seed: int = 2026) -> List[IntegralDatasetConfig]:
    """Expand ``INTEGRAL_GRID`` into config objects. Nothing is generated."""
    return [config_from_grid(e, seed + i) for i, e in enumerate(INTEGRAL_GRID)]


def check_gray_code_covers_set(active: Optional[ActiveSet] = None) -> bool:
    """The Gray-code enumeration must hit every state exactly once."""
    active = active or contiguous_active_set(3, 0, 4)
    seen = {tuple(s) for s in iter_integral_set([0] * NUM_LANES, active)}
    return len(seen) == active.cardinality


def check_one_round_balance() -> bool:
    active = contiguous_active_set(4, 0, 3)
    s = integral_xor_sum([0] * NUM_LANES, active, 1)
    return all(lane == 0 for lane in s)


def run_integral_self_checks() -> Dict[str, bool]:
    """Collect self-checks into a dict; prints nothing."""
    return {
        "gray_code": check_gray_code_covers_set(),
        "one_round": check_one_round_balance(),
    }


__all__ = [
    "BitPosition", "ActiveSet", "IntegralDatasetConfig", "IntegralSample",
    "contiguous_active_set", "column_active_set", "nonce_active_set",
    "iter_integral_set", "integral_xor_sum", "random_xor_sum",
    "empirical_balanced_bits", "sum_bit_bias", "DivisionPropertyModel",
    "IntegralDatasetGenerator", "rank_active_sets", "plan_integral_experiments",
]


if __name__ == "__main__":
    pass
