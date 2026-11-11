# ASCON Distinguisher: 
A design skeleton for a simple distinguisher against round-reduced ASCON.
Every function has a typed signature, a docstring
and numbered TODO steps, and raises `NotImplementedError`. Nothing is
implemented, and importing any module produces no output.

The purpose of this repository is to show the planned architecture, module
boundaries and data flow before implementation begins.

## Goal

Build and evaluate distinguishers that tell a round-reduced ASCON permutation
apart from an ideal random permutation, using two classic strategies:

1. **Differential distinguisher**: fixed input difference, statistics on the output difference.
2. **Cube / integral distinguisher**: XOR sums of outputs over a cube of inputs, tested for zero-sum behaviour.

Results are reported as a distinguishing advantage with a confidence level.

## Repository layout

| File | Lines | Responsibility |
|---|---|---|
| `ascon_permutation.py` | ~690 | State container, round layers (constant, S-box, linear), reduced-round permutation oracle |
| `sbox_analysis.py` | ~700 | S-box table, difference distribution table (DDT), linear approximation table (LAT), algebraic normal form |
| `differential_distinguisher.py` | ~630 | Input differences, trail representation and search, differential experiment driver |
| `cube_distinguisher.py` | ~690 | Cube definition and enumeration, superpoly analysis, zero-sum tests, cube experiment driver |
| `statistics_tests.py` | ~720 | Chi-square, binomial, KS tests, baselines, hypothesis testing, advantage and sample-size helpers |
| `experiment_runner.py` | ~660 | Configuration, seeded RNG, runner, result store, report builder |

## Architecture

```
                 ExperimentConfig
                        |
                 ExperimentRunner
                        |
        +---------------+----------------+
        |                                |
DifferentialDistinguisher        CubeDistinguisher
        |                                |
        +---------> AsconPermutation <---+     (reduced-round oracle)
                        |
                 sbox_analysis (DDT / LAT / ANF) guides parameter choice
                        |
                 statistics_tests (baseline, p-values, advantage)
                        |
                 ResultStore -> ReportBuilder
```

### Data flow

1. `ExperimentConfig` fixes the round count, sample budget, seed and distinguisher type.
2. `ExperimentRunner` builds the `AsconPermutation` oracle and the chosen distinguisher.
3. The distinguisher generates inputs (pairs or cubes) and queries the oracle.
4. Observations are collected into an `EmpiricalDistribution`.
5. `statistics_tests` compares them against a `BaselineModel` and returns an advantage.
6. `ResultStore` and `ReportBuilder` persist and present the outcome.

## Key classes

- `AsconState`, `AsconRound`, `AsconPermutation`: the primitive and its oracle interface.
- `SboxTable`, `DifferenceDistributionTable`, `LinearApproximationTable`, `AlgebraicNormalForm`: S-box properties used to choose differences and cubes.
- `DifferenceSpec`, `DifferentialTrail`, `TrailSearch`, `DifferentialDistinguisher`: differential pipeline.
- `CubeSpec`, `CubeEnumerator`, `SuperpolyAnalyzer`, `ZeroSumTester`, `CubeDistinguisher`: cube pipeline.
- `EmpiricalDistribution`, `BaselineModel`, `HypothesisTester`: statistical layer.
- `ExperimentConfig`, `SeededRng`, `ExperimentRunner`, `ResultStore`, `ReportBuilder`: orchestration.

## Planned implementation order

1. `ascon_permutation.py`: state, constants and round layers, checked against the official test vectors.
2. `sbox_analysis.py`: DDT, LAT and ANF for the 5-bit S-box.
3. `statistics_tests.py`: baselines and tests, validated on synthetic data.
4. `differential_distinguisher.py`: single-round trails first, then multi-round.
5. `cube_distinguisher.py`: small cubes on few rounds, then scale up.
6. `experiment_runner.py`: configuration, sweeps over round counts, reporting.

## Conventions

- Python 3.9+, type hints throughout, standard library plus an optional statistics package.
- Every unimplemented function raises `NotImplementedError` and lists its planned steps as `TODO(n)` comments.
- Randomness always goes through `SeededRng` so that experiments are reproducible.

## Open decisions

- Bit-numbering convention for state words and columns.
- Cube-size budget and the policy for choosing variable bits.
- Baseline model: random permutation versus random function.
- Whether to add a bitsliced S-box and a vectorised oracle for speed.
