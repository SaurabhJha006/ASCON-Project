from __future__ import annotations

import math
import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type


try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover
    xgb = None  # type: ignore

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    from sklearn.linear_model import LinearRegression as _SkLinearRegression  # type: ignore
    from sklearn.linear_model import Ridge as _SkRidge  # type: ignore
except ImportError:  # pragma: no cover
    _SkLinearRegression = None  # type: ignore
    _SkRidge = None  # type: ignore

NUM_LANES: int = 5
LANE_BITS: int = 64

DATASET_WIDTHS: Dict[str, int] = {
    "random_concat": 2 * NUM_LANES * LANE_BITS,   # 640
    "random_xor": NUM_LANES * LANE_BITS,          # 320
    "integral": NUM_LANES * LANE_BITS,            # 320
    "cube": 4 * LANE_BITS * 6,                    # 4 cubes x 64 bits x 6 tests
}

#: Decision threshold on model scores.
DEFAULT_THRESHOLD: float = 0.5


@dataclass
class TrainConfig:

    epochs: int = 20
    batch_size: int = 5000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    seed: int = 2026
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"


@dataclass
class EvalResult:

    accuracy: float = 0.0
    tpr: float = 0.0
    tnr: float = 0.0
    auc: float = 0.0
    n: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        d = {"accuracy": self.accuracy, "tpr": self.tpr, "tnr": self.tnr,
             "auc": self.auc, "n": float(self.n)}
        d.update(self.extra)
        return d


def load_shard(path: str) -> Tuple[List[List[int]], List[int]]:
    """
    Read a shard written by any of the dataset generators.

    Layout: [u32 rows][u32 width] then per row [u8 label][packed bits].
    """
    with open(path, "rb") as fh:
        rows, width = struct.unpack(">II", fh.read(8))
        row_bytes = (width + 7) // 8
        xs: List[List[int]] = []
        ys: List[int] = []
        for _ in range(rows):
            ys.append(fh.read(1)[0])
            raw = fh.read(row_bytes)
            xs.append([(raw[i // 8] >> (7 - i % 8)) & 1 for i in range(width)])
    return xs, ys


def load_directory(directory: str, prefix: str = "") -> Tuple[List[List[int]], List[int]]:
    """Concatenate every shard in ``directory`` whose name starts with ``prefix``."""
    xs: List[List[int]] = []
    ys: List[int] = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".bin") and name.startswith(prefix):
            sx, sy = load_shard(os.path.join(directory, name))
            xs.extend(sx)
            ys.extend(sy)
    return xs, ys


def to_array(xs: Sequence[Sequence[int]], dtype: str = "float32") -> Any:
    """Convert nested lists to a numpy array if numpy is available."""
    if np is None:
        return [list(map(float, row)) for row in xs]
    return np.asarray(xs, dtype=dtype)


def iter_minibatches(n: int, batch_size: int, seed: int) -> List[List[int]]:
    """Shuffled index batches (pure-Python Fisher-Yates via LCG)."""
    order = list(range(n))
    state = seed & 0xFFFFFFFF
    for i in range(n - 1, 0, -1):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return [order[k:k + batch_size] for k in range(0, n, batch_size)]


def binary_metrics(
    scores: Sequence[float], labels: Sequence[int], threshold: float = DEFAULT_THRESHOLD
) -> EvalResult:
    """Accuracy, TPR, TNR and AUC from real-valued scores."""
    tp = tn = fp = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 0 and y == 0:
            tn += 1
        elif pred == 1:
            fp += 1
        else:
            fn += 1
    n = tp + tn + fp + fn
    return EvalResult(
        accuracy=(tp + tn) / n if n else 0.0,
        tpr=tp / (tp + fn) if tp + fn else 0.0,
        tnr=tn / (tn + fp) if tn + fp else 0.0,
        auc=roc_auc(scores, labels),
        n=n,
    )


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUC (Mann-Whitney U), no external dependencies."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def significance_threshold(n: int, sigmas: float = 3.0) -> float:
    """
    Accuracy a random guesser exceeds with ~``sigmas`` standard deviations
    of luck on ``n`` balanced samples: 0.5 + sigmas * 0.5 / sqrt(n).
    """
    return 0.5 + sigmas * 0.5 / math.sqrt(max(n, 1))


def is_distinguisher(result: EvalResult, sigmas: float = 3.0) -> bool:
    """True if test accuracy is significantly above random guessing."""
    return result.accuracy > significance_threshold(result.n, sigmas)



class DistinguisherModel(ABC):
    """Abstract base: every block below implements this interface."""

    name: str = "base"

    def __init__(self, input_width: int, train_cfg: Optional[TrainConfig] = None) -> None:
        self.input_width = input_width
        self.train_cfg = train_cfg or TrainConfig()
        self.history: List[Dict[str, float]] = []
        self._fitted = False

    @abstractmethod
    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        """Train the model."""

    @abstractmethod
    def predict_scores(self, X: Any) -> List[float]:
        """Return a score in [0, 1] per row (probability of 'real')."""

    def predict(self, X: Any, threshold: float = DEFAULT_THRESHOLD) -> List[int]:
        return [1 if s >= threshold else 0 for s in self.predict_scores(X)]

    def evaluate(self, X: Any, y: Sequence[int]) -> Dict[str, float]:
        return binary_metrics(self.predict_scores(X), list(y)).as_dict()

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore model weights."""

    @property
    def fitted(self) -> bool:
        return self._fitted



@dataclass
class XGBoostParams:
    """Hyper-parameters for the gradient-boosted tree distinguisher."""

    n_estimators: int = 400
    max_depth: int = 8
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 20


class XGBOOST(DistinguisherModel):
 
    name = "xgboost"

    def __init__(
        self,
        input_width: int,
        train_cfg: Optional[TrainConfig] = None,
        params: Optional[XGBoostParams] = None,
    ) -> None:
        super().__init__(input_width, train_cfg)
        self.params = params or XGBoostParams()
        self.booster: Any = None

    def _xgb_param_dict(self) -> Dict[str, Any]:
        p = self.params
        return {
            "objective": "binary:logistic",
            "max_depth": p.max_depth,
            "eta": p.learning_rate,
            "subsample": p.subsample,
            "colsample_bytree": p.colsample_bytree,
            "min_child_weight": p.min_child_weight,
            "lambda": p.reg_lambda,
            "tree_method": p.tree_method,
            "eval_metric": p.eval_metric,
            "seed": self.train_cfg.seed,
        }

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        if xgb is None:
            raise RuntimeError("xgboost is not installed (skeleton only)")
        dtrain = xgb.DMatrix(to_array(X), label=list(y))
        evals = [(dtrain, "train")]
        if X_val is not None:
            evals.append((xgb.DMatrix(to_array(X_val), label=list(y_val)), "val"))
        self.booster = xgb.train(
            self._xgb_param_dict(),
            dtrain,
            num_boost_round=self.params.n_estimators,
            evals=evals,
            early_stopping_rounds=self.params.early_stopping_rounds if X_val is not None else None,
            verbose_eval=False,
        )
        self._fitted = True

    def predict_scores(self, X: Any) -> List[float]:
        if self.booster is None:
            raise RuntimeError("model not fitted")
        return list(map(float, self.booster.predict(xgb.DMatrix(to_array(X)))))

    def feature_importance(self, kind: str = "gain") -> Dict[int, float]:
        """Map feature index -> importance (bit position -> contribution)."""
        if self.booster is None:
            return {}
        raw = self.booster.get_score(importance_type=kind)
        return {int(k[1:]): float(v) for k, v in raw.items()}

    def save(self, path: str) -> None:
        if self.booster is not None:
            self.booster.save_model(path + ".json")

    def load(self, path: str) -> None:
        if xgb is None:
            raise RuntimeError("xgboost is not installed (skeleton only)")
        self.booster = xgb.Booster()
        self.booster.load_model(path + ".json")
        self._fitted = True

@dataclass
class MLPParams:
    """Architecture of the dense distinguisher."""

    hidden_sizes: Tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    activation: str = "relu"
    batch_norm: bool = True


def _activation(name: str) -> Any:
    if nn is None:
        return None
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}[name]


def _build_mlp_network(input_width: int, p: MLPParams) -> Any:
    """Linear -> [BN] -> act -> dropout, repeated, then a sigmoid head."""
    if nn is None:
        return None
    layers: List[Any] = []
    prev = input_width
    for h in p.hidden_sizes:
        layers.append(nn.Linear(prev, h))
        if p.batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(_activation(p.activation))
        layers.append(nn.Dropout(p.dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class _TorchTrainerMixin:

    network: Any
    train_cfg: TrainConfig
    history: List[Dict[str, float]]

    def _prepare(self, X: Any) -> Any:
        return torch.as_tensor(to_array(X), dtype=torch.float32)

    def _torch_fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        if torch is None:
            raise RuntimeError("torch is not installed (skeleton only)")
        cfg = self.train_cfg
        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)
        self.network.to(device)
        opt = torch.optim.Adam(
            self.network.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        loss_fn = nn.BCELoss()
        Xt = self._prepare(X).to(device)
        yt = torch.as_tensor(list(y), dtype=torch.float32).unsqueeze(1).to(device)
        best_acc, patience = 0.0, 0
        for epoch in range(cfg.epochs):
            self.network.train()
            total = 0.0
            for idx in iter_minibatches(len(yt), cfg.batch_size, cfg.seed + epoch):
                opt.zero_grad()
                out = self.network(Xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            record = {"epoch": float(epoch), "loss": total / max(len(yt), 1)}
            if X_val is not None:
                val_acc = binary_metrics(self._torch_scores(X_val), list(y_val)).accuracy
                record["val_accuracy"] = val_acc
                if val_acc > best_acc:
                    best_acc, patience = val_acc, 0
                else:
                    patience += 1
                    if patience >= cfg.early_stopping_patience:
                        self.history.append(record)
                        break
            self.history.append(record)

    def _torch_scores(self, X: Any) -> List[float]:
        self.network.eval()
        with torch.no_grad():
            out = self.network(self._prepare(X).to(self.train_cfg.device))
        return [float(v) for v in out.squeeze(1).cpu()]


class MLP(_TorchTrainerMixin, DistinguisherModel):

    name = "mlp"

    def __init__(
        self,
        input_width: int,
        train_cfg: Optional[TrainConfig] = None,
        params: Optional[MLPParams] = None,
    ) -> None:
        super().__init__(input_width, train_cfg)
        self.params = params or MLPParams()
        self.network = _build_mlp_network(input_width, self.params)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        self._torch_fit(X, y, X_val, y_val)
        self._fitted = True

    def predict_scores(self, X: Any) -> List[float]:
        return self._torch_scores(X)

    def save(self, path: str) -> None:
        if torch is not None and self.network is not None:
            torch.save(self.network.state_dict(), path + ".pt")

    def load(self, path: str) -> None:
        if torch is None:
            raise RuntimeError("torch is not installed (skeleton only)")
        self.network.load_state_dict(torch.load(path + ".pt"))
        self._fitted = True


@dataclass
class LinearRegressionParams:
    """Settings for the linear probe."""

    ridge_alpha: float = 0.0          # 0 -> ordinary least squares
    fit_intercept: bool = True
    centre_features: bool = True      # map bits {0,1} -> {-1,+1}


class LinearRegression(DistinguisherModel):

    name = "linear_regression"

    def __init__(
        self,
        input_width: int,
        train_cfg: Optional[TrainConfig] = None,
        params: Optional[LinearRegressionParams] = None,
    ) -> None:
        super().__init__(input_width, train_cfg)
        self.params = params or LinearRegressionParams()
        self.estimator: Any = None
        self.weights: List[float] = []
        self.bias: float = 0.0

    def _transform(self, X: Any) -> Any:
        arr = to_array(X)
        if self.params.centre_features and np is not None:
            return 2.0 * arr - 1.0
        return arr

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        if _SkLinearRegression is None:
            raise RuntimeError("scikit-learn is not installed (skeleton only)")
        if self.params.ridge_alpha > 0:
            self.estimator = _SkRidge(
                alpha=self.params.ridge_alpha, fit_intercept=self.params.fit_intercept
            )
        else:
            self.estimator = _SkLinearRegression(fit_intercept=self.params.fit_intercept)
        self.estimator.fit(self._transform(X), list(y))
        self.weights = [float(w) for w in self.estimator.coef_]
        self.bias = float(self.estimator.intercept_)
        self._fitted = True

    def predict_scores(self, X: Any) -> List[float]:
        if self.estimator is None:
            raise RuntimeError("model not fitted")
        raw = self.estimator.predict(self._transform(X))
        return [min(1.0, max(0.0, float(v))) for v in raw]

    def save(self, path: str) -> None:
        with open(path + ".lin", "wb") as fh:
            fh.write(struct.pack(">Id", len(self.weights), self.bias))
            for w in self.weights:
                fh.write(struct.pack(">d", w))

    def load(self, path: str) -> None:
        with open(path + ".lin", "rb") as fh:
            n, self.bias = struct.unpack(">Id", fh.read(12))
            self.weights = [struct.unpack(">d", fh.read(8))[0] for _ in range(n)]
        self._fitted = True

@dataclass
class CNNParams:

    filters: int = 32
    kernel_size: int = 3
    depth: int = 5                # number of residual blocks
    dense_units: int = 64
    reg: float = 1e-5
    circular_padding: bool = True  # lanes are rotated by p_L, so wrap around


def reshape_to_lanes(width: int) -> Tuple[int, int]:
    if width % LANE_BITS != 0:
        raise ValueError("CNN input width must be a multiple of 64")
    return width // LANE_BITS, LANE_BITS


if nn is not None:

    class _ResidualBlock(nn.Module):  # type: ignore[misc]
        """Conv -> BN -> ReLU -> Conv -> BN -> ReLU, plus skip connection."""

        def __init__(self, ch: int, k: int, circular: bool) -> None:
            super().__init__()
            mode = "circular" if circular else "zeros"
            self.body = nn.Sequential(
                nn.Conv1d(ch, ch, k, padding=k // 2, padding_mode=mode),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.Conv1d(ch, ch, k, padding=k // 2, padding_mode=mode),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
            )

        def forward(self, x: Any) -> Any:
            return x + self.body(x)

    class _CNNNetwork(nn.Module):  # type: ignore[misc]
        """
        Gohr-style network adapted to ASCON:

            input (C x 64) -> 1x1 conv (bit-slice mixing across lanes,
            i.e. the S-box column) -> residual stack -> dense head.
        """

        def __init__(self, channels: int, p: CNNParams) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(channels, p.filters, kernel_size=1),
                nn.BatchNorm1d(p.filters),
                nn.ReLU(),
            )
            self.tower = nn.Sequential(
                *[_ResidualBlock(p.filters, p.kernel_size, p.circular_padding)
                  for _ in range(p.depth)]
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(p.filters * LANE_BITS, p.dense_units),
                nn.BatchNorm1d(p.dense_units),
                nn.ReLU(),
                nn.Linear(p.dense_units, p.dense_units),
                nn.BatchNorm1d(p.dense_units),
                nn.ReLU(),
                nn.Linear(p.dense_units, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: Any) -> Any:
            return self.head(self.tower(self.stem(x)))

else:  # pragma: no cover
    _CNNNetwork = None  # type: ignore


class CNN(_TorchTrainerMixin, DistinguisherModel):

    name = "cnn"

    def __init__(
        self,
        input_width: int,
        train_cfg: Optional[TrainConfig] = None,
        params: Optional[CNNParams] = None,
    ) -> None:
        super().__init__(input_width, train_cfg)
        self.params = params or CNNParams()
        self.channels, self.length = reshape_to_lanes(input_width)
        self.network = (
            _CNNNetwork(self.channels, self.params) if _CNNNetwork is not None else None
        )

    def _prepare(self, X: Any) -> Any:
        t = torch.as_tensor(to_array(X), dtype=torch.float32)
        return t.view(-1, self.channels, self.length)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> None:
        self._torch_fit(X, y, X_val, y_val)
        self._fitted = True

    def predict_scores(self, X: Any) -> List[float]:
        return self._torch_scores(X)

    def save(self, path: str) -> None:
        if torch is not None and self.network is not None:
            torch.save(self.network.state_dict(), path + ".pt")

    def load(self, path: str) -> None:
        if torch is None:
            raise RuntimeError("torch is not installed (skeleton only)")
        self.network.load_state_dict(torch.load(path + ".pt"))
        self._fitted = True



MODEL_REGISTRY: Dict[str, Type[DistinguisherModel]] = {
    "xgboost": XGBOOST,
    "mlp": MLP,
    "linear_regression": LinearRegression,
    "cnn": CNN,
}


def build_model(name: str, input_width: int, **kwargs: Any) -> DistinguisherModel:
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](input_width, **kwargs)


@dataclass
class ExperimentSpec:
    """One cell of the (dataset x rounds x model) results table."""

    dataset: str
    rounds: int
    model: str
    data_dir: str
    input_width: int


def run_experiment(spec: ExperimentSpec, splits: Dict[str, Tuple[Any, Any]]) -> Dict[str, Any]:
    """
    Train ``spec.model`` on the given splits and return a result row.
    The row is returned, never printed.
    """
    model = build_model(spec.model, spec.input_width)
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]
    model.fit(Xtr, ytr, Xva, yva)
    metrics = model.evaluate(Xte, yte)
    result = EvalResult(
        accuracy=metrics["accuracy"], tpr=metrics["tpr"], tnr=metrics["tnr"],
        auc=metrics["auc"], n=int(metrics["n"]),
    )
    return {
        "dataset": spec.dataset,
        "rounds": spec.rounds,
        "model": spec.model,
        **metrics,
        "advantage": abs(result.tpr + result.tnr - 1.0),
        "distinguishes": is_distinguisher(result),
    }


def experiment_matrix(
    datasets: Sequence[str] = ("random_concat", "integral", "cube"),
    rounds: Sequence[int] = (3, 4, 5),
    models: Sequence[str] = tuple(MODEL_REGISTRY),
) -> List[ExperimentSpec]:
    """Full grid of experiments to report to the panel."""
    specs: List[ExperimentSpec] = []
    for d in datasets:
        for r in rounds:
            for m in models:
                specs.append(
                    ExperimentSpec(
                        dataset=d, rounds=r, model=m,
                        data_dir=os.path.join("data", d.split("_")[0]),
                        input_width=DATASET_WIDTHS[d],
                    )
                )
    return specs


if __name__ == "__main__":
    pass
