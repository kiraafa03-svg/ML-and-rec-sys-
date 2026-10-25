"""Sparse PPR, implicit weighted ALS and common ranking/fusion policies.

Implementations are independent of the reference repository. In particular,
no TensorFlow runtime or dense user-by-user transition matrix is required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from threadpoolctl import threadpool_limits

from .data import Dataset, week_start


def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.divide(1.0, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    return (sparse.diags(inv) @ matrix).tocsr()


def sparse_topk(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    if k < 1:
        raise ValueError("Graph top-k must be positive")
    values, indices, indptr = [], [], [0]
    for row in range(matrix.shape[0]):
        a, b = matrix.indptr[row : row + 2]
        ix, val = matrix.indices[a:b], matrix.data[a:b]
        if len(val) > k:
            # Deterministic ties: descending weight, then ascending item index.
            threshold = np.partition(val, -k)[-k]
            higher = np.flatnonzero(val > threshold)
            tied = np.flatnonzero(val == threshold)
            tied = tied[np.argsort(ix[tied], kind="stable")[: k - len(higher)]]
            selected = np.concatenate([higher, tied])
            ix, val = ix[selected], val[selected]
        order = np.argsort(ix, kind="stable")
        indices.extend(ix[order])
        values.extend(val[order])
        indptr.append(len(values))
    return sparse.csr_matrix(
        (
            np.asarray(values),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
    )


def ppr_graph(
    matrix: sparse.csr_matrix, transform: str = "raw", topk: int = 300, remove_self: bool = False
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if transform not in {"raw", "log"}:
        raise ValueError("PPR transform must be raw or log")
    weights = matrix.copy().astype(np.float64)
    if transform == "log":
        weights.data = np.log1p(weights.data)
    if np.any(weights.data < 0):
        raise ValueError("PPR edges cannot be negative")
    restart = row_normalize(weights)
    adjacency = (weights.T @ restart).tocsr()
    if remove_self:
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
    transition = row_normalize(sparse_topk(adjacency, topk))
    return transition, restart


def ppr_scores(
    transition: sparse.csr_matrix,
    restart: sparse.csr_matrix,
    alpha: float = 0.4,
    tolerance: float = 1e-7,
    max_iter: int = 200,
    batch_size: int = 256,
) -> tuple[np.ndarray, dict]:
    if not 0 <= alpha < 1 or tolerance <= 0 or max_iter < 1 or batch_size < 1:
        raise ValueError("Invalid PPR iteration settings")
    if transition.shape != (restart.shape[1], restart.shape[1]):
        raise ValueError("PPR graph and restart dimensions differ")
    t0 = time.perf_counter()
    scores = np.zeros(restart.shape, dtype=np.float32)
    dangling = np.asarray(transition.sum(1)).ravel() == 0
    worst_delta, most_iterations, all_converged = 0.0, 0, True
    for first in range(0, restart.shape[0], batch_size):
        e = restart[first : first + batch_size].toarray()
        p = e.copy()
        delta = float("inf")
        for iteration in range(1, max_iter + 1):
            mass = p[:, dangling].sum(1, keepdims=True)
            new = alpha * (transition.T @ p.T).T + ((1 - alpha) + alpha * mass) * e
            delta = float(np.abs(new - p).sum(1).max(initial=0))
            p = new
            if delta < tolerance:
                break
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite PPR scores")
        scores[first : first + len(p)] = p.astype(np.float32)
        worst_delta = max(worst_delta, delta)
        most_iterations = max(most_iterations, iteration)
        all_converged &= delta < tolerance
    return scores, {
        "iterations": most_iterations,
        "max_l1_delta": worst_delta,
        "converged": bool(all_converged),
        "seconds": time.perf_counter() - t0,
        "alpha": alpha,
        "tolerance": tolerance,
        "dangling_nodes": int(dangling.sum()),
    }


@dataclass
class WeightedALS:
    """Minimise sum_ui c_ui(p_ui-x_u*y_i)^2 + reg*(||X||^2+||Y||^2).

    c_ui = 1 + confidence*log(1+r_ui); p_ui is binary or log(1+r_ui).
    Unobserved entries have c=1,p=0. The regulariser is added to each
    normal equation without division by the full interaction-matrix size.
    """

    factors: int = 32
    confidence: float = 3.0
    regularization: float = 0.05
    iterations: int = 20
    target: str = "binary"
    seed: int = 42
    threads: int = 1

    def _step(self, rows: sparse.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        gram = fixed.T @ fixed
        regularizer = self.regularization * np.eye(self.factors)
        output = np.zeros((rows.shape[0], self.factors), dtype=np.float64)
        for row in range(rows.shape[0]):
            a, b = rows.indptr[row : row + 2]
            ix, values = rows.indices[a:b], rows.data[a:b]
            if not len(ix):
                continue
            y = fixed[ix]
            extra = self.confidence * np.log1p(values)
            preference = np.ones(len(ix)) if self.target == "binary" else np.log1p(values)
            A = gram + (y.T * extra) @ y + regularizer
            rhs = y.T @ ((1 + extra) * preference)
            output[row] = linalg.cho_solve(
                linalg.cho_factor(A, lower=True, check_finite=False), rhs, check_finite=False
            )
        return output

    def fit(self, matrix: sparse.csr_matrix) -> "WeightedALS":
        if (
            self.factors < 1
            or self.iterations < 1
            or self.confidence < 0
            or self.regularization <= 0
        ):
            raise ValueError("Invalid ALS hyperparameters")
        if (
            self.target not in {"binary", "log"}
            or not np.isfinite(matrix.data).all()
            or np.any(matrix.data < 0)
        ):
            raise ValueError("Invalid ALS input or target")
        if min(matrix.shape) == 0:
            self.user_factors = np.zeros((matrix.shape[0], self.factors))
            self.item_factors = np.zeros((matrix.shape[1], self.factors))
            self.info = {"seconds": 0.0, "iterations": 0, "objective": []}
            return self
        t0 = time.perf_counter()
        R = matrix.tocsr().astype(np.float64)
        R.sum_duplicates()
        R.eliminate_zeros()
        Rt = R.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.01, (R.shape[1], self.factors))
        self.user_factors = np.zeros((R.shape[0], self.factors))
        history = []
        with threadpool_limits(limits=self.threads, user_api="blas"):
            for iteration in range(self.iterations):
                self.user_factors = self._step(R, self.item_factors)
                self.item_factors = self._step(Rt, self.user_factors)
                if iteration == 0 or iteration == self.iterations - 1:
                    history.append({"iteration": iteration + 1, "value": self.objective(R)})
        if not np.isfinite(self.user_factors).all() or not np.isfinite(self.item_factors).all():
            raise FloatingPointError("Nonfinite ALS factors")
        self.info = {
            "seconds": time.perf_counter() - t0,
            "iterations": self.iterations,
            "objective": history,
        }
        return self

    def objective(self, matrix: sparse.csr_matrix) -> float:
        # Sum of squared predictions over ALL pairs, without forming U x I.
        X, Y = self.user_factors, self.item_factors
        loss = float(np.sum((X.T @ X) * (Y.T @ Y)))
        coo = matrix.tocoo()
        for first in range(0, len(coo.data), 100_000):
            rows, cols = coo.row[first : first + 100_000], coo.col[first : first + 100_000]
            values = coo.data[first : first + 100_000]
            prediction = np.einsum("ij,ij->i", X[rows], Y[cols])
            c = 1 + self.confidence * np.log1p(values)
            p = np.ones(len(values)) if self.target == "binary" else np.log1p(values)
            loss += float(np.sum(c * (p - prediction) ** 2 - prediction**2))
        return loss + self.regularization * float(np.sum(X * X) + np.sum(Y * Y))

    def scores(self) -> np.ndarray:
        with threadpool_limits(limits=self.threads, user_api="blas"):
            return (self.user_factors @ self.item_factors.T).astype(np.float32)


def fallback_list(data: Dataset) -> list[int]:
    popularity = np.asarray(data.matrix.sum(0)).ravel()
    amounts = {int(item): float(value) for item, value in zip(data.items, popularity)}
    # Include unseen candidate items at zero popularity: even a completely
    # new user/catalogue can receive five valid IDs, deterministically.
    return sorted(map(int, data.pool), key=lambda item: (-amounts.get(item, 0.0), item))


def user_lookup(known: np.ndarray, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(known, requested)
    valid = indices < len(known)
    if len(known):
        valid &= known[np.minimum(indices, len(known) - 1)] == requested
    return indices, valid


def rank_top5(
    scores: np.ndarray, data: Dataset, requested_users: np.ndarray, min_score: float = -np.inf
) -> dict[int, list[int]]:
    if scores.shape != (len(data.users), len(data.items)):
        raise ValueError("Unexpected score matrix shape")
    fallback = fallback_list(data)
    if len(fallback) < 5:
        raise ValueError("Cannot return five unique valid programmes")
    known, valid = user_lookup(data.users, requested_users)
    candidate = np.flatnonzero(np.isin(data.items, data.pool))
    output = {}
    for user, row, has_history in zip(requested_users, known, valid):
        rec = []
        if has_history and len(candidate):
            values = scores[row, candidate]
            usable = np.isfinite(values) & (values > min_score)
            cols = candidate[usable]
            values = values[usable]
            if len(cols):
                # Partial selection with deterministic tie resolution.
                if len(cols) > 5:
                    threshold = np.partition(values, -5)[-5]
                    selected = np.flatnonzero(values >= threshold)
                    cols, values = cols[selected], values[selected]
                order = np.lexsort((data.items[cols], -values))[:5]
                rec = [int(data.items[col]) for col in cols[order]]
        for item in fallback:
            if len(rec) == 5:
                break
            if item not in rec:
                rec.append(item)
        output[int(user)] = rec
    return output


def fuse(
    ppr: np.ndarray,
    mf: np.ndarray,
    items: np.ndarray,
    pool: np.ndarray,
    weight_mf: float = 0.75,
    kind: str = "minmax",
) -> np.ndarray:
    if not 0 <= weight_mf <= 1 or ppr.shape != mf.shape:
        raise ValueError("Invalid fusion arguments")
    cols = np.flatnonzero(np.isin(items, pool))
    output = np.zeros_like(ppr, dtype=np.float32)
    if not len(cols):
        return output

    def transform(scores):
        values = scores[:, cols].astype(np.float64)
        if kind == "minmax":
            lo, hi = values.min(1, keepdims=True), values.max(1, keepdims=True)
            return np.divide(values - lo, hi - lo, out=np.zeros_like(values), where=hi > lo)
        if kind == "rrf":
            order = np.argsort(-values, axis=1, kind="stable")
            rank = np.empty_like(order)
            np.put_along_axis(
                rank, order, np.broadcast_to(np.arange(1, len(cols) + 1), order.shape), axis=1
            )
            return 1 / (60 + rank)
        raise ValueError("Unknown fusion method")

    output[:, cols] = weight_mf * transform(mf) + (1 - weight_mf) * transform(ppr)
    return output


def schedule_ratio(data: Dataset) -> np.ndarray:
    """Target/history exposure ratio on channels present in TRAINING views.

    Uses only channel, programme, time and duration from the future schedule.
    The schedule never supplies future user behaviour or item-content features.
    """
    channels = np.unique(data.airings["channel"])
    context = data.channels
    user_ix, valid = user_lookup(data.users, context["user"])
    channel_ix = np.searchsorted(channels, context["channel"])
    ok = channel_ix < len(channels)
    if len(channels):
        ok &= channels[np.minimum(channel_ix, len(channels) - 1)] == context["channel"]
    valid &= ok
    profile = sparse.csr_matrix(
        (np.ones(int(valid.sum())), (user_ix[valid], channel_ix[valid])),
        shape=(len(data.users), len(channels)),
    )
    profile.data[:] = 1.0

    def exposure(weeks):
        a = data.airings
        m = (
            (a["start"] >= week_start(weeks[0]))
            & (a["start"] < week_start(weeks[1]))
            & np.isin(a["show"], data.items)
        )
        return sparse.csr_matrix(
            (
                np.ones(int(m.sum())) / (weeks[1] - weeks[0]),
                (
                    np.searchsorted(channels, a["channel"][m]),
                    np.searchsorted(data.items, a["show"][m]),
                ),
            ),
            shape=(len(channels), len(data.items)),
        )

    old = (profile @ exposure(data.history_weeks)).toarray()
    new = (profile @ exposure(data.target_weeks)).toarray()
    return np.clip(
        new / np.maximum(old, 1 / (data.history_weeks[1] - data.history_weeks[0])), 0.1, 10.0
    ).astype(np.float32)
