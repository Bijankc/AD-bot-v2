"""Choose what the human labels next: diverse, balanced across hint groups, plus the frames the labels disagree on.

Pure numpy on unit-length embeddings (cosine similarity = dot product), so it is testable without a model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np


def kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Spherical k-means. Returns a cluster index per row."""
    rng = np.random.default_rng(seed)
    k = max(1, min(k, len(x)))
    # k-means++ style seeding on cosine distance
    cent = [x[rng.integers(len(x))]]
    d = 1 - x @ cent[0]
    for _ in range(1, k):
        p = np.clip(d, 0, None) ** 2
        cent.append(x[rng.choice(len(x), p=p / p.sum()) if p.sum() > 0 else rng.integers(len(x))])
        d = np.minimum(d, 1 - x @ cent[-1])
    c = np.stack(cent)
    assign = np.zeros(len(x), int)
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new, assign) and _:
            break
        assign = new
        for j in range(k):
            m = x[assign == j]
            if len(m):
                v = m.sum(0)
                c[j] = v / (np.linalg.norm(v) + 1e-9)
    return assign


def knn_probs(x: np.ndarray, xl: np.ndarray, yl: list[str], classes: list[str], k: int = 10) -> np.ndarray:
    """Class probabilities from the k most similar labelled frames (similarity-weighted)."""
    sims = x @ xl.T
    k = min(k, xl.shape[0])
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    ci = np.array([classes.index(y) for y in yl])
    probs = np.zeros((len(x), len(classes)))
    for row in range(len(x)):
        for j in top[row]:
            probs[row, ci[j]] += max(sims[row, j], 0) + 1e-6
    return probs / probs.sum(1, keepdims=True)


@dataclass
class Candidate:
    id: str
    time: datetime | None
    group: str          # day/source key for the temporal guard
    hint: str | None


def pick(cands: list[Candidate], x: np.ndarray, labels: dict[str, str], n: int = 120, clusters: int = 64,
         uncertain_frac: float = 0.3, min_gap_s: float = 10.0, seed: int = 0) -> list[tuple[str, str]]:
    """Returns [(id, why)] for n unlabelled candidates. x rows match cands; labels: id -> screen (any of cands)."""
    if not cands:
        return []
    assign = kmeans(x.astype(np.float32), clusters, seed=seed)
    lab = np.array([c.id in labels and labels[c.id] != "unsure" for c in cands])
    unl = np.array([c.id not in labels for c in cands])
    cover = Counter(assign[lab].tolist())
    # distance of every frame to the nearest labelled or chosen one (2 = nothing yet)
    mind = 1 - (x @ x[lab].T).max(1) if lab.any() else np.full(len(cands), 2.0)
    taken_times: dict[str, list[datetime]] = defaultdict(list)
    for c in (c for c, is_l in zip(cands, lab) if is_l):
        if c.time:
            taken_times[c.group].append(c.time)
    chosen: list[tuple[int, str]] = []
    avail = unl.copy()

    def ok_time(i: int) -> bool:
        c = cands[i]
        return c.time is None or all(abs((c.time - t).total_seconds()) >= min_gap_s for t in taken_times[c.group])

    def take(i: int, why: str) -> None:
        nonlocal mind
        chosen.append((i, why))
        avail[i] = False
        cover[int(assign[i])] += 1
        if cands[i].time:
            taken_times[cands[i].group].append(cands[i].time)
        mind = np.minimum(mind, 1 - x @ x[i])

    # 1) frames the labels are unsure about, at most 2 per cluster
    yl = [labels[c.id] for c, is_l in zip(cands, lab) if is_l]
    classes = sorted(set(yl))
    if len(yl) >= 20 and len(classes) >= 2:
        probs = knn_probs(x[unl], x[lab], yl, classes)
        unc = 1 - probs.max(1)
        pred = [classes[j] for j in probs.argmax(1)]
        idx = np.flatnonzero(unl)
        disagree = np.array([bool(cands[i].hint) and pred[r] != cands[i].hint for r, i in enumerate(idx)])
        score = unc + 0.5 * disagree
        per_cluster: Counter = Counter()
        for r in np.argsort(-score):
            if len(chosen) >= int(n * uncertain_frac) or score[r] < 0.2:
                break
            i = int(idx[r])
            if per_cluster[assign[i]] >= 2 or not ok_time(i):
                continue
            per_cluster[assign[i]] += 1
            take(i, f"disagrees (knn {pred[r]}, hint {cands[i].hint})" if disagree[r] else
                 f"unsure (knn {pred[r]} {1 - unc[r]:.2f})")

    # 2) diverse and balanced: round-robin over hint groups, rarest first
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cands):
        groups[c.hint or "-"].append(i)
    order = sorted(groups, key=lambda g: len(groups[g]))
    while len(chosen) < n:
        progressed = False
        for g in order:
            if len(chosen) >= n:
                break
            pool = [i for i in groups[g] if avail[i]]
            while pool:
                cl = min({int(assign[i]) for i in pool}, key=lambda j: cover[j])
                in_cl = [i for i in pool if assign[i] == cl]
                best = max(in_cl, key=lambda i: mind[i])
                if ok_time(best):
                    take(best, f"diverse ({g}, cluster {cl})")
                    progressed = True
                    break
                avail[best] = False  # too close in time to a chosen frame
                pool.remove(best)
        if not progressed:
            break
    # label faster: same hint together, similar frames next to each other
    chosen.sort(key=lambda t: (cands[t[0]].hint or "~", int(assign[t[0]])))
    return [(cands[i].id, why) for i, why in chosen]
