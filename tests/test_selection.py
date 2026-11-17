from datetime import datetime, timedelta

import numpy as np

from adbot2.selection import Candidate, kmeans, knn_probs, pick

T0 = datetime(2026, 11, 15)


def blobs(sizes, dim=16, seed=0):
    """Unit vectors in tight clusters around random centres."""
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        c = rng.normal(size=dim)
        out.append(c + 0.05 * rng.normal(size=(n, dim)))
    x = np.concatenate(out)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_kmeans_finds_blobs():
    x = blobs([30, 30, 30])
    a = kmeans(x, 3)
    assert len({a[0], a[30], a[60]}) == 3
    assert len(set(a[:30])) == 1


def test_knn_probs():
    x = blobs([10, 10])
    p = knn_probs(x[[0, 15]], x[np.r_[1:10, 11:20]], ["a"] * 9 + ["b"] * 9, ["a", "b"], k=5)
    assert p[0, 0] > 0.9 and p[1, 1] > 0.9


def test_pick_reaches_rare_groups_and_spreads_over_clusters():
    x = blobs([200, 200, 5])
    cands = [Candidate(f"f{i}", T0 + timedelta(seconds=30 * i), "d", "fight" if i < 400 else "results")
             for i in range(405)]
    out = pick(cands, x, {}, n=20, clusters=3)
    ids = [i for i, _ in out]
    assert len(ids) == 20 == len(set(ids))
    assert sum(1 for i in ids if int(i[1:]) >= 400) >= 3      # the rare hint is not drowned out
    assert sum(1 for i in ids if int(i[1:]) < 200) >= 4       # both big clusters are used
    assert sum(1 for i in ids if 200 <= int(i[1:]) < 400) >= 4


def test_pick_skips_labelled_and_close_in_time():
    x = blobs([50])
    cands = [Candidate(f"f{i}", T0 + timedelta(seconds=i), "d", None) for i in range(50)]
    out = pick(cands, x, {"f0": "lobby"}, n=10, clusters=2, min_gap_s=10)
    ids = sorted(int(i[1:]) for i, _ in out)
    assert 0 not in ids
    assert ids[0] >= 10 and all(b - a >= 10 for a, b in zip(ids, ids[1:]))


def test_pick_asks_about_frames_the_labels_disagree_on():
    x = blobs([40, 40])
    cands = [Candidate(f"f{i}", None, "d", "lobby") for i in range(80)]
    labels = {f"f{i}": "lobby" for i in range(15)} | {f"f{i}": "bag" for i in range(40, 55)}
    out = pick(cands, x, labels, n=10, clusters=2)
    assert any(why.startswith("disagrees") and int(i[1:]) >= 55 for i, why in out)
