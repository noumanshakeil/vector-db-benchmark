"""Train/eval entry for vector-db-benchmark."""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_classification(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return X, y

def make_series(n=500, seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20, n)
    y = np.sin(t) + rng.normal(scale=0.15, size=n)
    idx = rng.choice(n, size=12, replace=False)
    y[idx] += rng.choice([-1, 1], size=12) * rng.uniform(2.5, 4.0, size=12)
    return y, idx

def run_tabular() -> None:
    X, y = make_classification()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=0)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(classification_report(yte, pred))

def run_anomaly() -> None:
    y, injected = make_series()
    X = y.reshape(-1, 1)
    model = IsolationForest(contamination=0.05, random_state=0)
    labels = model.fit_predict(X)
    flagged = set(np.where(labels == -1)[0].tolist())
    hits = len(flagged & set(injected.tolist()))
    print(f"Injected anomalies: {len(injected)} | Detected overlap: {hits}")
    print(f"Flagged indices (first 20): {sorted(flagged)[:20]}")

def run_cf() -> None:
    rng = np.random.default_rng(0)
    R = rng.integers(0, 6, size=(12, 10)).astype(float)
    R[R == 0] = np.nan
    col_mean = np.nanmean(R, axis=0)
    inds = np.where(np.isnan(R))
    R[inds] = np.take(col_mean, inds[1])
    nn = NearestNeighbors(metric="cosine", n_neighbors=3)
    nn.fit(R)
    dists, idxs = nn.kneighbors(R[0:1])
    print("User-0 neighbor indices:", idxs[0].tolist())
    print("Distances:", np.round(dists[0], 3).tolist())
    print("Recommended item scores:", np.round(R[idxs[0]].mean(axis=0), 2))

def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark FAISS vs simple NumPy cosine search on synthetic embeddings (latency/recall).")
    p.add_argument("--mode", choices=["tabular", "anomaly", "cf"], default="tabular")
    args = p.parse_args()
    {"tabular": run_tabular, "anomaly": run_anomaly, "cf": run_cf}[args.mode]()

if __name__ == "__main__":
    main()
