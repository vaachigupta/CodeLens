"""Autoencoder novelty score (Unit I: autoencoders). Offline, numpy only, no API, no torch.

Idea: train a small autoencoder (384 -> 16 -> 384) on the MiniLM embeddings of the functions
in the codebase. It learns to reconstruct "normal" functions of THIS codebase. A new function
that reconstructs badly (high error) is unlike what the codebase contains.

    embedding x --encoder--> code z (16 numbers) --decoder--> reconstruction x'
    error = mean((x - x')^2)

Scoring: 5-fold cross-validation gives honest reconstruction errors for functions the model
did NOT train on (the "normal" error distribution). A new function's novelty = the share of
those normal errors that are smaller than its own error (0.0 = very typical, 1.0 = more
unusual than every normal function).

What it can and cannot tell you: embeddings capture what code is ABOUT, not naming style, so
this is expected to flag off-topic code, not convention violations. Run both and report what
you actually find (see --foreign).

Usage:
  python autoencoder_novelty.py --repo <repo>
  python autoencoder_novelty.py --repo <repo> --cases heldout_cases
  python autoencoder_novelty.py --repo <repo> --foreign <other_repo>   # off-topic control
"""
import argparse
import json
import os
import random
import sys

import numpy as np

from codelens import collect_repo_chunks, embed_texts


class Autoencoder:
    """One hidden (bottleneck) layer: x -> tanh(xW1+b1) -> zW2+b2. Trained by Adam on MSE."""

    def __init__(self, dim, hidden=16, seed=0):
        r = np.random.RandomState(seed)
        self.p = {"W1": r.randn(dim, hidden) * 0.1, "b1": np.zeros(hidden),
                  "W2": r.randn(hidden, dim) * 0.1, "b2": np.zeros(dim)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def encode(self, X):
        return np.tanh(X @ self.p["W1"] + self.p["b1"])

    def reconstruct(self, X):
        return self.encode(X) @ self.p["W2"] + self.p["b2"]

    def errors(self, X):
        return ((X - self.reconstruct(X)) ** 2).mean(axis=1)

    def fit(self, X, epochs=400, lr=0.01):
        n = len(X)
        for _ in range(epochs):
            Z = self.encode(X)
            D = (Z @ self.p["W2"] + self.p["b2"] - X) * (2.0 / X.shape[1] / n)   # dLoss/dOutput
            dZ = (D @ self.p["W2"].T) * (1 - Z ** 2)                            # back through tanh
            grads = {"W2": Z.T @ D, "b2": D.sum(0), "W1": X.T @ dZ, "b1": dZ.sum(0)}
            self.t += 1
            for k, g in grads.items():                                          # Adam update
                self.m[k] = 0.9 * self.m[k] + 0.1 * g
                self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
                mh = self.m[k] / (1 - 0.9 ** self.t)
                vh = self.v[k] / (1 - 0.999 ** self.t)
                self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return self


def auc(pos, neg):
    """P(random positive scores higher than random negative); 0.5 = no better than chance."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def cv_errors(X, folds=5, seed=0, hidden=16):
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    out = np.zeros(len(X))
    for f in range(folds):
        test = idx[f::folds]
        train = [i for i in idx if i not in set(test)]
        ae = Autoencoder(X.shape[1], hidden, seed).fit(X[train])
        out[test] = ae.errors(X[test])
    return out


def novelty(err, normal_errors):
    return float((np.asarray(normal_errors) < err).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cases", default="test_cases")
    ap.add_argument("--foreign", help="another codebase, used as an off-topic control")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    chunks, _, _ = collect_repo_chunks(args.repo)
    if len(chunks) < 10:
        sys.exit("Need at least 10 functions in --repo.")
    X = np.array(embed_texts([c["source"] for c in chunks]))
    mu = X.mean(0)
    Xc = X - mu
    print(f"Training on {len(chunks)} functions from {args.repo} (hidden size {args.hidden}) ...")

    normal = cv_errors(Xc, hidden=args.hidden)
    ae = Autoencoder(Xc.shape[1], args.hidden).fit(Xc)
    train_err = ae.errors(Xc)

    with open(os.path.join(args.cases, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    codes = [open(os.path.join(args.cases, c["file"]), encoding="utf-8", errors="replace").read() for c in manifest]
    T = np.array(embed_texts(codes)) - mu
    errs = ae.errors(T)
    scores = [novelty(e, normal) for e in errs]

    dev = [s for s, c in zip(scores, manifest) if c["violations"]]
    conf = [s for s, c in zip(scores, manifest) if not c["violations"]]
    lines = ["# Autoencoder novelty evaluation",
             f"Repo: `{args.repo}` ({len(chunks)} functions) | cases: {args.cases} | bottleneck: {args.hidden}", "",
             f"- Mean reconstruction error on training functions: {train_err.mean():.5f}",
             f"- Mean error on held-out folds (normal, cross-validated): {normal.mean():.5f}",
             f"- Mean error on the test files: {errs.mean():.5f}", "",
             "| File | Violations | Error | Novelty (0-1) |", "|---|---|---|---|"]
    for c, e, s in zip(manifest, errs, scores):
        lines.append(f"| {c['file']} | {', '.join(c['violations']) or '-'} | {e:.5f} | {s:.2f} |")
    lines += ["", f"Mean novelty: conformant {np.mean(conf):.2f}, deviant {np.mean(dev):.2f}",
              f"AUC deviant-vs-conformant: {auc(dev, conf):.2f}  (0.50 = the score cannot tell them apart)"]

    if args.foreign:
        fch, _, _ = collect_repo_chunks(args.foreign)
        F = np.array(embed_texts([c["source"] for c in fch])) - mu
        fs = [novelty(e, normal) for e in ae.errors(F)]
        lines += ["", f"Off-topic control: {len(fch)} functions from `{args.foreign}`",
                  f"Mean novelty: foreign {np.mean(fs):.2f} vs the repo's own held-out folds "
                  f"{np.mean([novelty(e, normal) for e in normal]):.2f}",
                  f"AUC foreign-vs-own: {auc(fs, [novelty(e, normal) for e in normal]):.2f}"]
    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"autoencoder_{os.path.basename(os.path.normpath(args.cases))}.md")
    open(out, "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
