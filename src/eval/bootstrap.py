"""Paired bootstrap 95% CIs of (run - base) for every numeric per-prompt metric.

usage: python src/eval/bootstrap.py BASE_DIR RUN_DIR...
"""
import json
import random
import sys


def load(d):
    return {r["caption"]: r for r in json.load(open(f"{d}/metrics.json"))["rows"]}


def paired_ci(a, b, n_boot=2000, seed=0):
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main(base_dir, *run_dirs):
    base = load(base_dir)
    for run_dir in run_dirs:
        run = load(run_dir)
        keys = [c for c in base if c in run]
        print(f"{run_dir}  (n={len(keys)} vs {base_dir})")
        for m, v in run[keys[0]].items():
            if m == "caption":
                continue
            pairs = [(float(base[c][m]), float(run[c][m])) for c in keys if base[c].get(m) is not None and run[c][m] is not None]
            if not pairs:
                continue
            delta, lo, hi = paired_ci(*zip(*pairs))
            star = "" if lo <= 0 <= hi else " *"
            mb = sum(p[0] for p in pairs) / len(pairs)
            print(f"  {m:10s} base {mb:7.3f}  Δ {delta:+7.3f}  [{lo:+.3f}, {hi:+.3f}]{star}")


if __name__ == "__main__":
    assert paired_ci([0.0] * 10, [1.0] * 10) == (1.0, 1.0, 1.0)
    main(*sys.argv[1:])
