"""Build the n=30 baseline fixture (15 SWE-bench Lite + 15 SWE-bench Verified
multi-file).

Requires `datasets` from HuggingFace:
    python3 -m venv /tmp/eval_venv
    /tmp/eval_venv/bin/pip install datasets
    /tmp/eval_venv/bin/python build_n30_fixtures.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter

try:
    from datasets import load_dataset
except ImportError:
    sys.stderr.write("install datasets: pip install datasets\n")
    sys.exit(1)

random.seed(42)


def n_files(patch: str) -> int:
    return len(re.findall(r"^\+\+\+ b/", patch, re.MULTILINE))


def to_record(r: dict) -> dict:
    f2p = r.get("FAIL_TO_PASS") or []
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except Exception:
            f2p = [f2p]
    p2p = r.get("PASS_TO_PASS") or []
    if isinstance(p2p, str):
        try:
            p2p = json.loads(p2p)
        except Exception:
            p2p = [p2p]
    return {
        "instance_id": r["instance_id"],
        "repo": r["repo"],
        "base_commit": r["base_commit"],
        "problem_statement": r["problem_statement"],
        "patch": r["patch"],
        "test_patch": r.get("test_patch") or "",
        "FAIL_TO_PASS": list(f2p),
        "PASS_TO_PASS": list(p2p),
    }


def main():
    print("loading SWE-bench_Lite (test split)...")
    lite = list(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"))
    print("loading SWE-bench_Verified (test split)...")
    ver = list(load_dataset("princeton-nlp/SWE-bench_Verified", split="test"))

    lite_sample = random.sample(lite, 15)
    ver_multi = [r for r in ver if n_files(r["patch"]) >= 2]
    by_n = {n: [r for r in ver_multi if n_files(r["patch"]) == n] for n in (2, 3, 4)}
    by_n[5] = [r for r in ver_multi if n_files(r["patch"]) >= 5]
    ver_sample = (
        random.sample(by_n[2], min(8, len(by_n[2])))
        + random.sample(by_n[3], min(4, len(by_n[3])))
        + random.sample(by_n[4], min(2, len(by_n[4])))
        + random.sample(by_n[5], min(1, len(by_n[5])))
    )

    print(f"\nLite repo mix: {dict(Counter(r['repo'] for r in lite_sample))}")
    print(f"Verified n_files: {dict(Counter(n_files(r['patch']) for r in ver_sample))}")
    print(f"Verified repo mix: {dict(Counter(r['repo'] for r in ver_sample))}")

    with open("baseline_n30_lite.jsonl", "w") as f:
        for r in lite_sample:
            f.write(json.dumps(to_record(r)) + "\n")
    with open("baseline_n30_verified.jsonl", "w") as f:
        for r in ver_sample:
            f.write(json.dumps(to_record(r)) + "\n")
    print("\nwrote baseline_n30_{lite,verified}.jsonl")


if __name__ == "__main__":
    main()
