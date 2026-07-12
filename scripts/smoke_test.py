#!/usr/bin/env python3
"""Smoke test: send 100 questions from ground truth against a running API."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List

import httpx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GT_PATH = os.path.join(ROOT, "evaluation", "ground_truth.jsonl")
BASE_URL = os.getenv("SMOKE_BASE_URL", "http://127.0.0.1:8000")
N = int(os.getenv("SMOKE_N", "100"))


def load_questions(path: str) -> List[str]:
    questions: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                questions.append(json.loads(line)["question"])
    return questions


def main() -> None:
    questions = load_questions(GT_PATH)
    if not questions:
        print("No ground-truth questions found.")
        sys.exit(1)

    # Cycle GT questions to reach N requests.
    schedule = [questions[i % len(questions)] for i in range(N)]

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        if health.status_code != 200 or ready.status_code != 200:
            print(f"Service not ready: health={health.status_code} ready={ready.status_code}")
            sys.exit(1)

        # Validation smoke: empty question must be 422.
        bad = client.post("/ask", json={"question": ""})
        if bad.status_code != 422:
            print(f"Expected 422 for empty question, got {bad.status_code}")
            sys.exit(1)

        ok = 0
        errors = 0
        latencies: List[float] = []
        t0 = time.perf_counter()
        for i, question in enumerate(schedule, start=1):
            start = time.perf_counter()
            try:
                resp = client.post("/ask", json={"question": question, "top_k": 5})
                latencies.append((time.perf_counter() - start) * 1000.0)
                if resp.status_code == 200 and "answer" in resp.json():
                    ok += 1
                else:
                    errors += 1
                    print(f"[{i}] unexpected status={resp.status_code} body={resp.text[:200]}")
            except Exception as exc:
                errors += 1
                print(f"[{i}] exception: {exc}")

        elapsed = time.perf_counter() - t0
        latencies_sorted = sorted(latencies)
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0.0

        print(
            json.dumps(
                {
                    "requests": N,
                    "ok": ok,
                    "errors": errors,
                    "elapsed_sec": round(elapsed, 2),
                    "p95_ms": round(p95, 2),
                },
                indent=2,
            )
        )
        if errors > 0 or ok < N:
            sys.exit(1)


if __name__ == "__main__":
    main()
