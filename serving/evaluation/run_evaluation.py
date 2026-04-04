#!/usr/bin/env python3
"""Orchestration script for the full GNN recommendation evaluation suite.

Workflow
--------
1. Verify that every target endpoint is reachable and healthy.
2. Run ``benchmark.py`` against each serving option at multiple concurrency
   levels.
3. Collect, format, and persist results (console table + CSV + JSON).

Configuration
-------------
All options can be set via CLI arguments **or** environment variables:

  BASELINE_URL         URL for the baseline endpoint
  OPTIMIZED_URL        URL for the optimized (Redis) endpoint
  FURTHER_URL          URL for the further-optimized endpoint
  BENCHMARK_DURATION   Seconds per concurrency level  (default 30)
  USER_POOL_SIZE       Number of synthetic user ids    (default 1000)
  OUTPUT_DIR           Where to write CSV/JSON         (default ./results)
  MODEL_VERSION        Label for model version column  (default v1.0)
  CODE_VERSION         Label for code version column   (default HEAD)
  HARDWARE             Hardware description             (default cpu)
  COMPUTE_INSTANCE     Compute instance type            (default chameleon-m1.medium)

Usage::

    python run_evaluation.py \
        --baseline   http://localhost:8000 \
        --optimized  http://localhost:8001 \
        --further    http://localhost:8002

    # Or with env vars
    BASELINE_URL=http://baseline:8000 \
    OPTIMIZED_URL=http://optimized:8000 \
    python run_evaluation.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import httpx

# Local imports — benchmark.py lives next to this file.
from benchmark import (
    BenchmarkResult,
    async_main as benchmark_async_main,
    check_health,
    print_results_table,
    save_csv,
    save_json,
    CONCURRENCY_LEVELS,
    DURATION_SECONDS,
)


# ---------------------------------------------------------------------------
# Defaults from environment
# ---------------------------------------------------------------------------

_DEFAULT_BASELINE = os.environ.get("BASELINE_URL", "")
_DEFAULT_OPTIMIZED = os.environ.get("OPTIMIZED_URL", "")
_DEFAULT_FURTHER = os.environ.get("FURTHER_URL", "")
_DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./results")
_DEFAULT_DURATION = int(os.environ.get("BENCHMARK_DURATION", str(DURATION_SECONDS)))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate the full evaluation of GNN recommendation endpoints",
    )
    parser.add_argument(
        "--baseline",
        default=_DEFAULT_BASELINE,
        help="Baseline endpoint URL (env: BASELINE_URL)",
    )
    parser.add_argument(
        "--optimized",
        default=_DEFAULT_OPTIMIZED,
        help="Optimized (Redis cache) endpoint URL (env: OPTIMIZED_URL)",
    )
    parser.add_argument(
        "--further",
        default=_DEFAULT_FURTHER,
        help="Further-optimized endpoint URL (env: FURTHER_URL)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=_DEFAULT_DURATION,
        help=f"Seconds per concurrency level (default {_DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=CONCURRENCY_LEVELS,
        help=f"Concurrency levels to test (default {CONCURRENCY_LEVELS})",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory for output artefacts (default {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip the initial health-check gate",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Health-check phase
# ---------------------------------------------------------------------------

async def verify_endpoints(
    options: list[tuple[str, str]],
    *,
    strict: bool = True,
) -> list[tuple[str, str]]:
    """Check /health for every endpoint and return the healthy subset.

    When *strict* is True and any endpoint fails, the function raises
    ``SystemExit`` so the evaluation aborts early.
    """
    healthy: list[tuple[str, str]] = []

    for name, url in options:
        ok = await check_health(url, retries=3)
        status = "HEALTHY" if ok else "UNREACHABLE"
        print(f"[eval] {name:>20s}  {url:40s}  {status}")
        if ok:
            healthy.append((name, url))
        elif strict:
            print(
                f"[eval] ERROR: {name} ({url}) is not reachable. "
                "Aborting. Use --skip-health to bypass.",
                file=sys.stderr,
            )
            sys.exit(1)

    return healthy


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def orchestrate(args: argparse.Namespace) -> list[BenchmarkResult]:
    """Run the full evaluation pipeline and return collected results."""

    # 1. Build option list from arguments
    options: list[tuple[str, str]] = []
    if args.baseline:
        options.append(("baseline", args.baseline))
    if args.optimized:
        options.append(("optimized", args.optimized))
    if args.further:
        options.append(("further-optimized", args.further))

    if not options:
        print(
            "[eval] ERROR: supply at least one endpoint via "
            "--baseline, --optimized, or --further (or env vars).",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Health checks
    if not args.skip_health:
        print("[eval] --- Health-check phase ---")
        options = await verify_endpoints(options, strict=True)
    else:
        print("[eval] Skipping health checks (--skip-health)")

    # 3. Benchmark phase
    print(f"\n[eval] --- Benchmark phase ---")
    print(
        f"[eval] Options: {len(options)}  |  "
        f"Concurrency levels: {args.concurrency}  |  "
        f"Duration per level: {args.duration}s"
    )
    total_est = len(options) * len(args.concurrency) * args.duration
    print(f"[eval] Estimated wall time: ~{total_est}s ({total_est // 60}m {total_est % 60}s)\n")

    # Build a Namespace that benchmark.async_main expects
    from types import SimpleNamespace

    all_results: list[BenchmarkResult] = []

    for name, url in options:
        print(f"\n[eval] ====== Benchmarking: {name} ({url}) ======")
        bench_args = SimpleNamespace(
            url=None,
            baseline=url if name == "baseline" else None,
            optimized=url if name == "optimized" else None,
            further=url if name == "further-optimized" else None,
            duration=args.duration,
            concurrency=args.concurrency,
            output_dir=args.output_dir,
            csv=f"benchmark_{name}.csv",
            json_out=f"benchmark_{name}.json",
        )

        # Reconstruct the args so benchmark sees exactly one option
        bm_args = SimpleNamespace(
            url=url,
            baseline=None,
            optimized=None,
            further=None,
            duration=args.duration,
            concurrency=args.concurrency,
            output_dir=args.output_dir,
            csv=f"benchmark_{name}.csv",
            json_out=f"benchmark_{name}.json",
        )

        results = await benchmark_async_main(bm_args)
        # Patch the option name so it reads correctly in the combined table
        for r in results:
            r.option = name
        all_results.extend(results)

    # 4. Combined output
    print("\n[eval] ====== Combined Results ======")
    print_results_table(all_results)

    os.makedirs(args.output_dir, exist_ok=True)
    combined_csv = os.path.join(args.output_dir, "evaluation_results.csv")
    combined_json = os.path.join(args.output_dir, "evaluation_results.json")
    save_csv(all_results, combined_csv)
    save_json(all_results, combined_json)

    # Summary
    print("\n[eval] --- Summary ---")
    for name, url in options:
        subset = [r for r in all_results if r.option == name]
        if not subset:
            continue
        best_tp = max(r.throughput_rps for r in subset)
        best_p50 = min(r.p50_latency_ms for r in subset)
        print(
            f"  {name:>20s}  best throughput={best_tp} rps  "
            f"best p50={best_p50}ms"
        )

    print(f"\n[eval] Artefacts written to {args.output_dir}/")
    return all_results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(orchestrate(args))


if __name__ == "__main__":
    main()
