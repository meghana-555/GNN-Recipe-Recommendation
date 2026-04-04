#!/usr/bin/env python3
"""Comprehensive benchmark for the GNN recipe recommendation API.

Measures latency percentiles, throughput, error rate, and response sizes
at varying concurrency levels using ``httpx`` with ``asyncio``.

Usage::

    # Single endpoint
    python benchmark.py --url http://localhost:8000

    # Compare three serving options
    python benchmark.py \
        --baseline   http://localhost:8000 \
        --optimized  http://localhost:8001 \
        --further    http://localhost:8002

Environment variables
---------------------
BENCHMARK_DURATION   : seconds per concurrency level (default 30)
USER_POOL_SIZE       : number of synthetic user ids (default 1000)
MODEL_VERSION        : label for the model column (default "v1.0")
CODE_VERSION         : label for the code column  (default "HEAD")
HARDWARE             : hardware description       (default "cpu")
COMPUTE_INSTANCE     : cloud instance type         (default "chameleon-m1.medium")
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx
import numpy as np
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONCURRENCY_LEVELS: list[int] = [1, 5, 10, 20]

DURATION_SECONDS: int = int(os.environ.get("BENCHMARK_DURATION", "30"))
USER_POOL_SIZE: int = int(os.environ.get("USER_POOL_SIZE", "1000"))
MODEL_VERSION: str = os.environ.get("MODEL_VERSION", "v1.0")
CODE_VERSION: str = os.environ.get("CODE_VERSION", "HEAD")
HARDWARE: str = os.environ.get("HARDWARE", "cpu")
COMPUTE_INSTANCE: str = os.environ.get("COMPUTE_INSTANCE", "chameleon-m1.medium")
REQUEST_TIMEOUT: float = float(os.environ.get("REQUEST_TIMEOUT", "10.0"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Outcome of a single HTTP request."""
    latency_ms: float
    status_code: int
    response_size_bytes: int
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated statistics for one (option, concurrency) run."""
    option: str
    endpoint_url: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    throughput_rps: float
    error_rate: float
    avg_response_size_bytes: float
    model_version: str = MODEL_VERSION
    code_version: str = CODE_VERSION
    hardware: str = HARDWARE
    compute_instance: str = COMPUTE_INSTANCE
    notes: str = ""


# ---------------------------------------------------------------------------
# Worker coroutine
# ---------------------------------------------------------------------------

async def _worker(
    client: httpx.AsyncClient,
    base_url: str,
    results: list[RequestResult],
    stop_event: asyncio.Event,
    rng: np.random.Generator,
) -> None:
    """Continuously send POST /recommend until *stop_event* is set."""
    while not stop_event.is_set():
        user_id = f"user:{rng.integers(0, USER_POOL_SIZE)}"
        payload = {"user_id": user_id}

        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/recommend",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results.append(
                RequestResult(
                    latency_ms=elapsed_ms,
                    status_code=resp.status_code,
                    response_size_bytes=len(resp.content),
                    error=None if resp.status_code == 200 else resp.text[:200],
                )
            )
        except (httpx.HTTPError, asyncio.CancelledError) as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results.append(
                RequestResult(
                    latency_ms=elapsed_ms,
                    status_code=0,
                    response_size_bytes=0,
                    error=str(exc)[:200],
                )
            )

        # Brief yield so the event-loop can service other coroutines and
        # check the stop event promptly.
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Run a single benchmark pass
# ---------------------------------------------------------------------------

async def run_benchmark(
    option_name: str,
    base_url: str,
    concurrency: int,
    duration: int = DURATION_SECONDS,
) -> BenchmarkResult:
    """Benchmark *base_url* at the given *concurrency* for *duration* seconds."""
    results: list[RequestResult] = []
    stop_event = asyncio.Event()

    rng = np.random.default_rng()

    limits = httpx.Limits(
        max_connections=concurrency + 10,
        max_keepalive_connections=concurrency + 5,
    )

    async with httpx.AsyncClient(limits=limits) as client:
        workers = [
            asyncio.create_task(
                _worker(client, base_url, results, stop_event, rng)
            )
            for _ in range(concurrency)
        ]

        await asyncio.sleep(duration)
        stop_event.set()

        # Give workers a moment to finish in-flight requests
        done, pending = await asyncio.wait(workers, timeout=REQUEST_TIMEOUT + 2)
        for t in pending:
            t.cancel()

    if not results:
        return BenchmarkResult(
            option=option_name,
            endpoint_url=base_url,
            concurrency=concurrency,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            duration_seconds=duration,
            p50_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            mean_latency_ms=0,
            throughput_rps=0,
            error_rate=1.0,
            avg_response_size_bytes=0,
            notes="No requests completed",
        )

    latencies = [r.latency_ms for r in results]
    successes = [r for r in results if r.status_code == 200]
    failures = [r for r in results if r.status_code != 200]

    latencies_arr = np.array(latencies)
    sizes = [r.response_size_bytes for r in results if r.response_size_bytes > 0]

    return BenchmarkResult(
        option=option_name,
        endpoint_url=base_url,
        concurrency=concurrency,
        total_requests=len(results),
        successful_requests=len(successes),
        failed_requests=len(failures),
        duration_seconds=duration,
        p50_latency_ms=round(float(np.percentile(latencies_arr, 50)), 2),
        p95_latency_ms=round(float(np.percentile(latencies_arr, 95)), 2),
        p99_latency_ms=round(float(np.percentile(latencies_arr, 99)), 2),
        mean_latency_ms=round(float(np.mean(latencies_arr)), 2),
        throughput_rps=round(len(results) / duration, 2),
        error_rate=round(len(failures) / len(results), 4),
        avg_response_size_bytes=round(statistics.mean(sizes), 1) if sizes else 0,
    )


# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------

async def check_health(base_url: str, retries: int = 3) -> bool:
    """Return True if the /health endpoint reports healthy."""
    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            try:
                resp = await client.get(
                    f"{base_url}/health", timeout=5.0
                )
                if resp.status_code == 200 and resp.json().get("status") == "healthy":
                    return True
            except httpx.HTTPError:
                pass
            if attempt < retries - 1:
                await asyncio.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _table_row(r: BenchmarkResult) -> dict:
    """Return a dict suitable for tabulate display."""
    return {
        "Option": r.option,
        "Endpoint URL": r.endpoint_url,
        "Model version": r.model_version,
        "Code version": r.code_version,
        "Hardware": r.hardware,
        "Concurrency": r.concurrency,
        "p50 (ms)": r.p50_latency_ms,
        "p95 (ms)": r.p95_latency_ms,
        "p99 (ms)": r.p99_latency_ms,
        "Throughput (rps)": r.throughput_rps,
        "Error rate": f"{r.error_rate:.2%}",
        "Compute instance": r.compute_instance,
        "Notes": r.notes,
    }


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Pretty-print results to stdout."""
    rows = [_table_row(r) for r in results]
    print("\n" + tabulate(rows, headers="keys", tablefmt="grid"))


def save_csv(results: list[BenchmarkResult], path: str) -> None:
    """Write results to a CSV file."""
    if not results:
        return
    rows = [asdict(r) for r in results]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[benchmark] CSV saved to {path}")


def save_json(results: list[BenchmarkResult], path: str) -> None:
    """Write results to a JSON file."""
    payload = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds_per_level": DURATION_SECONDS,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[benchmark] JSON saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GNN recipe recommendation endpoints",
    )
    parser.add_argument(
        "--url",
        help="Single endpoint URL to benchmark (shorthand when only one option)",
    )
    parser.add_argument(
        "--baseline",
        help="Baseline endpoint URL (plain FastAPI + in-memory cache)",
    )
    parser.add_argument(
        "--optimized",
        help="Optimized endpoint URL (FastAPI + Redis cache)",
    )
    parser.add_argument(
        "--further",
        help="Further-optimized endpoint URL (compression / connection pooling)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DURATION_SECONDS,
        help=f"Seconds per concurrency level (default {DURATION_SECONDS})",
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
        default=".",
        help="Directory for CSV/JSON output files (default: cwd)",
    )
    parser.add_argument(
        "--csv",
        default="benchmark_results.csv",
        help="CSV output filename (default: benchmark_results.csv)",
    )
    parser.add_argument(
        "--json-out",
        default="benchmark_results.json",
        help="JSON output filename (default: benchmark_results.json)",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> list[BenchmarkResult]:
    # Build the list of (name, url) options to benchmark
    options: list[tuple[str, str]] = []
    if args.url:
        options.append(("single", args.url))
    if args.baseline:
        options.append(("baseline", args.baseline))
    if args.optimized:
        options.append(("optimized", args.optimized))
    if args.further:
        options.append(("further-optimized", args.further))

    if not options:
        print(
            "[benchmark] ERROR: provide at least one endpoint via "
            "--url, --baseline, --optimized, or --further",
            file=sys.stderr,
        )
        sys.exit(1)

    # Health checks
    for name, url in options:
        healthy = await check_health(url)
        if healthy:
            print(f"[benchmark] {name} ({url}) is healthy")
        else:
            print(
                f"[benchmark] WARNING: {name} ({url}) health check failed; "
                "proceeding anyway",
                file=sys.stderr,
            )

    all_results: list[BenchmarkResult] = []
    total_runs = len(options) * len(args.concurrency)
    run_idx = 0

    for name, url in options:
        for conc in args.concurrency:
            run_idx += 1
            print(
                f"\n[benchmark] [{run_idx}/{total_runs}] "
                f"option={name}  url={url}  concurrency={conc}  "
                f"duration={args.duration}s"
            )
            result = await run_benchmark(name, url, conc, args.duration)
            all_results.append(result)
            print(
                f"  => requests={result.total_requests}  "
                f"throughput={result.throughput_rps} rps  "
                f"p50={result.p50_latency_ms}ms  "
                f"p95={result.p95_latency_ms}ms  "
                f"errors={result.error_rate:.2%}"
            )

    # Output
    print_results_table(all_results)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.csv)
    json_path = os.path.join(args.output_dir, args.json_out)
    save_csv(all_results, csv_path)
    save_json(all_results, json_path)

    return all_results


def main(argv: list[str] | None = None) -> list[BenchmarkResult]:
    args = parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
