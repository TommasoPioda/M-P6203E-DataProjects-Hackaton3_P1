#!/usr/bin/env python3
"""
================================================================================
PARALLEL SPLIT PROCESSOR - Multi-Process Implementation
================================================================================
Process train/validation/test splits in parallel using multiprocessing.Pool
Supports both multiprocessing and threading backends.

Usage:
    python parallel_split_processor.py [--backend {multiprocessing|threading}] [--verbose]
================================================================================
"""

import argparse
import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# =========================
# PROJECT SETUP
# =========================
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "utils").exists():
    for parent in PROJECT_ROOT.parents:
        if (parent / "utils").exists():
            PROJECT_ROOT = parent
            break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_utils import (
    build_training_dataframe,
    load_clean_citation_dataframe_from_files,
)

# =========================
# CONFIGURATION
# =========================
OUTPUT_DIR = PROJECT_ROOT / "data" / "exploded_splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_SPLIT_DIR = PROJECT_ROOT / "data" / "old_data" / "split"

SPLIT_FILES = {
    "train": DATA_SPLIT_DIR / "train.parquet",
    "validation": DATA_SPLIT_DIR / "validation.parquet",
    "test": DATA_SPLIT_DIR / "test.parquet",
}

NUM_WORKERS = min(3, cpu_count())


# =========================
# UTILITIES
# =========================
def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        result["elapsed_time"] = round(elapsed, 2)
        return result

    return wrapper


def print_header(text: str, char: str = "=", width: int = 70):
    """Print formatted header"""
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_separator(char: str = "=", width: int = 70):
    """Print formatted separator"""
    print(f"\n{char * width}\n")


# =========================
# CORE PROCESSING FUNCTION
# =========================
@timer
def process_and_save_split(split_name: str, seed: int) -> Dict:
    """
    Process and save a single dataset split.

    Args:
        split_name: Name of the split ('train', 'validation', 'test')
        seed: Random seed for reproducibility

    Returns:
        Dictionary with processing results and metadata
    """
    try:
        # Load raw data
        raw_df = load_clean_citation_dataframe_from_files(
            [SPLIT_FILES[split_name]], remove_empty_rows=True
        )
        n_raw = len(raw_df)

        if raw_df.empty:
            return {
                "split": split_name,
                "status": "empty",
                "n_raw": 0,
                "n_final": 0,
                "error": None,
            }

        # Build training dataframe
        df = build_training_dataframe(raw_df, seed=seed, filter_years=True).assign(
            split=split_name
        )

        # Clean data
        df.dropna(subset=["ref_id"], inplace=True)
        df.drop(
            columns=["year", "n_citation_ref"], inplace=True, errors="ignore"
        )
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        raw_df[str_cols] = df[str_cols].fillna("")

        n_final = len(df)

        # Save to parquet (I/O operation)
        output_path = OUTPUT_DIR / f"{split_name}_pairs.parquet"
        df.to_parquet(output_path, index=False)

        # Clean memory
        del raw_df, df
        gc.collect()

        return {
            "split": split_name,
            "status": "ok",
            "n_raw": n_raw,
            "n_final": n_final,
            "ratio": round(n_final / n_raw, 3) if n_raw > 0 else 0,
            "output_path": str(output_path),
            "error": None,
        }

    except Exception as e:
        return {
            "split": split_name,
            "status": "error",
            "n_raw": 0,
            "n_final": 0,
            "error": str(e),
        }


# =========================
# PARALLEL EXECUTORS
# =========================
def execute_multiprocessing(split_jobs: List[Tuple]) -> List[Dict]:
    """Execute splits using multiprocessing.Pool"""
    print_header("🚀 MULTIPROCESSING BACKEND")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Splits: {[s[0] for s in split_jobs]}\n")

    start_time = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.starmap(process_and_save_split, split_jobs)

    total_time = time.time() - start_time

    return results, total_time


def execute_threading(split_jobs: List[Tuple]) -> List[Dict]:
    """Execute splits using concurrent.futures.ThreadPoolExecutor"""
    print_header("🧵 THREADING BACKEND")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Splits: {[s[0] for s in split_jobs]}\n")

    start_time = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_and_save_split, name, seed): name
            for name, seed in split_jobs
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    total_time = time.time() - start_time

    return results, total_time


# =========================
# RESULTS DISPLAY
# =========================
def display_results(results: List[Dict], total_time: float, verbose: bool = False):
    """Display processing results"""
    print_separator()

    # Status indicator
    for res in results:
        status_icon = "✓" if res["status"] == "ok" else "✗"
        print(
            f"{status_icon} {res['split'].upper():15} | "
            f"Rows: {res['n_raw']:8,} → {res['n_final']:8,} | "
            f"Time: {res.get('elapsed_time', 'N/A')}s"
        )
        if res["error"]:
            print(f"  ⚠️  ERROR: {res['error']}")

    print_separator()

    # Summary statistics
    summary_df = pd.DataFrame(results)

    print_header("📊 SUMMARY STATISTICS")
    print(summary_df[["split", "status", "n_raw", "n_final", "ratio", "elapsed_time"]].to_string(index=False))

    print_separator()

    successful_splits = summary_df[summary_df["status"] == "ok"]
    if len(successful_splits) > 0:
        print("📈 AGGREGATE METRICS:")
        print(f"  Total Rows (Raw):      {successful_splits['n_raw'].sum():>10,}")
        print(f"  Total Rows (Final):    {successful_splits['n_final'].sum():>10,}")
        print(f"  Avg Compression Ratio: {successful_splits['ratio'].mean():>10.3f}")
        print(f"  Sum Processing Time:   {successful_splits['elapsed_time'].sum():>10.2f}s")
        print(f"  Wall Clock Time:       {total_time:>10.2f}s")
        speedup = successful_splits["elapsed_time"].sum() / total_time
        print(f"  Speedup:               {speedup:>10.2f}x")

    if verbose:
        print_separator()
        print("📁 OUTPUT FILES:")
        for res in results:
            if res.get("output_path"):
                print(f"  ✓ {res['output_path']}")

    print_header("✅ PROCESSING COMPLETE")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Process citation dataset splits in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parallel_split_processor.py --backend multiprocessing
  python parallel_split_processor.py --backend threading --verbose
        """,
    )

    parser.add_argument(
        "--backend",
        choices=["multiprocessing", "threading"],
        default="multiprocessing",
        help="Parallel backend to use (default: multiprocessing)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )

    args = parser.parse_args()

    # Job definition
    split_jobs = [("train", 42), ("validation", 43), ("test", 44)]

    print_header("PARALLEL SPLIT PROCESSOR")
    print(f"Backend: {args.backend.upper()}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Splits: {len(split_jobs)}")

    # Execute based on backend
    if args.backend == "multiprocessing":
        results, total_time = execute_multiprocessing(split_jobs)
    else:  # threading
        results, total_time = execute_threading(split_jobs)

    # Display results
    display_results(results, total_time, verbose=args.verbose)


if __name__ == "__main__":
    main()
