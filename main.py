# main.py
"""
LLM Data Preparation Pipeline (Sequential Version)
- Streaming input (low memory)
- Batch processing for efficiency
- Clean separation of stages
- Shuffling & shard export
- Drop reason tracking

This version avoids multiprocessing for simplicity,
stability, and easier debugging.
"""

import os
import json
import time
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# optional accelerated JSON
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    import json as orjson
    USE_ORJSON = False
    print("⚠ orjson not installed — using Python json.")

# your cleaning module
from mod.data_cleaning import process_batch


# =====================================================
# Configuration
# =====================================================
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

MIN_TOKENS = 10
MAX_TOKENS = 1000

BATCH_SIZE = 1000
SHARD_SIZE = 5000

random.seed(42)  # reproducible shuffle, IYKYK


# =====================================================
# Helpers
# =====================================================

def load_jsonl_streaming(file_path):
    """Efficient generator for JSONL."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if USE_ORJSON:
                    yield orjson.loads(line)
                else:
                    yield json.loads(line)
            except Exception:
                continue  # malformed


def get_input_files():
    raw = Path(RAW_DIR)
    files = list(raw.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files in {RAW_DIR}")
    return files


def write_shard(records, shard_idx):
    """Save shard to disk."""
    out_path = Path(PROCESSED_DIR) / f"shard_{shard_idx:04d}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            if USE_ORJSON:
                f.write(orjson.dumps(rec).decode("utf-8"))
            else:
                json.dump(rec, f)
            f.write("\n")
    return out_path


# =====================================================
# Stage 1 — Load & Light Filter
# =====================================================

def stage1_load_light_filter(input_files):
    """
    Loads records in a streaming fashion and applies fast filters:
    - remove empty texts
    - basic structural checks
    Returns a list of valid raw records.
    """
    print("\n" + "="*60)
    print("STAGE 1 — Load & Light Filter")
    print("="*60)

    valid_records = []
    empty = 0
    total = 0

    for file_path in input_files:
        print(f"\n📄 Reading {file_path.name}")

        for rec in tqdm(load_jsonl_streaming(file_path),
                        desc="Loading", unit="rec"):

            total += 1
            text = rec.get("text", "").strip()

            if not text:
                empty += 1
                continue

            valid_records.append(rec)

    print(f"\n✓ Loaded {total:,} records")
    print(f"✓ Kept {len(valid_records):,} after removing empty")
    print(f"✗ Removed empty texts: {empty:,}")

    return valid_records


# =====================================================
# Stage 2 — Heavy Cleaning (Sequential Batches)
# =====================================================

def stage2_heavy_processing(records):
    """
    Sequential batch processing:
    - language detection
    - PII scrub/replace
    - tokenization
    - length filtering
    - deduplication (inside cleaning module)
    """
    print("\n" + "="*60)
    print("STAGE 2 — Heavy Processing")
    print("="*60)

    final_records = []
    global_drop = Counter()

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]

        processed, drop_reasons = process_batch(
            batch,
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS,
            do_pii=True
        )

        final_records.extend(processed)

        for k, v in drop_reasons.items():
            global_drop[k] += v

    print(f"\n✓ Heavy processing complete")
    print(f"✓ Final kept: {len(final_records):,}")
    print(f"✓ Dropped total: {sum(global_drop.values()):,}")

    return final_records, global_drop


# =====================================================
# Stage 3 — Shuffle & Shard
# =====================================================

def stage3_shuffle_and_shard(records):
    print("\n" + "="*60)
    print("STAGE 3 — Shuffle & Shard Export")
    print("="*60)

    random.shuffle(records)
    print(f"🔀 Shuffled {len(records):,} records")

    shard_idx = 0
    shard_buf = []

    for rec in tqdm(records, desc="Writing shards", unit="rec"):
        shard_buf.append(rec)

        if len(shard_buf) >= SHARD_SIZE:
            write_shard(shard_buf, shard_idx)
            shard_idx += 1
            shard_buf = []

    if shard_buf:
        write_shard(shard_buf, shard_idx)
        shard_idx += 1

    print(f"\n✓ Created {shard_idx} shards")

    return shard_idx


# =====================================================
# Main
# =====================================================

def main():
    start = time.time()

    input_files = get_input_files()
    print("\nFound files:")
    for f in input_files:
        print(f"  • {f.name}")

    # Stage 1 — Light Filtering
    raw_records = stage1_load_light_filter(input_files)

    # Stage 2 — Heavy Cleaning
    cleaned_records, drop_reasons = stage2_heavy_processing(raw_records)

    # Stage 3 — Export
    num_shards = stage3_shuffle_and_shard(cleaned_records)

    # Save metadata
    drop_path = Path(PROCESSED_DIR) / "drop_reasons.json"
    json.dump(drop_reasons, open(drop_path, "w"), indent=2)

    summary = {
        "records_after_cleaning": len(cleaned_records),
        "shards": num_shards,
        "shard_size": SHARD_SIZE,
        "drop_reasons": drop_reasons,
        "runtime_sec": round(time.time()-start, 2)
    }
    json.dump(summary, open(Path(PROCESSED_DIR) / "pipeline_summary.json", "w"), indent=2)

    print("\n" + "="*60)
    print("✨ PIPELINE COMPLETE")
    print("="*60)
    print(f"Time: {summary['runtime_sec']} sec")
    print(f"Records: {summary['records_after_cleaning']:,}")
    print(f"Shards: {num_shards}")
    print("Drop reasons saved to:", drop_path)


if __name__ == "__main__":
    main()
