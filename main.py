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
from mod.data_cleaning import process_batch_dual_output  # FIXED: use dual output


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
    """Save shard to disk as JSONL (one JSON object per line)."""
    out_path = Path(PROCESSED_DIR) / f"shard_{shard_idx:04d}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            if USE_ORJSON:
                f.write(orjson.dumps(rec).decode("utf-8"))
            else:
                json.dump(rec, f)
            f.write("\n")  # CRITICAL: newline after each JSON object
    return out_path

def write_text_jsonl(texts):
    """
    Write cleaned text as proper JSONL (one JSON object per line).
    Each line: {"text": "cleaned content here"}
    """
    output_file = Path(PROCESSED_DIR) / "all_processed_text.jsonl"
    
    print(f"\n📄 Writing submission-ready JSONL to {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for text in texts:
            # Each line is a separate JSON object
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")  # CRITICAL: newline after each JSON object
    
    print(f"✅ Written {len(texts):,} records in JSONL format")
    
    # Verify format
    print("\n🔍 Verifying JSONL format...")
    with open(output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:  # Check first 3 lines
                break
            try:
                obj = json.loads(line)
                preview = obj.get("text", "")[:60]
                print(f"  Line {i+1}: {{'text': '{preview}...'}}")
            except json.JSONDecodeError as e:
                print(f"  Line {i+1}: ERROR - {e}")
    
    return output_file


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
    Sequential batch processing - returns BOTH tokens and cleaned text.
    """
    print("\n" + "="*60)
    print("STAGE 2 — Heavy Processing")
    print("="*60)

    final_tokens = []
    final_texts = []
    global_drop = Counter()

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]

        # FIXED: Get BOTH tokens and cleaned text
        tokens_batch, texts_batch, drop_reasons = process_batch_dual_output(
            batch,
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS,
            do_pii=True
        )

        # Both lists have same length and correspond 1:1
        final_tokens.extend(tokens_batch)
        final_texts.extend(texts_batch)

        for k, v in drop_reasons.items():
            global_drop[k] += v

    print(f"\n✓ Heavy processing complete")
    print(f"✓ Final kept: {len(final_tokens):,}")
    print(f"✓ Dropped total: {sum(global_drop.values()):,}")
    
    # Verify they match
    assert len(final_tokens) == len(final_texts), "Token/text mismatch!"

    return final_tokens, final_texts, global_drop


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
# Stage 4 — Combine Shards into single JSONL
# =====================================================

def stage4_combine_shards():
    """
    Combine all shard_*.jsonl files into a single JSONL file.
    Each line is a JSON object with {"text": "..."} format.
    """
    print("\n" + "="*60)
    print("STAGE 4 — Combine Shards into JSONL")
    print("="*60)

    shard_files = sorted(Path(PROCESSED_DIR).glob("shard_*.jsonl"))
    output_file = Path(PROCESSED_DIR) / "all_processed.jsonl"

    total_records = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for shard_file in tqdm(shard_files, desc="Combining shards"):
            with open(shard_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:  # skip empty lines
                        out_f.write(line)
                        out_f.write("\n")
                        total_records += 1

    print(f"✅ Combined {len(shard_files)} shards into {output_file}")
    print(f"✅ Total records in combined file: {total_records:,}")
    
    # Verify JSONL format
    print("\n📋 Verifying JSONL format...")
    with open(output_file, "r", encoding="utf-8") as f:
        sample_lines = [f.readline() for _ in range(3)]
    
    print("Sample lines from output:")
    for i, line in enumerate(sample_lines, 1):
        if line.strip():
            try:
                obj = json.loads(line)
                print(f"  Line {i}: {{'text': '{obj.get('text', '')[:50]}...'}}")
            except:
                print(f"  Line {i}: ERROR parsing JSON")
    
    return output_file


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

    # Stage 2 — Heavy Cleaning (returns BOTH tokens and text)
    tokenized_records, cleaned_texts, drop_reasons = stage2_heavy_processing(raw_records)

    # Write submission-ready JSONL file
    text_output_file = write_text_jsonl(cleaned_texts)

    # FIXED: Create records from tokenized data for sharding
    cleaned_records = [{"tokens": tokens} for tokens in tokenized_records]

    # Stage 3 — Export to shards
    num_shards = stage3_shuffle_and_shard(cleaned_records)

    # Stage 4 — Combine into single JSONL
    combined_file = stage4_combine_shards()

    # Save metadata
    drop_path = Path(PROCESSED_DIR) / "drop_reasons.json"
    with open(drop_path, "w") as f:
        json.dump(drop_reasons, f, indent=2)

    summary = {
        "records_after_cleaning": len(cleaned_texts),
        "shards": num_shards,
        "shard_size": SHARD_SIZE,
        "drop_reasons": drop_reasons,
        "runtime_sec": round(time.time()-start, 2),
        "text_output": str(text_output_file),
        "token_output": str(combined_file)
    }
    
    with open(Path(PROCESSED_DIR) / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("✨ PIPELINE COMPLETE")
    print("="*60)
    print(f"Time: {summary['runtime_sec']} sec")
    print(f"Records: {summary['records_after_cleaning']:,}")
    print(f"Shards: {num_shards}")
    print(f"Text output (SUBMIT THIS): {text_output_file}")
    print(f"Token output: {combined_file}")
    print(f"Drop reasons saved to: {drop_path}")


if __name__ == "__main__":
    main()
