# scripts/output_inspection.py
"""
Output Data Analysis - Post-Processing Inspection
Evaluates processed data quality after pipeline

Metrics aligned with evaluation criteria:
- Deduplication: Duplicate detection on tokenized sequences
- Noise/Integrity: Token length distribution, drop reasons
- Safety: PII detection (should be 0 or near-0)
- Quality: Vocabulary richness, lexical diversity
- Performance: Pipeline throughput, data flow
- Linguistics: Similarity distribution (repeats detection)
"""
import json
import os
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("⚠️  tiktoken not found - PII detection will be skipped")

PROCESSED_DIR = "../data/processed/"
PLOTS_DIR = "../data/plots/"

# Create plots directory
os.makedirs(PLOTS_DIR, exist_ok=True)

# PII patterns for safety verification
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')


def analyze_output_data():
    """
    Analyze processed output data for evaluation metrics
    """
    print("\n" + "="*60)
    print("OUTPUT DATA ANALYSIS (POST-PROCESSING)")
    print("="*60)
    
    # Metrics for evaluation
    token_lengths = []
    duplicate_hashes = set()
    is_duplicate_list = []
    sequences_for_similarity = []
    decoded_texts = []  # For vocabulary and PII analysis
    
    total_shards = 0
    total_records = 0
    
    # Load all shard files
    processed_path = Path(PROCESSED_DIR)
    shard_files = sorted(list(processed_path.glob("shard_*.jsonl")))
    
    if not shard_files:
        print("❌ No shard files found in processed/ directory.")
        return
    
    print(f"\nFound {len(shard_files)} shard file(s)")
    
    # Iterate through shards
    for shard_file in tqdm(shard_files, desc="Reading shards"):
        total_shards += 1
        
        with open(shard_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                total_records += 1
                
                # Data is tokenized (list of integers)
                if isinstance(data, list):
                    tokens = data
                else:
                    continue
                
                # Token length (NOISE/INTEGRITY)
                token_lengths.append(len(tokens))
                
                # Duplicate detection (DEDUPLICATION)
                token_hash = hash(tuple(tokens))
                is_dup = token_hash in duplicate_hashes
                is_duplicate_list.append(is_dup)
                duplicate_hashes.add(token_hash)
                
                # Store for similarity analysis (LINGUISTICS)
                if len(sequences_for_similarity) < 500:
                    sequences_for_similarity.append(tokens)
                
                # Decode tokens for vocabulary and PII analysis (QUALITY + SAFETY)
                if HAS_TIKTOKEN and len(decoded_texts) < 1000:
                    try:
                        decoded = enc.decode(tokens)
                        decoded_texts.append(decoded)
                    except:
                        pass
    
    # Load drop reasons (NOISE/INTEGRITY)
    drop_reasons_path = processed_path / "drop_reasons.json"
    drop_reasons = {}
    if drop_reasons_path.exists():
        with open(drop_reasons_path, 'r') as f:
            drop_reasons = json.load(f)
    
    # Load pipeline summary (PERFORMANCE)
    summary_path = processed_path / "pipeline_summary.json"
    pipeline_summary = {}
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            pipeline_summary = json.load(f)
    
    # Calculate vocabulary metrics (QUALITY)
    vocab_richness = []
    lexical_diversity = []
    pii_counts_output = []
    
    if decoded_texts:
        print("\n  Computing vocabulary and safety metrics...")
        for text in decoded_texts:
            words = text.lower().split()
            if len(words) > 0:
                unique_words = len(set(words))
                vocab_richness.append(unique_words)
                lexical_diversity.append(unique_words / len(words))
                
                # Check for PII (SAFETY - should be near 0)
                email_count = len(EMAIL_PATTERN.findall(text))
                phone_count = len(PHONE_PATTERN.findall(text))
                ssn_count = len(SSN_PATTERN.findall(text))
                pii_counts_output.append(email_count + phone_count + ssn_count)
    
    # Print summary
    print("\n" + "="*60)
    print("OUTPUT DATA METRICS")
    print("="*60)
    print(f"Total shards:            {total_shards:,}")
    print(f"Total records:           {total_records:,}")
    
    dup_counts = Counter(is_duplicate_list)
    
    print(f"\n📊 DEDUPLICATION:")
    print(f"  Unique: {dup_counts.get(False, 0):,} ({dup_counts.get(False, 0)/len(is_duplicate_list)*100:.1f}%)")
    print(f"  Duplicate: {dup_counts.get(True, 0):,} ({dup_counts.get(True, 0)/len(is_duplicate_list)*100:.1f}%)")
    
    print(f"\n📏 NOISE/INTEGRITY - Token Lengths:")
    print(f"  Mean: {np.mean(token_lengths):.1f}")
    print(f"  Median: {np.median(token_lengths):.1f}")
    print(f"  Min: {np.min(token_lengths):.0f}")
    print(f"  Max: {np.max(token_lengths):.0f}")
    print(f"  Std Dev: {np.std(token_lengths):.1f}")
    
    if drop_reasons:
        print(f"\n❌ NOISE/INTEGRITY - Drop Reasons:")
        total_dropped = sum(drop_reasons.values())
        print(f"  Total dropped: {total_dropped:,}")
        for reason, count in sorted(drop_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_dropped * 100) if total_dropped > 0 else 0
            print(f"    {reason}: {count:,} ({percentage:.1f}%)")
    
    if vocab_richness:
        print(f"\n📚 QUALITY - Vocabulary Metrics:")
        print(f"  Avg unique words per text: {np.mean(vocab_richness):.1f}")
        print(f"  Avg lexical diversity: {np.mean(lexical_diversity):.3f}")
    
    if pii_counts_output:
        pii_hits = sum(1 for x in pii_counts_output if x > 0)
        pii_rate = pii_hits / len(pii_counts_output) * 100
        print(f"\n🔒 SAFETY - PII Verification:")
        print(f"  Texts with PII: {pii_hits:,} ({pii_rate:.1f}%)")
        print(f"  Total PII instances: {sum(pii_counts_output):,}")
        if pii_rate > 1.0:
            print(f"  ⚠️  WARNING: PII rate > 1% - pipeline may need improvement")
        else:
            print(f"  ✅ PII rate acceptable")
    
    if pipeline_summary:
        print(f"\n⚡ PERFORMANCE - Pipeline Throughput:")
        runtime = pipeline_summary.get('runtime_sec', 0)
        if runtime > 0:
            throughput = total_records / runtime
            print(f"  Runtime: {runtime:.1f} seconds")
            print(f"  Throughput: {throughput:.1f} records/sec")
        print(f"  Records processed: {total_records:,}")
    
    # Generate plots
    generate_output_plots(
        token_lengths, dup_counts, drop_reasons, total_records,
        pipeline_summary, sequences_for_similarity, vocab_richness,
        lexical_diversity, pii_counts_output
    )
    
    print("\n" + "="*60)
    print("✅ OUTPUT ANALYSIS COMPLETE")
    print("="*60)


def generate_output_plots(token_lengths, dup_counts, drop_reasons, total_records,
                         pipeline_summary, sequences_for_similarity, vocab_richness,
                         lexical_diversity, pii_counts_output):
    """
    Generate visualization plots for output evaluation metrics
    """
    print("\n📊 Generating plots...")

    # Set global font size
    plt.rcParams.update({'font.size': 18})

    # 1. Token Length Distribution
    plt.figure(figsize=(10, 10))
    plt.hist(token_lengths, bins=50, color='lightgray', edgecolor='black', alpha=0.7, density=True)
    plt.axvline(np.mean(token_lengths), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(token_lengths):.0f}')
    plt.axvline(np.median(token_lengths), color='magenta', linestyle='--', linewidth=2, label=f'Median: {np.median(token_lengths):.0f}')
    plt.xlabel('Token Length')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'output_01_token_length.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output_01_token_length.png")

    # 2. Duplicate Detection
    plt.figure(figsize=(10, 10))
    labels = ['Unique', 'Duplicate']
    values = [dup_counts.get(False, 0), dup_counts.get(True, 0)]
    colors = ['lightgreen', 'lightcoral']
    plt.bar(labels, values, color=colors, edgecolor='black')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, axis='y')
    total = sum(values)
    for i, v in enumerate(values):
        percentage = (v / total * 100) if total > 0 else 0
        plt.text(i, v, f'{v:,}-({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'output_02_deduplication.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output_02_deduplication.png")

    # 3. Drop Reasons
    if drop_reasons:
        plt.figure(figsize=(10, 10))
        reasons = list(drop_reasons.keys())
        counts = list(drop_reasons.values())
        tot_counts = sum(counts)
        colors_list = plt.cm.Set3(range(len(reasons)))
        bars = plt.bar(reasons, counts, color=colors_list, edgecolor='black')
        plt.ylabel('Count')
        plt.xticks(rotation=20, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            percentage = (height / tot_counts * 100) if tot_counts > 0 else 0
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}-({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'output_03_drop_reasons.png'), bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: output_03_drop_reasons.png")

    # 4. Vocabulary Richness
    if vocab_richness:
        plt.figure(figsize=(10, 10))
        plt.hist(vocab_richness, bins=30, color='lightgray', edgecolor='black', alpha=0.7, density=True)
        plt.axvline(np.mean(vocab_richness), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(vocab_richness):.1f}')
        plt.xlabel('Unique Words per Text')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'output_04_vocabulary_richness.png'), bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: output_04_vocabulary_richness.png")

    # 5. Lexical Diversity
    if lexical_diversity:
        plt.figure(figsize=(10, 10))
        plt.hist(lexical_diversity, bins=30, color='lightgray', edgecolor='black', alpha=0.7, density=True)
        plt.axvline(np.mean(lexical_diversity), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lexical_diversity):.3f}')
        plt.xlabel('Lexical Diversity (Unique Words / Total Words)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'output_05_lexical_diversity.png'), bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: output_05_lexical_diversity.png")

    # 6. PII Detection
    if pii_counts_output:
        plt.figure(figsize=(10, 10))
        max_pii = max(pii_counts_output) if pii_counts_output else 0
        pii_rate = sum(1 for x in pii_counts_output if x > 0) / len(pii_counts_output) * 100
        plt.hist(pii_counts_output, bins=range(min(max_pii + 2, 10)), color='lightgray', edgecolor='black', alpha=0.7, density=True)
        plt.xlabel('Number of PII Instances per Text')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.text(0.95, 0.95, f'Hit Rate: {pii_rate:.2f}%', ha='right', va='top', transform=plt.gca().transAxes, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'output_06_pii_verification.png'), bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: output_06_pii_verification.png")

    # 7. Pipeline Flow
    if drop_reasons and total_records > 0:
        total_dropped = sum(drop_reasons.values())
        total_input = total_records + total_dropped
        plt.figure(figsize=(10, 10))
        categories = ['Input\nRecords', 'Dropped', 'Output\nRecords']
        values = [total_input, total_dropped, total_records]
        colors_bar = ['lightblue', 'lightcoral', 'lightgreen']
        bars = plt.bar(categories, values, color=colors_bar, edgecolor='black', width=0.6)
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_input * 100) if total_input > 0 else 0
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}-({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'output_07_pipeline_flow.png'), bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: output_07_pipeline_flow.png")

    # 8. Similarity Distribution
    if len(sequences_for_similarity) >= 10:
        print("\n  Computing similarity distribution...")
        try:
            all_tokens = set()
            for seq in sequences_for_similarity:
                all_tokens.update(seq)
            vocab = sorted(list(all_tokens))
            vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
            count_matrix = np.zeros((len(sequences_for_similarity), len(vocab)))
            for i, seq in enumerate(sequences_for_similarity):
                for token in seq:
                    if token in vocab_to_idx:
                        count_matrix[i, vocab_to_idx[token]] += 1
            similarity_matrix = cosine_similarity(count_matrix)
            similarity_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            plt.figure(figsize=(10, 10))
            plt.hist(similarity_values, bins=50, color='lightgray', edgecolor='black', alpha=0.7, density=True)
            plt.axvline(np.mean(similarity_values), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarity_values):.3f}')
            plt.axvline(np.median(similarity_values), color='magenta', linestyle='--', linewidth=2, label=f'Median: {np.median(similarity_values):.3f}')
            plt.xlabel('Cosine Similarity Score')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'output_08_similarity_distribution.png'), bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: output_08_similarity_distribution.png")
        except Exception as e:
            print(f"  ⚠ Could not compute similarity: {e}")

    print(f"\n📈 Plots saved to: {PLOTS_DIR}")
    print("  1. Token Length (Noise/Integrity)")
    print("  2. Deduplication")
    print("  3. Drop Reasons (Noise/Integrity)")
    print("  4. Vocabulary Richness (Quality)")
    print("  5. Lexical Diversity (Quality)")
    print("  6. PII Verification (Safety)")
    print("  7. Pipeline Flow (Performance)")
    print("  8. Similarity Distribution (Linguistics)")


if __name__ == "__main__":
    analyze_output_data()