# scripts/input_inspection.py
"""
Input Data Analysis - Pre-Processing Inspection
Evaluates raw data quality before pipeline processing

Metrics aligned with evaluation criteria:
- Coverage: Language distribution
- Safety: PII detection rates
- Quality: Language confidence scores
"""
import json
import os
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from langdetect import detect_langs, DetectorFactory
from tqdm import tqdm
import seaborn as sns

# Make langdetect deterministic
DetectorFactory.seed = 0

RAW_DIR = "data/raw/"
PLOTS_DIR = "data/plots/"
SAMPLE_LIMIT = 50000  # max samples to scan

# Create plots directory
os.makedirs(PLOTS_DIR, exist_ok=True)

# PII patterns for safety evaluation
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')


def analyze_raw_input():
    """
    Analyze raw input data for evaluation metrics
    """
    print("\n" + "="*60)
    print("INPUT DATA ANALYSIS (PRE-PROCESSING)")
    print("="*60)
    
    # Metrics for evaluation
    lang_scores = []
    detected_langs = []
    pii_counts = []
    
    empty = 0
    errors = 0
    total_lines = 0
    
    # Load all raw JSONL files
    all_files = [
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.endswith(".jsonl")
    ]
    
    if not all_files:
        print("❌ No JSONL files found in raw/ directory.")
        return
    
    print(f"\nFound {len(all_files)} JSONL file(s):")
    for f in all_files:
        print(f"  • {os.path.basename(f)}")
    
    # Iterate through files
    for file_path in all_files:
        print(f"\n📄 Processing {os.path.basename(file_path)}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading"):
                total_lines += 1
                if total_lines > SAMPLE_LIMIT:
                    break
                
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue
                
                text = rec.get("text", "").strip()
                
                if not text:
                    empty += 1
                    continue
                
                # Language detection with confidence (COVERAGE)
                try:
                    lang_probs = detect_langs(text)
                    if lang_probs:
                        top_lang = lang_probs[0]
                        detected_langs.append(top_lang.lang)
                        lang_scores.append(top_lang.prob)
                    else:
                        detected_langs.append('unknown')
                        lang_scores.append(0.0)
                except:
                    detected_langs.append('unknown')
                    lang_scores.append(0.0)
                
                # PII detection (SAFETY)
                email_count = len(EMAIL_PATTERN.findall(text))
                phone_count = len(PHONE_PATTERN.findall(text))
                ssn_count = len(SSN_PATTERN.findall(text))
                total_pii = email_count + phone_count + ssn_count
                pii_counts.append(total_pii)
        
        if total_lines >= SAMPLE_LIMIT:
            break
    
    # Calculate metrics
    pii_with_hits = sum(1 for x in pii_counts if x > 0)
    pii_hit_rate = pii_with_hits / len(pii_counts) * 100 if pii_counts else 0
    
    # Print summary
    print("\n" + "="*60)
    print("INPUT DATA METRICS")
    print("="*60)
    print(f"Total lines read:        {total_lines:,}")
    print(f"Empty texts:             {empty:,}")
    print(f"JSON errors:             {errors:,}")
    print(f"Valid texts analyzed:    {len(detected_langs):,}")
    
    print(f"\n🌍 COVERAGE - Language Distribution:")
    print(f"  Mean confidence: {np.mean(lang_scores):.3f}")
    print(f"  Languages detected: {len(set(detected_langs))}")
    lang_counter = Counter(detected_langs)
    print(f"  Top 5 languages:")
    for lang, count in lang_counter.most_common(5):
        print(f"    {lang}: {count:,} ({count/len(detected_langs)*100:.1f}%)")
    
    print(f"\n🔒 SAFETY - PII Detection:")
    print(f"  Texts with PII: {pii_with_hits:,} ({pii_hit_rate:.1f}%)")
    print(f"  Total PII instances: {sum(pii_counts):,}")
    print(f"  Average PII per text: {np.mean(pii_counts):.2f}")
    print(f"  ⚠️  These should be removed by pipeline!")
    
    # Generate plots
    generate_input_plots(lang_scores, detected_langs, pii_counts, pii_hit_rate)
    
    print("\n" + "="*60)
    print("✅ INPUT ANALYSIS COMPLETE")
    print("="*60)

def generate_input_plots(lang_scores, detected_langs, pii_counts, pii_hit_rate):
    """
    Generate visualization plots for input evaluation metrics
    """
    print("\n📊 Generating plots...")

    # Set global font size
    plt.rcParams.update({'font.size': 18})

    # 1. Language Confidence Scores (Quality)
    plt.figure(figsize=(10, 10))
    sns.violinplot(y=lang_scores, color='lightgray', inner='quartile')
    mean_score = np.mean(lang_scores)
    plt.axhline(mean_score, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')

    plt.ylabel('Language Detection Confidence Score')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'input_01_language_confidence_violin.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: input_01_language_confidence_violin.png")

    # 2. Language Distribution (Coverage)
    lang_counts = Counter(detected_langs)
    total = sum(lang_counts.values())

    # Separate major and minor languages
    major_langs = {lang: count for lang, count in lang_counts.items() if count / total >= 0.001}  # ≥0.1%
    minor_count = sum(count for lang, count in lang_counts.items() if count / total < 0.001)
    if minor_count > 0:
        major_langs["Others"] = minor_count

    # Sort by count descending
    major_langs = dict(sorted(major_langs.items(), key=lambda x: x[1], reverse=True))

    langs, counts = zip(*major_langs.items()) if major_langs else ([], [])
    percentages = [(c / total * 100) for c in counts]

    plt.figure(figsize=(10, 10))
    bars = plt.bar(range(len(langs)), counts, color='lightblue', edgecolor='black')
    plt.xticks(range(len(langs)), langs, rotation=45, ha='right')
    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.ylim(0, max(counts) * 1.1) 
    plt.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on top
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({pct:.2f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'input_02_language_distribution.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: input_02_language_distribution.png")

    # 3. PII Detection (Safety)
    plt.figure(figsize=(10, 10))
    max_pii = max(pii_counts) if pii_counts else 0
    plt.hist(pii_counts, bins=range(min(max_pii+2, 20)), color='mediumpurple', edgecolor='black', alpha=0.7, density=True)

    plt.xlabel('Number of PII Instances per Text')
    plt.ylabel('Density')

    plt.plot([], [], ' ', label=f'Hit Rate: {pii_hit_rate:.1f}%')
    plt.legend(markerscale=0)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'input_03_pii_detection.png'), bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: input_03_pii_detection.png")

    print(f"\n📈 Plots saved to: {PLOTS_DIR}")
    print("  1. Language Confidence (Quality)")
    print("  2. Language Distribution (Coverage)")
    print("  3. PII Detection (Safety)")



if __name__ == "__main__":
    analyze_raw_input()