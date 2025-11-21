# LLM Data Preparation Pipeline

End-to-end containerized pipeline for processing and cleaning text data for LLM training.

## Project Structure

```
mainpipe-assignment/
│
├── mod/                          # Core pipeline modules
│   ├── __init__.py
│   └── data_cleaning.py          # Cleaning, tokenization, PII removal
│
├── scripts/                      # Inspection/analysis scripts
│   ├── input_inspection.py       # Pre-processing analysis
│   └── output_inspection.py      # Post-processing analysis
│
├── data/
│   ├── raw/                      # Input JSONL files
│   ├── processed/                # Output shards & metadata
│   └── plots/                    # Generated visualizations
│
├── main.py                       # Pipeline orchestrator
├── run_pipeline.sh               # Automated execution script
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
└── ND_Mainpipe_Assignment.pdf    # pdf of the report
```

## Features

### Pipeline Stages
1. **Light Filtering**: Remove empty texts, basic validation
2. **Heavy Processing**: Language detection, PII removal, tokenization, deduplication
3. **Shuffle & Shard**: Randomize and split into manageable shards

### Evaluation Metrics
- **Deduplication**: Hash-based duplicate detection
- **Noise/Integrity**: Token length filtering, drop reason tracking
- **Safety**: PII detection and removal
- **Coverage**: Language distribution analysis
- **Quality**: Vocabulary richness, lexical diversity
- **Performance**: Throughput metrics, pipeline timing
- **Linguistics**: Similarity analysis for repeat detection

## Installation
### ⚠️ **Before Running – Add Raw Data**
Download your input `.jsonl` files and place them in a new folder inside data/: data/raw

The pipeline expects this directory to contain all raw input files.

### Option 1: Local Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Manual Execution
```bash
# Run full pipeline
bash run_pipeline.sh
```

### Configuration
Edit `main.py` to adjust pipeline parameters:
```python
MIN_TOKENS = 10      # Minimum token length
MAX_TOKENS = 1000    # Maximum token length
BATCH_SIZE = 1000    # Processing batch size
SHARD_SIZE = 5000    # Records per output shard
```

## Output

### Processed Data
- `data/processed/shard_XXXX.jsonl` - Tokenized sequences
- `data/processed/drop_reasons.json` - Filtering statistics
- `data/processed/pipeline_summary.json` - Performance metrics

### Visualizations
All plots saved to `data/plots/`:

**Input Analysis (3 plots):**
- Language confidence scores
- Language distribution (coverage)
- PII detection rates

**Output Analysis (8 plots):**
- Token length distribution
- Deduplication results
- Drop reasons breakdown
- Vocabulary richness
- Lexical diversity
- PII verification
- Pipeline data flow
- Similarity distribution

## Technical Details

### Data Cleaning
- Language detection: `langdetect` library
- Tokenization: `tiktoken` (cl100k_base encoding)
- PII removal: Regex-based (emails, phones, SSNs)
- Deduplication: Hash-based within batches

### Performance
- Streaming input: Low memory footprint
- Batch processing: Efficient vectorization
- Deterministic: Reproducible with seed=42

### Safety
- PII patterns replaced with placeholders
- Output verification ensures PII removal
- Safety metrics tracked pre/post processing

## Requirements

- Python 3.10+
- 8GB+ RAM recommended
- Disk space: ~3x input data size




