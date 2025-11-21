# mod/data_cleaning.py
"""
Optimized data cleaning module
- Uses tiktoken for proper tokenization
- Optimized for batch processing
- Compatible with multiprocessing
"""
import re
from collections import Counter
from langdetect import detect, DetectorFactory
import tiktoken

DetectorFactory.seed = 42 # IYKYK

# Initialize tokenizer (globally, once)
ENCODING_NAME = "cl100k_base"  # GPT-3.5/4 encoding
enc = tiktoken.get_encoding(ENCODING_NAME)

# Compile regex patterns once (much faster)
EMAIL_PATTERN = re.compile(r"\S+@\S+")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-]{7,}\d")
WHITESPACE_PATTERN = re.compile(r"\s+")

def clean_text(text):
    """Basic normalization: strip, collapse spaces, remove weird characters."""
    text = text.strip()
    text = WHITESPACE_PATTERN.sub(" ", text)  # collapse multiple spaces
    text = text.replace("\n", " ").replace("\r", "")
    return text

def is_english(text, sample_size=500):
    """
    Detect if text is English.
    Only sample first N chars for speed.
    """
    try:
        # Only use first N characters for faster detection
        sample = text[:sample_size] if len(text) > sample_size else text
        return detect(sample) == "en"
    except:
        return False

def remove_pii(text):
    """
    Simple regex-based PII removal: emails and phone numbers.
    Returns: (cleaned_text, pii_found)
    """
    original_text = text
    
    # Replace emails
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    
    # Replace phone numbers
    text = PHONE_PATTERN.sub("[PHONE]", text)
    
    # Check if any PII was found
    pii_found = (text != original_text)
    
    return text, pii_found

def tokenize_text(text):
    """Convert text into tokens for LLM training."""
    return enc.encode(text)

def process_batch(records, min_tokens=5, max_tokens=1000, do_pii=True):
    """    
    Args:
        records (list): list of dicts with "text" key
        min_tokens (int): minimum token length to keep
        max_tokens (int): maximum token length to keep
        do_pii (bool): whether to replace PII
    
    Returns:
        tuple: (tokenized_texts, drop_reasons)
    """
    drop_reasons = Counter()
    cleaned_texts = []
    seen_hashes = set()  # For deduplication within this batch
    
    for rec in records:
        text = rec.get("text", "")
        
        # Step 1: Clean text
        text = clean_text(text)
        
        # Step 2: Check if empty
        if not text:
            drop_reasons["empty"] += 1
            continue
        
        # Step 3: Language detection (expensive, do early to avoid wasted work)
        if not is_english(text):
            drop_reasons["non_english"] += 1
            continue
        
        # Step 4: PII handling
        if do_pii:
            text, pii_found = remove_pii(text)
            if pii_found:
                drop_reasons["pii_replaced"] += 1
        
        # Step 5: Deduplication (before tokenization to save time)
        text_hash = hash(text.lower().strip())
        if text_hash in seen_hashes:
            drop_reasons["duplicate"] += 1
            continue
        seen_hashes.add(text_hash)
        
        # Passed all filters, add to cleaned list
        cleaned_texts.append(text)
    
    # Step 6: Tokenize all at once (batch operation)
    tokenized_texts = []
    for text in cleaned_texts:
        try:
            tokens = tokenize_text(text)
            
            # Step 7: Length filtering
            if min_tokens <= len(tokens) <= max_tokens:
                tokenized_texts.append(tokens)
            else:
                drop_reasons["length_filtered"] += 1
        except Exception as e:
            # Handle tokenization errors gracefully
            drop_reasons["tokenization_error"] += 1
            continue
    
    return tokenized_texts, dict(drop_reasons)

def process_records(records, drop_reasons=None, do_pii=True, min_tokens=5, max_tokens=1000):
    """
    Legacy function for backwards compatibility.
    Process a list of raw records (dicts with 'text' key).
    Returns tokenized, cleaned, deduplicated list and updates drop reasons.
    
    Args:
        records (list): list of dicts with "text" key
        drop_reasons (dict): dict to track discarded records (will be merged)
        do_pii (bool): whether to replace PII
        min_tokens (int): minimum token length to keep
        max_tokens (int): maximum token length to keep
    
    Returns:
        tuple: (tokenized_texts, drop_reasons)
    """
    if drop_reasons is None:
        drop_reasons = Counter()
    else:
        drop_reasons = Counter(drop_reasons)
    
    # Use the optimized batch processing
    tokenized_texts, batch_drop_reasons = process_batch(
        records, 
        min_tokens=min_tokens, 
        max_tokens=max_tokens, 
        do_pii=do_pii
    )
    
    # Merge drop reasons
    for reason, count in batch_drop_reasons.items():
        drop_reasons[reason] += count
    
    return tokenized_texts, dict(drop_reasons)

# ----------------------
# Utility functions for analysis
# ----------------------

def get_text_stats(text):
    """
    Get statistics about a text.
    Useful for inspection and quality control.
    """
    tokens = tokenize_text(text)
    
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "token_count": len(tokens),
        "avg_token_length": len(text) / len(tokens) if tokens else 0,
    }

def estimate_quality_score(text):
    """
    Rough quality estimation heuristic.
    Returns score 0-1 (higher is better).
    
    Checks for:
    - Reasonable length
    - Proper capitalization
    - Punctuation usage
    - Non-repetitive content
    """
    score = 1.0
    
    # Check length (prefer medium-length texts)
    word_count = len(text.split())
    if word_count < 10 or word_count > 2000:
        score *= 0.7
    
    # Check capitalization (should have some capitals)
    if text.isupper() or text.islower():
        score *= 0.8
    
    # Check for punctuation
    if not any(p in text for p in '.!?'):
        score *= 0.9
    
    # Check for excessive repetition
    words = text.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Very repetitive
            score *= 0.6
    
    return score

def process_batch_text_only(records, min_tokens=5, max_tokens=1000, do_pii=True):
    """
    Process a batch of records and return CLEANED TEXT (not tokens).
    
    This version is for final output submission where text format is required.
    Tokenization is only used for length validation.
    
    Args:
        records (list): list of dicts with "text" key
        min_tokens (int): minimum token length to keep
        max_tokens (int): maximum token length to keep
        do_pii (bool): whether to replace PII
    
    Returns:
        tuple: (text_records, drop_reasons)
        where text_records is list of dicts: [{"text": "cleaned text"}, ...]
    """
    drop_reasons = Counter()
    text_records = []
    seen_hashes = set()  # For deduplication within this batch
    
    for rec in records:
        text = rec.get("text", "")
        
        # Step 1: Clean text
        text = clean_text(text)
        
        # Step 2: Check if empty
        if not text:
            drop_reasons["empty"] += 1
            continue
        
        # Step 3: Language detection
        if not is_english(text):
            drop_reasons["non_english"] += 1
            continue
        
        # Step 4: PII handling
        if do_pii:
            text, pii_found = remove_pii(text)
            if pii_found:
                drop_reasons["pii_replaced"] += 1
        
        # Step 5: Deduplication (before tokenization to save time)
        text_hash = hash(text.lower().strip())
        if text_hash in seen_hashes:
            drop_reasons["duplicate"] += 1
            continue
        seen_hashes.add(text_hash)
        
        # Step 6: Tokenize for length validation ONLY
        try:
            tokens = tokenize_text(text)
            
            # Step 7: Length filtering
            if min_tokens <= len(tokens) <= max_tokens:
                # Store TEXT, not tokens!
                text_records.append({"text": text})
            else:
                drop_reasons["length_filtered"] += 1
        except Exception as e:
            drop_reasons["tokenization_error"] += 1
            continue
    
    return text_records, dict(drop_reasons)


# Add this to your data_cleaning.py

def process_batch_dual_output(records, min_tokens=5, max_tokens=1000, do_pii=True):
    """
    Process batch and return BOTH tokenized data AND cleaned text.
    
    Returns:
        tuple: (tokenized_list, text_list, drop_reasons)
        Both lists have the SAME length and correspond 1:1
    """
    from collections import Counter
    
    drop_reasons = Counter()
    tokenized_list = []
    text_list = []
    seen_hashes = set()
    
    for rec in records:
        text = rec.get("text", "")
        
        # Step 1: Clean text
        text = clean_text(text)
        
        # Step 2: Check if empty
        if not text:
            drop_reasons["empty"] += 1
            continue
        
        # Step 3: Language detection
        if not is_english(text):
            drop_reasons["non_english"] += 1
            continue
        
        # Step 4: PII handling
        if do_pii:
            text, pii_found = remove_pii(text)
            if pii_found:
                drop_reasons["pii_replaced"] += 1
        
        # Step 5: Deduplication
        text_hash = hash(text.lower().strip())
        if text_hash in seen_hashes:
            drop_reasons["duplicate"] += 1
            continue
        seen_hashes.add(text_hash)
        
        # Step 6: Tokenize
        try:
            tokens = tokenize_text(text)
            
            # Step 7: Length filtering
            if min_tokens <= len(tokens) <= max_tokens:
                # BOTH outputs - they correspond 1:1
                tokenized_list.append(tokens)
                text_list.append(text)  # CLEANED text with PII removed
            else:
                drop_reasons["length_filtered"] += 1
        except Exception:
            drop_reasons["tokenization_error"] += 1
            continue
    
    return tokenized_list, text_list, dict(drop_reasons)
