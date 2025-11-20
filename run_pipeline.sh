#!/bin/bash
# run_pipeline.sh
# Automated execution of the complete data preparation pipeline

echo "=========================================="
echo "LLM Data Preparation Pipeline"
echo "=========================================="
echo ""

# Step 1: Input Inspection
echo "Step 1/3: Analyzing raw input data..."
python scripts/input_inspection.py
if [ $? -ne 0 ]; then
    echo "❌ Input inspection failed!"
    exit 1
fi
echo "✅ Input analysis complete"
echo ""

# Step 2: Main Pipeline
echo "Step 2/3: Running data processing pipeline..."
python main.py
if [ $? -ne 0 ]; then
    echo "❌ Pipeline execution failed!"
    exit 1
fi
echo "✅ Pipeline complete"
echo ""

# Step 3: Output Inspection
echo "Step 3/3: Analyzing processed output data..."
python scripts/output_inspection.py
if [ $? -ne 0 ]; then
    echo "❌ Output inspection failed!"
    exit 1
fi
echo "✅ Output analysis complete"
echo ""

echo "=========================================="
echo "✨ PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "📊 Plots saved to: data/plots/"
echo "📁 Processed data: data/processed/"
echo ""