# Dockerfile
# Container for LLM data preparation pipeline

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY mod/ ./mod/
COPY scripts/ ./scripts/
COPY main.py .
COPY run_pipeline.sh .

# Make shell script executable
RUN chmod +x run_pipeline.sh

# Create data directories
RUN mkdir -p data/raw data/processed data/plots

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["bash", "run_pipeline.sh"]