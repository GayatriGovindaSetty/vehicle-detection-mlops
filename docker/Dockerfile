# Production Dockerfile for Vehicle Detection
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/

# Copy model (if exists, otherwise will download default)
COPY models/best.pt ./models/best.pt* 2>/dev/null || echo "No model found, will use default"

# Create necessary directories
RUN mkdir -p models data logs

# Expose port for Gradio
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app/app.py"]
