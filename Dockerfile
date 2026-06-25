FROM python:3.12-slim

# Install system dependencies (ffmpeg and opencv prerequisites)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy application source files
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["python", "main_web.py"]
