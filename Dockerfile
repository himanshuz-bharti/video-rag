FROM python:3.12-slim

# Force stdin, stdout, and stderr to be unbuffered to get logs in real time
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg and opencv prerequisites)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency definition and install only dependencies (caching layer)
COPY pyproject.toml .
RUN uv pip install --verbose --system --no-cache -r pyproject.toml

# Copy application source files
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["python", "main_web.py"]
