FROM python:3.12-slim

# Force stdin, stdout, and stderr to be unbuffered to get logs in real time
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg, opencv prerequisites, and curl/certs for Ollama)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama inside the container
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency definition and install only dependencies (caching layer)
COPY pyproject.toml .
RUN uv pip install --verbose --system --no-cache -r pyproject.toml

# Copy application source files
COPY . .

# Ensure start.sh is executable and has Unix line endings
RUN chmod +x start.sh && sed -i 's/\r$//' start.sh

# Expose port (FastAPI will read this dynamically from PORT env var, e.g. 7860 on HF)
EXPOSE 7860

# Run entrypoint script
CMD ["./start.sh"]
