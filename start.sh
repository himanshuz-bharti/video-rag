#!/bin/bash

# Start Ollama service in background
echo "Starting Ollama daemon..."
ollama serve &

# Wait for Ollama service to start responding
echo "Waiting for Ollama to initialize..."
while ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; do
  sleep 1
done
echo "Ollama is ready!"

# Pull the required models (Fast on Hugging Face due to cloud backend speeds)
echo "Pulling vision model: qwen2.5vl:3b..."
ollama pull qwen2.5vl:3b

echo "Pulling embedding model: nomic-embed-text..."
ollama pull nomic-embed-text

# Start the FastAPI application on the port set by Hugging Face (PORT env var)
echo "Starting FastAPI server..."
python main_web.py
