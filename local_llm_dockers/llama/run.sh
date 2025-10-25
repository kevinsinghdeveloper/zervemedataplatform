#!/bin/bash

echo "Building Ollama Docker image..."
docker build -t llama-local .

echo "Stopping and removing existing container if any..."
docker stop ollama 2>/dev/null || true
docker rm ollama 2>/dev/null || true

echo "Starting Ollama container..."
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  llama-local

echo "Waiting for Ollama to start..."
sleep 5

echo "Pulling llama3.2 model..."
docker exec ollama ollama pull llama3.2

echo "Ollama is ready!"
echo ""
echo "Check if it's running:"
docker ps | grep ollama

echo ""
echo "To view logs, run: docker logs -f ollama"
echo "To test the API, run: curl http://localhost:11434/api/tags"
echo "To stop: docker stop ollama"
echo "To start again: docker start ollama"