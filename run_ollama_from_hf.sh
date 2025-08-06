#!/bin/bash

# Hardcoded Hugging Face model repo name
MODEL_REPO="<user_name>/my-finetuned-gemma3n"  # <-- Change this to your actual repo name
PORT=11434

echo "🔄 Pulling model from Hugging Face: $MODEL_REPO"
ollama pull $MODEL_REPO

echo "🚀 Starting Ollama server on port $PORT..."
OLLAMA_PORT=$PORT ollama serve &

OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama API to be ready on port $PORT..."
until curl -s "http://localhost:$PORT/api/tags" > /dev/null; do
  sleep 1
done

echo "✅ Ollama is running on http://localhost:$PORT"
echo "You can now use your Next.js app with this model!"

# Optionally, run the model (uncomment if you want to start a REPL)
# ollama run $MODEL_REPO

# Wait for Ollama server to exit (keep script running)
wait $OLLAMA_PID