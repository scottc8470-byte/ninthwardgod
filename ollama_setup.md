# 🤖 Ollama Setup Guide for Ultimate AI System

## Quick Install

### macOS/Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download from: https://ollama.com/download/windows

## Install Models

```bash
# Start Ollama service
ollama serve

# In another terminal, pull models:
# Small & Fast (recommended for quick responses)
ollama pull llama2:7b
ollama pull mistral:7b

# Larger & More Capable
ollama pull llama2:13b
ollama pull mixtral:8x7b

# Code-focused
ollama pull codellama:7b

# Very small (for testing)
ollama pull tinyllama
```

## Verify Installation

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Test a model
ollama run llama2:7b "Hello, I am an AI assistant"
```

## Integration with Mega AI System

The system automatically detects if Ollama is running. When available:
- 🟢 "Local LLM: Online" appears in sidebar
- Enhanced responses with actual LLM generation
- Multiple personality modes leverage LLM
- Fallback to knowledge base if offline

## Recommended Models by Use Case

### For General Chat
- `llama2:7b` - Balanced performance
- `mistral:7b` - Fast and capable

### For Code/Technical
- `codellama:7b` - Optimized for code
- `deepseek-coder:6.7b` - Great for programming

### For Creative/Rodeo Mode
- `mixtral:8x7b` - Most creative
- `llama2:13b` - Good balance

## Performance Tips

1. **GPU Acceleration** (if available):
   - NVIDIA: Ollama uses CUDA automatically
   - Apple Silicon: Uses Metal automatically

2. **Memory Management**:
   ```bash
   # Set memory limit (e.g., 8GB)
   OLLAMA_MAX_LOADED_MODELS=1 ollama serve
   ```

3. **Response Speed**:
   - Smaller models (7B) respond faster
   - Quantized models use less memory

## Troubleshooting

### Ollama not detected?
1. Ensure service is running: `ollama serve`
2. Check port 11434 is not blocked
3. Restart the Streamlit app

### Slow responses?
1. Try smaller model: `ollama pull tinyllama`
2. Close other applications
3. Check GPU is being used: `ollama list`

### Model not loading?
1. Check disk space
2. Re-pull model: `ollama pull llama2:7b`
3. Check logs: `journalctl -u ollama`

## Advanced Configuration

### Custom Models
```bash
# Create custom personality
cat > rodeo-ai.modelfile << EOF
FROM llama2:7b
PARAMETER temperature 0.9
PARAMETER top_p 0.95
SYSTEM "You are Rodeo AI, a confident AGI assistant with quantum capabilities."
EOF

ollama create rodeo-ai -f rodeo-ai.modelfile
```

### API Usage
The Mega AI System uses Ollama's API:
```python
POST http://localhost:11434/api/generate
{
  "model": "llama2:7b",
  "prompt": "Your question here",
  "stream": false
}
```

## Security Note

Ollama runs locally - your data never leaves your machine!

---

🎉 Once installed, the Ultimate AI System will automatically use your local LLM for enhanced responses!