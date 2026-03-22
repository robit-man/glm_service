# GLM-5 Service

Auto-downloading daemon for serving [GLM-5](https://huggingface.co/zai-org/GLM-5) (744B MoE, 40B active) via an OpenAI-compatible API.

Uses the [Unsloth Dynamic 1-bit quantization](https://huggingface.co/unsloth/GLM-5-GGUF) (164GB on disk) with llama.cpp for inference.

## Quick Start

```bash
chmod +x glm-service.sh
./glm-service.sh
```

This will:
1. Download the 1-bit GGUF (164GB) from Hugging Face if not present
2. Build llama.cpp with CUDA if needed
3. Start serving on `http://localhost:8081/v1`

## API Usage

```bash
# Chat completion
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Health check
curl http://localhost:8081/health
```

## Connect from oa (open-agents)

```bash
oa --backend openai --api-url http://localhost:8081/v1 --model glm-5
```

## Install as System Daemon

```bash
./glm-service.sh --install-daemon
sudo systemctl start glm-service
journalctl -u glm-service -f
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `GLM_STORAGE` | `/media/roko/sdb1/GLM-5-GGUF` | Model storage directory |
| `GLM_PORT` | `8081` | Server port |
| `GLM_CTX_SIZE` | `8192` | Context window |
| `GLM_GPU_LAYERS` | `99` | GPU layers to offload |
| `GLM_TENSOR_SPLIT` | `80,80,80,0` | VRAM split across GPUs |
| `LLAMA_CPP_DIR` | `/media/roko/sdb1/llama.cpp` | llama.cpp directory |

## Requirements

- ~180GB RAM/VRAM combined
- CUDA-capable GPU (recommended: A100 80GB or better)
- Python 3 with `huggingface_hub` (auto-installed)
- CMake, GCC, CUDA toolkit (for building llama.cpp)

## Architecture

GLM-5 uses `glm_moe_dsa` (GLM with DeepSeek Sparse Attention):
- 744B total parameters, 40B active per token
- 256 routed experts, 8 active per token
- 200K context window (configurable via `GLM_CTX_SIZE`)
- MLA (Multi-head Latent Attention) with KV compression

## License

AGPL-3.0 (matching GLM-5's license)
