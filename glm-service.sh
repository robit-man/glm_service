#!/usr/bin/env bash
# ============================================================================
# GLM-5 Service Daemon
# ============================================================================
#
# Auto-downloads GLM-5 (744B MoE, 40B active) from Unsloth and serves
# it via llama-server with an OpenAI-compatible API.
#
# Architecture: glm_moe_dsa (256 experts, 8 active, DeepSeek Sparse Attention)
# Default quant: Unsloth Dynamic 2-bit (IQ2_XXS) — 241GB, 6 shards
# Alt quant:     Unsloth Dynamic 1-bit (TQ1_0) — 164GB (NOT RECOMMENDED: gibberish)
# Requirements: ~250GB RAM/VRAM combined, CUDA-capable GPU recommended
#
# NOTE: TQ1_0 (1-bit) produces incoherent output (Chinese, repetition, garbage).
# IQ2_XXS (2-bit) is the minimum viable quantization for this model.
#
# Usage:
#   ./glm-service.sh                    # Download + serve (default port 8081)
#   ./glm-service.sh --port 9000        # Custom port
#   ./glm-service.sh --download-only    # Just download, don't serve
#   ./glm-service.sh --storage /path    # Custom storage location
#   ./glm-service.sh --install-daemon   # Install as systemd service
#
# API Endpoints (OpenAI-compatible):
#   POST /v1/chat/completions    — Chat completion (streaming supported)
#   POST /v1/completions         — Text completion
#   GET  /v1/models              — List models
#   GET  /health                 — Health check
#
# Connect from Ollama (as external model):
#   OLLAMA_HOST=http://localhost:8081 ollama run glm-5
#
# Connect from oa (open-agents):
#   oa --backend openai --api-url http://localhost:8081/v1 --model glm-5
# ============================================================================

set -euo pipefail

# ── Defaults ──
STORAGE="${GLM_STORAGE:-/media/roko/sdb1/GLM-5-GGUF}"
PORT="${GLM_PORT:-8081}"
HOST="${GLM_HOST:-0.0.0.0}"
CTX_SIZE="${GLM_CTX_SIZE:-8192}"
PARALLEL="${GLM_PARALLEL:-2}"
GPU_LAYERS="${GLM_GPU_LAYERS:-99}"
TENSOR_SPLIT="${GLM_TENSOR_SPLIT:-80,80,80,0}"  # 3x A100 + skip GT1030
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/media/roko/sdb1/llama.cpp}"
QUANT="${GLM_QUANT:-iq2}"  # iq2 (default, recommended) or tq1 (not recommended)
HF_REPO="unsloth/GLM-5-GGUF"
DOWNLOAD_ONLY=false
INSTALL_DAEMON=false

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --storage) STORAGE="$2"; shift 2 ;;
        --ctx-size) CTX_SIZE="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        --tensor-split) TENSOR_SPLIT="$2"; shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        --download-only) DOWNLOAD_ONLY=true; shift ;;
        --install-daemon) INSTALL_DAEMON=true; shift ;;
        --help|-h)
            head -40 "$0" | tail -35
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Resolve model file from quant selection ──
case "${QUANT}" in
    iq2|IQ2|2bit)
        MODEL_SUBDIR="UD-IQ2_XXS"
        MODEL_FILE="GLM-5-UD-IQ2_XXS-00001-of-00006.gguf"
        HF_PATTERN="UD-IQ2_XXS/*"
        MODEL_SIZE="241GB (6 shards)"
        ;;
    tq1|TQ1|1bit)
        MODEL_SUBDIR=""
        MODEL_FILE="GLM-5-UD-TQ1_0.gguf"
        HF_PATTERN="GLM-5-UD-TQ1_0*"
        MODEL_SIZE="164GB"
        warn "TQ1_0 (1-bit) produces incoherent output. Use --quant iq2 instead."
        ;;
    *)
        echo "Unknown quant: ${QUANT}. Use: iq2 (recommended) or tq1"; exit 1
        ;;
esac

if [[ -n "${MODEL_SUBDIR}" ]]; then
    MODEL_PATH="${STORAGE}/${MODEL_SUBDIR}/${MODEL_FILE}"
else
    MODEL_PATH="${STORAGE}/${MODEL_FILE}"
fi

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[glm-service]${NC} $*"; }
ok()  { echo -e "${GREEN}[glm-service]${NC} $*"; }
warn(){ echo -e "${YELLOW}[glm-service]${NC} $*"; }
err() { echo -e "${RED}[glm-service]${NC} $*" >&2; }

# ── Step 1: Ensure storage directory ──
log "Storage: ${STORAGE}"
mkdir -p "${STORAGE}" 2>/dev/null || {
    err "Cannot create ${STORAGE} — check permissions"
    exit 1
}

# ── Step 2: Download model if not present ──
if [[ ! -f "${MODEL_PATH}" ]]; then
    log "Model not found at ${MODEL_PATH}"
    log "Downloading GLM-5 ${QUANT} (${MODEL_SIZE}) from Hugging Face..."
    log "Repo: ${HF_REPO} | Pattern: ${HF_PATTERN}"

    # Check for huggingface_hub
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        warn "Installing huggingface_hub..."
        pip3 install --user --break-system-packages huggingface-hub 2>/dev/null ||
        pip3 install --user huggingface-hub 2>/dev/null ||
        pip3 install huggingface-hub 2>/dev/null || {
            err "Failed to install huggingface-hub. Install manually: pip3 install huggingface-hub"
            exit 1
        }
    fi

    python3 -c "
from huggingface_hub import snapshot_download
import time

print('Downloading GLM-5 model shards...', flush=True)
print(f'Started: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}', flush=True)
start = time.time()

path = snapshot_download(
    repo_id='${HF_REPO}',
    allow_patterns='${HF_PATTERN}',
    local_dir='${STORAGE}',
)

elapsed = time.time() - start
print(f'Complete: {path}', flush=True)
print(f'Duration: {elapsed/60:.1f} minutes', flush=True)
"

    if [[ ! -f "${MODEL_PATH}" ]]; then
        err "Download failed — model file not found"
        exit 1
    fi
    ok "Download complete: ${MODEL_PATH}"
else
    ok "Model found: ${MODEL_PATH} ($(du -h "${MODEL_PATH}" | cut -f1))"
fi

if [[ "${DOWNLOAD_ONLY}" == "true" ]]; then
    ok "Download-only mode. Exiting."
    exit 0
fi

# ── Step 3: Ensure llama-server binary ──
LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"
if [[ ! -x "${LLAMA_SERVER}" ]]; then
    warn "llama-server not found at ${LLAMA_SERVER}"
    log "Building llama.cpp with CUDA support..."

    if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
        log "Cloning llama.cpp..."
        git clone https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_DIR}"
    fi

    cd "${LLAMA_CPP_DIR}"
    git pull origin master 2>/dev/null || true
    mkdir -p build && cd build

    # Detect CUDA architectures
    CUDA_ARCHS="80"  # Default: A100
    if command -v nvidia-smi &>/dev/null; then
        CUDA_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' .' | sort -u | paste -sd ';')
        log "Detected CUDA architectures: ${CUDA_ARCHS}"
    fi

    cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" -DCMAKE_BUILD_TYPE=Release
    make -j"$(nproc)" llama-server

    if [[ ! -x "${LLAMA_SERVER}" ]]; then
        err "Build failed — llama-server not found"
        exit 1
    fi
    ok "llama-server built: ${LLAMA_SERVER}"
fi

# ── Step 4: Install as systemd service (if requested) ──
if [[ "${INSTALL_DAEMON}" == "true" ]]; then
    SERVICE_FILE="/etc/systemd/system/glm-service.service"
    SCRIPT_PATH="$(readlink -f "$0")"

    log "Installing systemd service..."
    sudo tee "${SERVICE_FILE}" > /dev/null <<UNIT
[Unit]
Description=GLM-5 Inference Service (744B MoE, 1-bit)
After=network.target
Wants=network.target

[Service]
Type=exec
User=$(whoami)
Environment="GLM_STORAGE=${STORAGE}"
Environment="GLM_PORT=${PORT}"
Environment="GLM_HOST=${HOST}"
Environment="GLM_CTX_SIZE=${CTX_SIZE}"
Environment="GLM_PARALLEL=${PARALLEL}"
Environment="GLM_GPU_LAYERS=${GPU_LAYERS}"
Environment="GLM_TENSOR_SPLIT=${TENSOR_SPLIT}"
Environment="LLAMA_CPP_DIR=${LLAMA_CPP_DIR}"
ExecStart=${SCRIPT_PATH}
Restart=on-failure
RestartSec=10
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
UNIT

    sudo systemctl daemon-reload
    sudo systemctl enable glm-service
    ok "Systemd service installed: glm-service"
    ok "Start with: sudo systemctl start glm-service"
    ok "Logs with:  journalctl -u glm-service -f"
    exit 0
fi

# ── Step 5: Launch llama-server ──
log "Starting GLM-5 inference server..."
log "  Model:    ${MODEL_PATH}"
log "  Endpoint: http://${HOST}:${PORT}/v1"
log "  Context:  ${CTX_SIZE} tokens"
log "  Parallel: ${PARALLEL} concurrent requests"
log "  GPU:      ${GPU_LAYERS} layers offloaded"
log "  Split:    ${TENSOR_SPLIT}"
log ""
ok "API ready at http://localhost:${PORT}/v1/chat/completions"
log "Press Ctrl+C to stop"
echo ""

exec "${LLAMA_SERVER}" \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --n-gpu-layers "${GPU_LAYERS}" \
    --ctx-size "${CTX_SIZE}" \
    --parallel "${PARALLEL}" \
    --flash-attn on \
    --split-mode layer \
    --tensor-split "${TENSOR_SPLIT}" \
    --chat-template glm4
