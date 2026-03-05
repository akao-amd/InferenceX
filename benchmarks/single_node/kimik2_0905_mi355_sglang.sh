#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

export HF_HOME=/huggingface
export HF_HUB_CACHE=/huggingface/hub
# export HF_TOKEN="..."
# export MODEL="/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"          # or the FP8 variant HF repo if one exists
export MODEL="/huggingface/hub/models--moonshotai--Kimi-K2-Instruct-0905/snapshots/ac6c49f04883bd0a0598b790693a72061c676629"
export PRECISION="bf16"
export IMAGE="rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260303"
export TP=8
export CONC=64
export ISL=1024
export OSL=1024
export RANDOM_RANGE_RATIO=1.0
export RESULT_FILENAME="kimik2i0905_fp16_I1K_O1K_C64_sglang"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
KEEP_SERVER_ALIVE=false
BENCHMARK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-server-alive)
            KEEP_SERVER_ALIVE=true
            shift
            ;;
        --benchmark-only)
            BENCHMARK_ONLY=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--keep-server-alive] [--benchmark-only]"
            exit 1
            ;;
    esac
done

check_env_vars \
    MODEL \
    PRECISION \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    IMAGE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# NOTE: SGLang's aiter attention backend does not currently support num_heads=8
# (used by Kimi K2.5 MLA architecture), so triton is used for all precisions.
# Ref: https://github.com/vllm-project/vllm/issues/35641 (same root constraint)

if [[ "$PRECISION" == "fp8" ]]; then
    QUANTIZATION_ARGS="--quantization fp8 --kv-cache-dtype fp8_e5m2"
elif [[ "$PRECISION" == "bf16" ]]; then
    QUANTIZATION_ARGS=""
else
    echo "Error: PRECISION must be 'fp8' or 'bf16', got '${PRECISION}'"
    exit 1
fi

export RCCL_MSCCL_ENABLE=0
PORT=${PORT:-8888}
WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
OUT_SERVER_LOG=${WORKSPACE}/server.log
IN_SERVER_LOG=/workspace/server.log
NAME="akao_kimik2_sglang_server"

set -x

# ---------------------------------------------------------------------------
# Server launch — skipped in --benchmark-only mode
# ---------------------------------------------------------------------------
if [[ "$BENCHMARK_ONLY" == "false" ]]; then
    docker run --rm -d \
        --name "${NAME}" \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --ipc=host \
        --network=host \
        -v "${WORKSPACE}/:/workspace/" \
        -v "${HF_HUB_CACHE}:${HF_HUB_CACHE}" \
        -e HF_HOME="${HF_HOME}" \
        -e HF_HUB_CACHE="${HF_HUB_CACHE}" \
        -e RCCL_MSCCL_ENABLE \
        -e SGLANG_USE_AITER=1 \
        -e PYTHONPATH="/sgl-workspace/aiter" \
        -e SGLANG_ROCM_FUSED_DECODE_MLA=0 \
        "${IMAGE}" \
        bash -lc "\
            python -m sglang.launch_server \
            --decode-attention-backend triton \
            --prefill-attention-backend aiter \
            --model-path "${MODEL}" \
            --host 0.0.0.0 \
            --port "${PORT}" \
            --tensor-parallel-size "${TP}" \
            --trust-remote-code \
            --chunked-prefill-size 131072 \
            --mem-fraction-static 0.95 \
            --disable-radix-cache \
            --num-continuous-decode-steps 4 \
            --cuda-graph-max-bs "${CONC}" \
	     --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 \
            ${QUANTIZATION_ARGS} > "${IN_SERVER_LOG}" 2>&1"

    sleep 3
    SERVER_PID=$(docker inspect -f '{{.State.Pid}}' ${NAME})

    # Wait for server to be ready
    wait_for_server_ready --port "${PORT}" --server-log "${OUT_SERVER_LOG}" --server-pid "${SERVER_PID}"
else
    echo "[benchmark-only] Skipping server launch; assuming server is already running on port ${PORT}."
fi

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
docker exec -it "${NAME}" \
    bash -lc "source /workspace/benchmarks/benchmark_lib.sh; \
    run_benchmark_serving \
    --model "${MODEL}" \
    --port "${PORT}" \
    --backend vllm \
    --input-len "${ISL}" \
    --output-len "${OSL}" \
    --random-range-ratio "${RANDOM_RANGE_RATIO}" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "${CONC}" \
    --result-filename "${RESULT_FILENAME}" \
    --result-dir /workspace \
    --bench-serving-dir /workspace \
    --trust-remote-code"

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "${PORT}" --concurrent-requests "${CONC}"
    append_lm_eval_summary
fi

# ---------------------------------------------------------------------------
# Server teardown — skipped when --keep-server-alive or --benchmark-only
# ---------------------------------------------------------------------------
if [[ "$KEEP_SERVER_ALIVE" == "false" && "$BENCHMARK_ONLY" == "false" ]]; then
    docker stop ${NAME}
else
    echo "[keep-server-alive] Container '${NAME}' left running."
fi

set +x
