#!/bin/bash
# Triclassifier (3 binary inferences + majority vote) sweep
# 22 cells: 2 models × 11 projects, project-specific only, K=6
# Idempotent: skips a cell if preds_k6.csv already exists.
set -e
cd /home/ahosein/llm-labler

PROJECTS=(
    "ansible_ansible"
    "bitcoin_bitcoin"
    "dart-lang_sdk"
    "dotnet_roslyn"
    "facebook_react"
    "flutter_flutter"
    "kubernetes_kubernetes"
    "microsoft_TypeScript"
    "microsoft_vscode"
    "opencv_opencv"
    "tensorflow_tensorflow"
)

MODELS=(
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit:unsloth_Qwen2_5_3B_Instruct_bnb_4bit"
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit:unsloth_Qwen2_5_7B_Instruct_bnb_4bit"
)

K=6
POOL_K=30

run_cell() {
    local model="$1"
    local tag="$2"
    local proj="$3"

    local out_dir="results/issues11k/project_specific/${proj}/${tag}/triclassifier"
    local neighbors_csv="results/issues11k/project_specific/${proj}/neighbors/neighbors_k${POOL_K}.csv"
    local preds_csv="${out_dir}/predictions/preds_k${K}.csv"

    if [[ -f "${preds_csv}" ]]; then
        echo "    SKIP: ${preds_csv} already exists"
        return 0
    fi

    if [[ ! -f "${neighbors_csv}" ]]; then
        echo "    FAIL: neighbors file not found: ${neighbors_csv}"
        return 1
    fi

    mkdir -p "${out_dir}"
    echo "    Output: ${out_dir}"
    python tri_classifier.py \
        --model "${model}" \
        --neighbors_csv "${neighbors_csv}" \
        --output_dir "${out_dir}" \
        --top_k "${K}" \
        --vote_pool_k "${POOL_K}" \
        --max_seq_length 8192 \
        --max_new_tokens 20 \
        --inference_batch_size 8 \
        --model_name_for_eval "${tag}"

    if [[ -f "${preds_csv}" ]]; then
        echo "    DONE"
    else
        echo "    FAIL: no preds produced"
        return 1
    fi
}

START_TS=$(date +%s)
echo "=== Triclassifier campaign starting at $(date) ==="

for entry in "${MODELS[@]}"; do
    model="${entry%%:*}"
    tag="${entry##*:}"

    echo ""
    echo "############################################"
    echo "## MODEL: ${model}"
    echo "############################################"

    for proj in "${PROJECTS[@]}"; do
        echo ""
        echo "[${proj}] ${tag}"
        run_cell "${model}" "${tag}" "${proj}"
    done
done

END_TS=$(date +%s)
echo ""
echo "=== Done at $(date) (elapsed: $(( (END_TS - START_TS) / 60 )) min) ==="
echo ""
echo "=== Summary of triclassifier F1 macro per cell ==="
for entry in "${MODELS[@]}"; do
    tag="${entry##*:}"
    short="${tag#unsloth_}"
    short="${short%_Instruct_bnb_4bit}"
    for proj in "${PROJECTS[@]}"; do
        eval_csv="results/issues11k/project_specific/${proj}/${tag}/triclassifier/evaluations/eval_k${K}.csv"
        if [[ -f "${eval_csv}" ]]; then
            f1=$(awk -F, 'NR==2{print $21}' "${eval_csv}" 2>/dev/null || echo "N/A")
            printf "  %-12s %-25s %s\n" "${short}" "${proj}" "${f1}"
        fi
    done
done
