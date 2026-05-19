#!/bin/bash
# eval_full_last4_run1_last6_run1_all_ckpts.sh
# 跑 full_last4_run1 和 full_last6_run1 两个目录下所有 checkpoint，
# 并对 demo_randomized / demo_clean 两种模式分别评测。
#
# Usage:
#   bash eval_full_last4_run1_run2_all_ckpts.sh [gpu_id] [batch_size] [total_episodes]
#
# Defaults: gpu_id=4, batch_size=4, total_episodes=100

set -uo pipefail

gpu_id=${1:-6}
batch_size=${2:-1}
total_episodes=${3:-100}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

task_name="move_can_pot"
unnorm_key="aloha_move_can_pot"
seed_start=0

TASK_CONFIGS=(
    "demo_clean"
    "demo_randomized"
    
)

CHECKPOINT_ROOTS=(
    "/data1/jiangshaohan/tuxiaoxiao/openvla-oft-dino-checkpoints/dino_ft_merged_openvla_checkpoints/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/dino_finetune_v2/merged_openvla_checkpoints/full_last4_run1"
    "/data1/jiangshaohan/tuxiaoxiao/openvla-oft-dino-checkpoints/dino_ft_merged_openvla_checkpoints/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/dino_finetune_v2/merged_openvla_checkpoints/full_last6_run1"
)

RESULT_FILE="${SCRIPT_DIR}/eval_results_randomized.log"

declare -a CHECKPOINTS=()

for root in "${CHECKPOINT_ROOTS[@]}"; do
    if [[ ! -d "${root}" ]]; then
        echo "[WARN] checkpoint root not found: ${root}" | tee -a "${RESULT_FILE}"
        continue
    fi

    while IFS= read -r ckpt; do
        CHECKPOINTS+=("${ckpt}")
    done < <(find "${root}" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | sort)
done

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "[ERROR] No checkpoints found under configured roots." | tee -a "${RESULT_FILE}"
    exit 1
fi

echo "============================================"
echo "  eval_full_last4_run1_last6_run1_all_ckpts.sh"
echo "  gpu: ${gpu_id}"
echo "  batch_size: ${batch_size}"
echo "  total_episodes: ${total_episodes}"
echo "  task_name: ${task_name}"
echo "  task_configs: ${TASK_CONFIGS[*]}"
echo "  checkpoints: ${#CHECKPOINTS[@]}"
echo "  results log: ${RESULT_FILE}"
echo "============================================"

{
    echo ""
    echo "============================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START EVAL"
    echo "gpu=${gpu_id}, batch_size=${batch_size}, total_episodes=${total_episodes}"
    echo "task_name=${task_name}, unnorm_key=${unnorm_key}, seed_start=${seed_start}"
    echo "checkpoints=${#CHECKPOINTS[@]}"
    echo "============================================"
} >> "${RESULT_FILE}"

declare -A RESULTS
declare -a SUMMARY_KEYS

for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(basename "${ckpt}")"
    run_name="$(basename "$(dirname "${ckpt}")")"

    for task_config in "${TASK_CONFIGS[@]}"; do
        key="${run_name}/${ckpt_name}/${task_config}"

        echo ""
        echo "########################################"
        echo "  START: ${key}"
        echo "########################################"

        tmp_log="$(mktemp)"

        set +e
        bash "${SCRIPT_DIR}/eval_batch.sh" \
            "${task_name}" "${task_config}" "${ckpt}" \
            "${seed_start}" "${gpu_id}" "${unnorm_key}" \
            "${batch_size}" "${total_episodes}" 2>&1 | tee "${tmp_log}"

        exit_code=${PIPESTATUS[0]}
        set -e

        output="$(cat "${tmp_log}")"
        rm -f "${tmp_log}"

        final="$(echo "${output}" | grep "FINAL RESULT:" | tail -1)"

        if [[ -z "${final}" ]]; then
            final="FINAL RESULT NOT FOUND, exit_code=${exit_code}"
        fi

        RESULTS["${key}"]="${final}"
        SUMMARY_KEYS+=("${key}")

        echo ""
        echo "########################################"
        echo "  DONE: ${key}"
        echo "  exit_code=${exit_code}"
        echo "  ${final}"
        echo "########################################"

        {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: ${key}"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] exit_code=${exit_code}"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${final}"
            echo ""
        } >> "${RESULT_FILE}"
    done
done

echo ""
echo "============================================"
echo "  ALL CHECKPOINTS SUMMARY"
echo "============================================"

for key in "${SUMMARY_KEYS[@]}"; do
    echo "  ${key}"
    echo "    ${RESULTS[${key}]}"
done

echo "============================================"

{
    echo ""
    echo "============================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL CHECKPOINTS SUMMARY"
    echo "============================================"
    for key in "${SUMMARY_KEYS[@]}"; do
        echo "  ${key}"
        echo "    ${RESULTS[${key}]}"
    done
    echo "============================================"
} >> "${RESULT_FILE}"
