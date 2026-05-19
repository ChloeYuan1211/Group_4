#!/bin/bash
# eval_demo_clean_model.sh
# Run demo_clean evaluation for a single checkpoint path.
#
# Usage:
#   bash eval_demo_clean_model.sh [gpu_id] [batch_size] [total_episodes] [checkpoint_path]
#
# Defaults:
#   gpu_id=4
#   batch_size=4
#   total_episodes=100
#   checkpoint_path=/data1/jiangshaohan/tuxiaoxiao/aloha_move_can_pot_100000_chkpt_extracted

gpu_id=${1:-6}
batch_size=${2:-4}
total_episodes=${3:-100}
checkpoint_path=${4:-/data1/jiangshaohan/tuxiaoxiao/openvla-oft-dino-checkpoints/dino_ft_merged_openvla_checkpoints/home/qdhe/workspace/dda4210/project/RoboTwin/policy/openvla-oft/dino_finetune_v2/merged_openvla_checkpoints/full_last4_run1/step_0002000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK_NAME="move_can_pot"
TASK_CONFIG="demo_clean"
UNNORM_KEY="aloha_move_can_pot"
RESULT_FILE="/data1/jiangshaohan/DengFengting/eval_demo_clean_model_results.log"

# Patch demo_clean.yml: clear_cache_freq 5 -> 2 to reduce SAPIEN buffer issues.
CLEAN_YML="${REPO_ROOT}/task_config/demo_clean.yml"
if [ -f "${CLEAN_YML}" ] && grep -q "clear_cache_freq: 5" "${CLEAN_YML}"; then
    sed -i 's/clear_cache_freq: 5/clear_cache_freq: 2/' "${CLEAN_YML}"
    echo "[setup] Patched ${CLEAN_YML}: clear_cache_freq -> 2"
fi

if [ ! -e "${checkpoint_path}" ]; then
    echo "ERROR: checkpoint path not found: ${checkpoint_path}"
    exit 1
fi

echo "============================================"
echo "  eval_demo_clean_model.sh"
echo "  task:          ${TASK_NAME} / ${TASK_CONFIG}"
echo "  gpu:           ${gpu_id}"
echo "  batch_size:    ${batch_size}"
echo "  total_episodes:${total_episodes}"
echo "  checkpoint:    ${checkpoint_path}"
echo "  results log:   ${RESULT_FILE}"
echo "============================================"

output=$(bash "${SCRIPT_DIR}/eval_batch.sh" \
    "${TASK_NAME}" "${TASK_CONFIG}" "${checkpoint_path}" \
    0 "${gpu_id}" "${UNNORM_KEY}" "${batch_size}" "${total_episodes}" 2>&1 | tee /dev/stderr)

final=$(echo "${output}" | grep "FINAL RESULT:" | tail -1 | sed 's/.*FINAL RESULT: //')
final=${final:-PARSE ERROR}

echo ""
echo "============================================"
echo "  FINAL RESULT"
echo "  ${final}"
echo "============================================"

{
    echo "============================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] demo_clean model eval"
    echo "  task:        ${TASK_NAME} / ${TASK_CONFIG}"
    echo "  checkpoint:  ${checkpoint_path}"
    echo "  result:      ${final}"
    echo "============================================"
    echo ""
} >> "${RESULT_FILE}"