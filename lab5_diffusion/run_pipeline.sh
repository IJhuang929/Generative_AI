#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh
# 自動執行 train.py -> inference.py -> score.py 完整流程
#
# 用法:
#   bash run_pipeline.sh                        # 從頭訓練
#   bash run_pipeline.sh --resume outputs/20260603_1433/checkpoints/unet_5000
#   bash run_pipeline.sh --skip-train           # 跳過訓練，直接 inference + score
#   bash run_pipeline.sh --skip-train --skip-inference  # 只跑 score
#   bash run_pipeline.sh --resume outputs/20260604_2340/checkpoints/unet_final --reset-lr
# =============================================================================

set -e

# ========== 預設參數 ==========
RESUME_FROM=""
SKIP_TRAIN=false
SKIP_INFERENCE=false
RESET_LR=false

# 自動找最新的 run 目錄（outputs/YYYYMMDD_HHMM/）
LATEST_RUN=$(ls -td outputs/20*/ 2>/dev/null | head -1)
CHECKPOINT_DIR="${LATEST_RUN}checkpoints/unet_final"
OUTPUT_DIR="scoring_program/input/res"
SCORING_INPUT="scoring_program/input"
SCORING_OUTPUT="scoring_program/output"
SCORE_PY="scoring_program/score.py"
SCORES_HISTORY="outputs/scores_history.json"

NUM_SAMPLES=3000
BATCH_SIZE=8
DDIM_STEPS=100

# ========== 顏色輸出 ==========
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ========== 解析參數 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)        RESUME_FROM="$2"; shift 2 ;;
        --skip-train)    SKIP_TRAIN=true; shift ;;
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --reset-lr)      RESET_LR=true; shift ;;
        --checkpoint)    CHECKPOINT_DIR="$2"; shift 2 ;;
        --ddim-steps)    DDIM_STEPS="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        *) log_error "未知參數: $1"; exit 1 ;;
    esac
done

# ========== 顯示執行設定 ==========
echo "=============================================="
echo "  HW5 Diffusion Pipeline"
echo "=============================================="
log_info "Skip train:      $SKIP_TRAIN"
log_info "Skip inference:  $SKIP_INFERENCE"
log_info "Resume from:     ${RESUME_FROM:-'(none, 從頭訓練)'}"
log_info "Checkpoint dir:  $CHECKPOINT_DIR"
log_info "DDIM steps:      $DDIM_STEPS"
log_info "Batch size:      $BATCH_SIZE"
log_info "Num samples:     $NUM_SAMPLES"
echo "=============================================="

START_TIME=$(date +%s)

# ========== Step 1: 訓練 ==========
if [ "$SKIP_TRAIN" = false ]; then
    log_info "▶ Step 1/3: 開始訓練..."
    TRAIN_START=$(date +%s)

    if [ -n "$RESUME_FROM" ]; then
        log_info "從 checkpoint resume: $RESUME_FROM"
        RESET_LR_FLAG=""
        [ "$RESET_LR" = true ] && RESET_LR_FLAG="--reset_lr"
        uv run python train.py --resume_from "$RESUME_FROM" $RESET_LR_FLAG
    else
        uv run python train.py
    fi

    # 訓練完後更新 LATEST_RUN 和 CHECKPOINT_DIR
    LATEST_RUN=$(ls -td outputs/20*/ 2>/dev/null | head -1)
    CHECKPOINT_DIR="${LATEST_RUN}checkpoints/unet_final"

    TRAIN_END=$(date +%s)
    TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
    log_info "✓ 訓練完成，耗時 $(( TRAIN_ELAPSED / 3600 ))h $(( (TRAIN_ELAPSED % 3600) / 60 ))m"
else
    log_warning "⏭ 跳過訓練"
fi

# ========== Step 2: Inference ==========
if [ "$SKIP_INFERENCE" = false ]; then
    log_info "▶ Step 2/3: 開始生成 $NUM_SAMPLES 張圖片..."
    INFER_START=$(date +%s)

    # 確認 checkpoint 存在，找不到就 fallback
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        FALLBACK="${LATEST_RUN}checkpoints/unet_final"
        log_warning "找不到 $CHECKPOINT_DIR，嘗試 fallback: $FALLBACK"
        if [ ! -d "$FALLBACK" ]; then
            log_error "找不到任何 checkpoint，請先訓練或指定 --checkpoint"
            exit 1
        fi
        CHECKPOINT_DIR="$FALLBACK"
    fi

    # 清空舊的生成結果
    if [ -d "$OUTPUT_DIR" ]; then
        log_warning "清空舊的生成結果: $OUTPUT_DIR"
        rm -f "$OUTPUT_DIR"/*.png
    fi
    mkdir -p "$OUTPUT_DIR"

    uv run python inference.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --sampler ddim \
        --num_steps "$DDIM_STEPS"

    ACTUAL_COUNT=$(ls "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l)
    if [ "$ACTUAL_COUNT" -ne "$NUM_SAMPLES" ]; then
        log_error "生成圖片數量不符：預期 $NUM_SAMPLES，實際 $ACTUAL_COUNT"
        exit 1
    fi

    INFER_END=$(date +%s)
    INFER_ELAPSED=$(( INFER_END - INFER_START ))
    log_info "✓ 生成完成（$ACTUAL_COUNT 張），耗時 $(( INFER_ELAPSED / 60 ))m $(( INFER_ELAPSED % 60 ))s"
else
    log_warning "⏭ 跳過 Inference"
fi

# ========== Step 3: 計算 FID ==========
log_info "▶ Step 3/3: 計算 FID 分數..."
mkdir -p "$SCORING_OUTPUT"

if [ ! -f "$SCORE_PY" ]; then
    log_error "找不到 $SCORE_PY"
    exit 1
fi

uv run python "$SCORE_PY" \
    --input_dir "$SCORING_INPUT" \
    --output_dir "$SCORING_OUTPUT" \
    --image_size 256 \
    --num_images "$NUM_SAMPLES" \
    --scores fid

# ========== 顯示結果 + Append 歷史 ==========
if [ -f "$SCORING_OUTPUT/scores.json" ]; then
    FID_NUM=$(python3 -c "import json; d=json.load(open('$SCORING_OUTPUT/scores.json')); print(d.get('FID', d.get('fid', 999)))" 2>/dev/null || echo "999")
    RUN_NAME=$(basename "${LATEST_RUN%/}" 2>/dev/null || echo "manual")
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

    echo ""
    echo "=============================================="
    log_info "✓ FID 分數：$FID_NUM"

    if (( $(echo "$FID_NUM < 50" | python3 -c "import sys; print(int(eval(sys.stdin.read())))") )); then
        BASELINE="strong (FID < 50)"
        log_info "🎉 達到 strong baseline（FID < 50）！可以提交！"
    elif (( $(echo "$FID_NUM < 90" | python3 -c "import sys; print(int(eval(sys.stdin.read())))") )); then
        BASELINE="weak (FID < 90)"
        log_warning "⚠ 達到 weak baseline（FID < 90），還有改善空間"
    else
        BASELINE="below baseline"
        log_error "✗ FID > 90，需要繼續優化"
    fi
    echo "=============================================="

    # Append 到 scores_history.json
    python3 -c "
import json, os
history_path = '$SCORES_HISTORY'
entry = {
    'timestamp':   '$TIMESTAMP',
    'run_id':      '$RUN_NAME',
    'checkpoint':  '$CHECKPOINT_DIR',
    'fid':         $FID_NUM,
    'ddim_steps':  $DDIM_STEPS,
    'num_samples': $NUM_SAMPLES,
    'baseline':    '$BASELINE',
}
history = json.load(open(history_path)) if os.path.exists(history_path) else []
history.append(entry)
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2, ensure_ascii=False)
print()
print(f'FID 歷史記錄（共 {len(history)} 筆）：')
print(f\"{'Timestamp':<20} {'Run ID':<18} {'FID':>8}  Baseline\")
print('-' * 68)
for h in history:
    print(f\"{h['timestamp']:<20} {h['run_id']:<18} {h['fid']:>8.2f}  {h['baseline']}\")
"
else
    log_error "找不到 scores.json，score.py 可能執行失敗"
    exit 1
fi

# ========== 總耗時 ==========
END_TIME=$(date +%s)
TOTAL=$(( END_TIME - START_TIME ))
log_info "總耗時：$(( TOTAL / 3600 ))h $(( (TOTAL % 3600) / 60 ))m $(( TOTAL % 60 ))s"

# ========== 提醒打包上傳 ==========
if (( $(echo "$FID_NUM < 90" | python3 -c "import sys; print(int(eval(sys.stdin.read())))") )); then
    echo ""
    log_info "📦 打包上傳指令："
    echo "    cd $OUTPUT_DIR && zip -r ../../../submission.zip *.png"
fi