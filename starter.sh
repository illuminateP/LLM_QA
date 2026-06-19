#!/usr/bin/env bash
# =============================================================================
# 단일 진입점: 데이터 준비 → 오염 데이터 생성 → 10개 실험 학습/평가
#
# 사용법:
#   ./starter.sh sample   동봉된 소량 샘플로 전체 파이프라인을 빠르게 검증(GPU 불필요)
#   ./starter.sh full     실제 AI Hub 데이터로 본 실험 수행(datasets/raw 필요, GPU 권장)
#   ./starter.sh          = full
#
# 이 스크립트 하나만 실행하면 prepare → corrupt → finetune → evaluate 가
# 순서대로 수행되어 results/<실험명>/final_test_metrics.json 까지 자동 생성된다.
# (과거의 별도 로그 복구/재평가 단계는 run_finetuning.py 에 통합되어 더 이상 필요 없음)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"                       # 항상 저장소 루트 기준으로 동작

MODE="${1:-full}"

# 실험 목록(정상군 + 9개 오염군). 결과 폴더명으로도 사용된다.
EXPERIMENTS=(
  baseline
  deletion_10 deletion_20 deletion_30
  insertion_que_10 insertion_que_20 insertion_que_30
  insertion_ans_10 insertion_ans_20 insertion_ans_30
)

if [ "$MODE" = "sample" ]; then
  DATA_DIR="datasets/sample"
  RAW_DIR="datasets/sample/raw"
  TRAIN_OPTS=(--epochs 1 --batch_size 4 --eval_steps 1 --save_steps 1 --logging_steps 1 --max_train_samples 50)

  RESULT_ROOT="results"
  for exp in "${EXPERIMENTS[@]}"; do
    if [ -e "results/${exp}" ]; then
      RESULT_ROOT="results/sample"
      echo "[INFO] results/ 에 기존 결과가 있어 샘플 결과는 '${RESULT_ROOT}' 에 분리 저장합니다."
      break
    fi
  done
else
  DATA_DIR="datasets"
  RAW_DIR="datasets/raw"
  RESULT_ROOT="results"
  TRAIN_OPTS=(--require_gpu)
fi

mkdir -p logs "$RESULT_ROOT"

echo "========================================================"
echo "LLM_QA 파이프라인 시작 (mode=$MODE)"
echo "========================================================"

# --- 1) 데이터 준비: bert_train.json / bert_validation.json ---
if [ ! -f "$DATA_DIR/bert_train.json" ] || [ ! -f "$DATA_DIR/bert_validation.json" ]; then
  if [ -d "$RAW_DIR/1.Training" ]; then
    echo "[1/3] 원천 데이터 → KorQuAD 변환"
    python -u src/prepare_data.py --raw_dir "$RAW_DIR" --out_dir "$DATA_DIR"
  else
    echo "[ERROR] 학습/검증 데이터를 찾을 수 없습니다."
    echo "  full 모드: AI Hub 원천을 '$RAW_DIR/{1.Training,2.Validation}/' 에 배치하세요."
    echo "  자세한 절차는 datasets/README.md 를 참고하세요."
    exit 1
  fi
else
  echo "[1/3] 학습/검증 데이터 확인 완료 → 변환 건너뜀"
fi

# --- 2) 오염 데이터 10종 생성(없을 때만) ---
if [ ! -f "$DATA_DIR/train_baseline.json" ]; then
  echo "[2/3] 오염 데이터 10종 생성"
  python -u src/corrupt.py --input "$DATA_DIR/bert_train.json" --out_dir "$DATA_DIR"
else
  echo "[2/3] 오염 데이터 확인 완료 → 생성 건너뜀"
fi

# --- 3) 10개 실험 순차 학습/평가 ---
echo "[3/3] ${#EXPERIMENTS[@]}개 실험 시작"
idx=0
for exp in "${EXPERIMENTS[@]}"; do
  idx=$((idx + 1))
  echo ""
  echo "-------- [$idx/${#EXPERIMENTS[@]}] $exp --------"
  python -u src/run_finetuning.py \
    --experiment_name "$exp" \
    --train_file "$DATA_DIR/train_${exp}.json" \
    --validation_file "$DATA_DIR/bert_validation.json" \
    --output_root "$RESULT_ROOT" \
    "${TRAIN_OPTS[@]}" \
    2>&1 | tee -a "logs/log_${exp}.txt"
done

echo ""
echo "========================================================"
echo "전체 완료. 결과: $RESULT_ROOT/<실험명>/final_test_metrics.json"
echo "========================================================"
