import os
import json
import pandas as pd
import numpy as np
import torch
import collections
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    TrainerCallback
)
from datasets import load_dataset
import evaluate

# ==============================================================================
# [CONFIG] 파일별로 이 부분만 수정해서 저장하세요.
# ==============================================================================
# 1. run_finetuning_baseline.py
EXPERIMENT_NAME = "baseline"
TRAIN_FILE = "./dataset/cleaned/bert_train.json"

# 2. run_finetuning_deletion.py
# EXPERIMENT_NAME = "deletion"
# TRAIN_FILE = "./dataset/corrupted/train_deletion.json"

# 3. run_finetuning_insertion_que.py
# EXPERIMENT_NAME = "insertion_que"
# TRAIN_FILE = "./dataset/corrupted/train_insertion_que.json"

# 4. run_finetuning_insertion_ans.py
# EXPERIMENT_NAME = "insertion_ans"
# TRAIN_FILE = "./dataset/corrupted/train_insertion_ans.json"
# ==============================================================================

# 공통 설정
VALIDATION_FILE = "./dataset/cleaned/bert_validation.json"
OUTPUT_DIR = f"./output/{EXPERIMENT_NAME}"
MODEL_CHECKPOINT = "klue/bert-base"
MAX_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 16
EPOCHS = 3

# GPU 설정 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. CSV Logger Callback ---
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, "training_logs.csv")
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_entry = logs.copy()
            log_entry['epoch'] = state.epoch
            log_entry['step'] = state.global_step
            self.logs.append(log_entry)
            df = pd.DataFrame(self.logs)
            df.to_csv(self.log_path, index=False)

# --- 2. 전처리 함수 (Sliding Window 적용) ---
def prepare_train_features(examples, tokenizer):
    questions = [q['qas'][0]['question'] for q in examples['paragraphs']]
    contexts = [p['context'] for p in examples['paragraphs']]
    answers = [q['qas'][0]['answers'][0] for q in examples['paragraphs']]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        ans = answers[sample_index]
        start_char = ans["answer_start"]
        end_char = start_char + len(ans["text"])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# --- 3. 검증 데이터 전처리 (평가용) ---
def prepare_validation_features(examples, tokenizer):
    questions = [q['qas'][0]['question'] for q in examples['paragraphs']]
    contexts = [p['context'] for p in examples['paragraphs']]
    example_ids = [q['qas'][0]['id'] for q in examples['paragraphs']]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(example_ids[sample_index])
        sequence_ids = tokenized_examples.sequence_ids(i)
        offset_mapping = tokenized_examples["offset_mapping"][i]
        tokenized_examples["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping)
        ]

    return tokenized_examples

# --- 4. Logits Preprocessing Function ---
def preprocess_logits_for_metrics(logits, labels):
    """
    Evaluation 단계에서 메모리를 절약하기 위해 Logits을 가공하는 함수입니다.
    Trainer 인자로 전달되어 사용됩니다.
    
    여기서는 원본 Logits(tuple)을 그대로 반환하여 compute_metrics에서 
    정교한 Span 탐색(start < end 등)을 수행할 수 있도록 합니다.
    필요 시 여기서 torch.argmax 등으로 변환하여 메모리를 더 아낄 수 있으나,
    SQuAD F1 계산을 위해서는 전체 확률 분포가 필요할 수 있습니다.
    """
    if isinstance(logits, tuple):
        # (start_logits, end_logits) 형태
        return logits
    return logits

# --- 5. F1, EM 계산 함수 ---
metric = evaluate.load("squad")

def compute_metrics(p, validation_dataset, raw_dataset):
    start_logits, end_logits = p
    
    if isinstance(start_logits, tuple): start_logits = start_logits[0]
    if isinstance(end_logits, tuple): end_logits = end_logits[0]
    
    features = validation_dataset
    examples = raw_dataset
    
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    print("Computing metrics...")
    
    for example in examples:
        para = example['paragraphs']
        if isinstance(para, list): para = para[0]
        example_id = para['qas'][0]['id']
        context = para['context']
        
        feature_indices = example_to_features.get(example_id, [])
        valid_answers = []
        
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1:-21:-1].tolist()
            end_indexes = np.argsort(end_logit)[-1:-21:-1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > 30:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    
                    valid_answers.append({
                        "score": start_logit[start_index] + end_logit[end_index],
                        "text": context[start_char:end_char]
                    })
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

    references = []
    for ex in examples:
        para = ex['paragraphs']
        if isinstance(para, list): para = para[0]
        qa = para['qas'][0]
        references.append({"id": qa['id'], "answers": qa['answers']})

    results = metric.compute(predictions=predicted_answers, references=references)
    return results

# --- Main Execution ---
def main():
    print(f"[DEBUG] Experiment: {EXPERIMENT_NAME}")
    print(f"[DEBUG] Loading Train: {TRAIN_FILE}")
    print(f"[DEBUG] Loading Valid (Final Test): {VALIDATION_FILE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)
    model.to(device)

    # 1. 데이터셋 로드
    raw_datasets = load_dataset(
        'json', 
        data_files={'train': TRAIN_FILE, 'validation': VALIDATION_FILE}, 
        field='data'
    )
    
    # [수정됨] Train 데이터를 8:2로 분할하여 (Train / Eval)로 사용
    print("[DEBUG] Splitting Train file into Train set (80%) and Eval set (20%)...")
    train_split = raw_datasets["train"].train_test_split(test_size=0.2, seed=42)
    
    train_ds_raw = train_split["train"] # 실제 학습용 (80%)
    eval_ds_raw = train_split["test"]   # 학습 중 검증용 (20%)
    test_ds_raw = raw_datasets["validation"] # 최종 평가용 (Original Validation File)

    print(f"   - Train Samples : {len(train_ds_raw)}")
    print(f"   - Eval Samples  : {len(eval_ds_raw)} (Used for monitoring)")
    print(f"   - Test Samples  : {len(test_ds_raw)} (Original Validation File)")

    # 2. 전처리
    train_dataset = train_ds_raw.map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=train_ds_raw.column_names
    )
    
    eval_dataset = eval_ds_raw.map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=eval_ds_raw.column_names
    )
    
    test_dataset = test_ds_raw.map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=test_ds_raw.column_names
    )

    # 3. Trainer 설정
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=3e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Train에서 분할된 20% 사용
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
        callbacks=[CSVLoggerCallback(OUTPUT_DIR)],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics # 요청된 옵션 추가
    )

    # 4. 학습 시작
    print("[DEBUG] Starting Training...")
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    
    # 5. 최종 평가 (Original Validation Set 사용)
    print("[DEBUG] Evaluating on Final Test Set (Original Validation File)...")
    
    predictions = trainer.predict(test_dataset)
    metrics = compute_metrics(predictions.predictions, test_dataset, test_ds_raw)
    
    print(f"Final Test Metrics: {metrics}")
    
    with open(os.path.join(OUTPUT_DIR, "final_test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[DEBUG] Experiment {EXPERIMENT_NAME} Finished Successfully.")

if __name__ == "__main__":
    main()