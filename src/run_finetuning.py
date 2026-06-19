import os
import csv
import json
import sys
import argparse
import collections

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import evaluate

LOG_FIELDS = [
    "loss", "grad_norm", "learning_rate", "epoch", "step",
    "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
    "train_runtime", "train_samples_per_second", "train_steps_per_second",
    "total_flos", "train_loss",
]


class CSVLoggerCallback(TrainerCallback):
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "training_logs.csv")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = dict(logs)
        row["epoch"] = state.epoch
        row["step"] = state.global_step

        write_header = (not os.path.exists(self.log_path)) or os.path.getsize(self.log_path) == 0
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def prepare_train_features(examples, tokenizer, max_length, doc_stride):
    questions, contexts, answers = [], [], []
    for doc_paragraphs in examples["paragraphs"]:
        for para in doc_paragraphs:
            context = para["context"]
            for qa in para["qas"]:
                questions.append(qa["question"])
                contexts.append(context)
                answers.append(qa["answers"][0])

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
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


def prepare_validation_features(examples, tokenizer, max_length, doc_stride):
    questions, contexts, example_ids, answers = [], [], [], []
    for doc_paragraphs in examples["paragraphs"]:
        for para in doc_paragraphs:
            context = para["context"]
            for qa in para["qas"]:
                questions.append(qa["question"])
                contexts.append(context)
                example_ids.append(qa["id"])
                answers.append(qa["answers"][0])

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(example_ids[sample_index])

        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        offset_mapping = tokenized_examples["offset_mapping"][i]

        ans = answers[sample_index]
        start_char = ans["answer_start"]
        end_char = start_char + len(ans["text"])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offset_mapping[token_start_index][0] <= start_char and offset_mapping[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offset_mapping[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

        tokenized_examples["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping)
        ]

    return tokenized_examples


def preprocess_logits_for_metrics(logits, labels):
    return logits


metric = evaluate.load("squad")


def compute_metrics(p, validation_dataset, raw_dataset):
    start_logits, end_logits = p
    if isinstance(start_logits, tuple):
        start_logits = start_logits[0]
    if isinstance(end_logits, tuple):
        end_logits = end_logits[0]

    features = validation_dataset
    examples = raw_dataset
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in examples:
        para = example["paragraphs"]
        if isinstance(para, list):
            para = para[0]
        example_id = para["qas"][0]["id"]
        context = para["context"]

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
                        "text": context[start_char:end_char],
                    })

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

    references = []
    for ex in examples:
        para = ex["paragraphs"]
        if isinstance(para, list):
            para = para[0]
        qa = para["qas"][0]
        raw_answers = qa["answers"]
        formatted_answers = {
            "text": [ans["text"] for ans in raw_answers],
            "answer_start": [ans["answer_start"] for ans in raw_answers],
        }
        references.append({"id": qa["id"], "answers": formatted_answers})

    return metric.compute(predictions=predicted_answers, references=references)


def parse_args():
    parser = argparse.ArgumentParser(description="KorQuAD 추출형 QA 파인튜닝 · 평가")
    parser.add_argument("--experiment_name", type=str, required=True, help="결과가 저장될 실험 이름")
    parser.add_argument("--train_file", type=str, required=True, help="학습 데이터(JSON, KorQuAD 1.0 포맷)")
    parser.add_argument("--validation_file", type=str, default="./datasets/bert_validation.json")
    parser.add_argument("--output_root", type=str, default="./results", help="실험별 결과 폴더의 상위 경로")
    parser.add_argument("--model_checkpoint", type=str, default="klue/bert-base")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="지정 시 학습 샘플 수를 제한(샘플/스모크 테스트용)")
    parser.add_argument("--require_gpu", action="store_true",
                        help="GPU가 없으면 종료(실제 대규모 실험용 안전장치)")
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.require_gpu:
        print("[CRITICAL ERROR] GPU(CUDA) NOT FOUND. --require_gpu 가 설정되어 종료합니다.")
        sys.exit(1)
    else:
        device = torch.device("cpu")
        print("[WARN] GPU 미탐지 → CPU로 실행합니다(느림). 대규모 실험에는 GPU를 권장합니다.")

    experiment_name = args.experiment_name
    train_file = os.path.abspath(args.train_file)
    output_dir = os.path.join(args.output_root, experiment_name)

    print(f"[DEBUG] Experiment: {experiment_name}")

    # 재실행 시 멱등성 보장
    final_metric_path = os.path.join(output_dir, "final_test_metrics.json")
    if os.path.exists(final_metric_path):
        print(f"[INFO] '{experiment_name}' 이미 완료됨. 건너뜁니다...")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)
    model.to(device)

    raw_datasets = load_dataset(
        "json",
        data_files={"train": train_file, "validation": args.validation_file},
        field="data",
    )

    train_split = raw_datasets["train"].train_test_split(test_size=0.2, seed=42)
    train_ds_raw = train_split["train"]
    eval_ds_raw = train_split["test"]
    test_ds_raw = raw_datasets["validation"]

    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(train_ds_raw))
        train_ds_raw = train_ds_raw.select(range(n))
        print(f"[INFO] max_train_samples 적용 → 학습 문서 {n}개로 제한")

    fn_train = lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride)
    fn_valid = lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride)

    train_dataset = train_ds_raw.map(fn_train, batched=True, remove_columns=train_ds_raw.column_names)
    eval_dataset = eval_ds_raw.map(fn_valid, batched=True, remove_columns=eval_ds_raw.column_names)
    test_dataset = test_ds_raw.map(fn_valid, batched=True, remove_columns=test_ds_raw.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
        callbacks=[CSVLoggerCallback(output_dir)],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint is not None:
        print(f"[INFO] 체크포인트에서 재개: {last_checkpoint}")
    else:
        print("[INFO] 처음부터 학습 시작")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(os.path.join(output_dir, "final_model"))

    # 최종 평가: F1/EM 과 test_loss 산출
    predictions = trainer.predict(test_dataset)
    metrics = compute_metrics(predictions.predictions, test_dataset, test_ds_raw)
    if predictions.metrics and "test_loss" in predictions.metrics:
        metrics["test_loss"] = predictions.metrics["test_loss"]

    os.makedirs(output_dir, exist_ok=True)
    with open(final_metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[DONE] {experiment_name} → {metrics}")


if __name__ == "__main__":
    main()
