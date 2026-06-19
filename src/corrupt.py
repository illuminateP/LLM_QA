import os
import json
import random
import argparse

SPECIAL_CHARS = ["!", "@", "#", "$", "%", "^", "&", "*"]
RATIOS = [0.1, 0.2, 0.3]


# --- 1. Deletion (데이터 결측률) -------------------------------------------------
def get_deletion_noise(text, answer_text, answer_start, ratio):
    answer_end = answer_start + len(answer_text)
    ans_text_original = text[answer_start:answer_end]
    pre_text = text[:answer_start]
    post_text = text[answer_end:]
    pre_tokens = pre_text.split(" ")
    post_tokens = post_text.split(" ")

    def delete_tokens(tokens, r):
        if not tokens:
            return []
        valid_indices = [i for i, t in enumerate(tokens) if t.strip()]
        n_delete = int(len(valid_indices) * r)
        if n_delete == 0:
            return tokens
        indices_to_delete = set(random.sample(valid_indices, n_delete))
        return [t for i, t in enumerate(tokens) if i not in indices_to_delete]

    new_pre_text = " ".join(delete_tokens(pre_tokens, ratio))
    if len(new_pre_text) > 0 and not new_pre_text.endswith(" ") and pre_text.endswith(" "):
        new_pre_text += " "
    new_answer_start = len(new_pre_text)
    new_post_text = " ".join(delete_tokens(post_tokens, ratio))
    if len(new_post_text) > 0 and not new_post_text.startswith(" ") and post_text.startswith(" "):
        new_post_text = " " + new_post_text
    new_context = new_pre_text + ans_text_original + new_post_text
    return new_context, new_answer_start, ans_text_original


# --- 2. Context Insertion (구문 정확성 저하) -------------------------------------
def get_context_insertion_noise(text, answer_text, answer_start, ratio):
    answer_end = answer_start + len(answer_text)
    pre_text = list(text[:answer_start])
    ans_text_original = text[answer_start:answer_end]
    post_text = list(text[answer_end:])

    def insert_noise(char_list, r):
        if not char_list:
            return ""
        n_insert = int(len(char_list) * r)
        for _ in range(n_insert):
            pos = random.randint(0, len(char_list))
            char_list.insert(pos, random.choice(SPECIAL_CHARS))
        return "".join(char_list)

    new_pre_text = insert_noise(pre_text, ratio)
    new_post_text = insert_noise(post_text, ratio)
    new_answer_start = len(new_pre_text)
    new_context = new_pre_text + ans_text_original + new_post_text
    return new_context, new_answer_start, ans_text_original


# --- 3. Answer Insertion (라벨링 정확성 저하) ------------------------------------
def get_answer_insertion_noise(text, answer_text, answer_start, ratio):
    answer_end = answer_start + len(answer_text)
    pre_text = text[:answer_start]
    ans_chars = list(text[answer_start:answer_end])
    post_text = text[answer_end:]

    n_insert = max(1, int(len(ans_chars) * ratio))
    for _ in range(n_insert):
        pos = random.randint(1, len(ans_chars) - 1) if len(ans_chars) > 1 else 0
        ans_chars.insert(pos, random.choice(SPECIAL_CHARS))

    new_ans_text = "".join(ans_chars)
    new_context = pre_text + new_ans_text + post_text
    return new_context, answer_start, new_ans_text


def generate_all_datasets(input_file, out_dir):
    print(f"[load] {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    datasets = {"baseline": {"version": "KorQuAD_v1.0_baseline", "data": []}}
    for r in RATIOS:
        p = int(r * 100)
        datasets[f"deletion_{p}"] = {"version": f"KorQuAD_v1.0_deletion_{p}", "data": []}
        datasets[f"insertion_que_{p}"] = {"version": f"KorQuAD_v1.0_insertion_que_{p}", "data": []}
        datasets[f"insertion_ans_{p}"] = {"version": f"KorQuAD_v1.0_insertion_ans_{p}", "data": []}

    print("[corrupt] 모든 유형/비율 생성 중...")
    for doc in source_data["data"]:
        doc_structs = {k: {"title": doc["title"], "paragraphs": []} for k in datasets}
        for para in doc["paragraphs"]:
            original_context = para["context"]
            for qa in para["qas"]:
                ans = qa["answers"][0]
                a_text, a_start = ans["text"], ans["answer_start"]

                doc_structs["baseline"]["paragraphs"].append(
                    {"context": original_context, "qas": [qa]}
                )
                for r in RATIOS:
                    p = int(r * 100)
                    d_ctx, d_start, d_ans = get_deletion_noise(original_context, a_text, a_start, r)
                    doc_structs[f"deletion_{p}"]["paragraphs"].append({
                        "context": d_ctx,
                        "qas": [{"id": qa["id"], "question": qa["question"],
                                 "answers": [{"text": d_ans, "answer_start": d_start}]}],
                    })
                    q_ctx, q_start, q_ans = get_context_insertion_noise(original_context, a_text, a_start, r)
                    doc_structs[f"insertion_que_{p}"]["paragraphs"].append({
                        "context": q_ctx,
                        "qas": [{"id": qa["id"], "question": qa["question"],
                                 "answers": [{"text": q_ans, "answer_start": q_start}]}],
                    })
                    a_ctx, a_start2, a_ans = get_answer_insertion_noise(original_context, a_text, a_start, r)
                    doc_structs[f"insertion_ans_{p}"]["paragraphs"].append({
                        "context": a_ctx,
                        "qas": [{"id": qa["id"], "question": qa["question"],
                                 "answers": [{"text": a_ans, "answer_start": a_start2}]}],
                    })
        for k in datasets:
            if doc_structs[k]["paragraphs"]:
                datasets[k]["data"].append(doc_structs[k])

    os.makedirs(out_dir, exist_ok=True)
    print("[save] 파일 저장 중...")
    for key, content in datasets.items():
        path = os.path.join(out_dir, f"train_{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        print(f"  - {path}")


def main():
    parser = argparse.ArgumentParser(description="학습셋 오염 데이터 생성")
    parser.add_argument("--input", type=str, default="./datasets/bert_train.json")
    parser.add_argument("--out_dir", type=str, default="./datasets")
    parser.add_argument("--seed", type=int, default=42, help="재현성을 위한 난수 시드")
    args = parser.parse_args()

    random.seed(args.seed)
    generate_all_datasets(args.input, args.out_dir)
    print("[corrupt] 완료")


if __name__ == "__main__":
    main()
