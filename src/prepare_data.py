import os
import glob
import json
import argparse


def load_raw_records(folder):
    records = []
    file_list = sorted(glob.glob(os.path.join(folder, "*.json")))
    print(f"[scan] '{folder}' → JSON {len(file_list)}개")
    for fp in file_list:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] 읽기 실패 {fp}: {e}")
            continue

        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            inner = next((v for v in data.values() if isinstance(v, list)), None)
            records.extend(inner if inner is not None else [data])
    return records


def convert_to_korquad(records, version_tag):
    stats = {"docs": len(records), "qas": 0, "valid": 0, "no_index": 0, "empty": 0}
    out = {"version": version_tag, "data": []}

    for doc in records:
        k_doc = {"title": doc.get("doc_title", "No Title"), "paragraphs": []}
        for para in doc.get("paragraphs", []):
            context = para["context"]
            qas_list = []
            for qa in para["qas"]:
                stats["qas"] += 1
                answer_text = qa["answers"].get("text", "").strip()
                if not answer_text:
                    stats["empty"] += 1
                    continue
                start_idx = context.find(answer_text)
                if start_idx == -1:
                    stats["no_index"] += 1
                    continue
                qas_list.append({
                    "id": qa["question_id"],
                    "question": qa["question"],
                    "answers": [{"text": answer_text, "answer_start": start_idx}],
                })
                stats["valid"] += 1
            if qas_list:
                k_doc["paragraphs"].append({"context": context, "qas": qas_list})
        if k_doc["paragraphs"]:
            out["data"].append(k_doc)

    print(f"  [convert:{version_tag}] 문서 {stats['docs']} / QA {stats['qas']} / "
          f"유효 {stats['valid']} / 정답미발견 {stats['no_index']} / 빈정답 {stats['empty']}")
    return out


def build(raw_dir, out_dir, split_dir, out_name, version_tag):
    records = load_raw_records(os.path.join(raw_dir, split_dir))
    if not records:
        print(f"  [warn] {split_dir} 레코드 없음 — 건너뜀")
        return
    converted = convert_to_korquad(records, version_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"  [save] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Hub 원천 → KorQuAD 변환")
    parser.add_argument("--raw_dir", type=str, default="./datasets/raw",
                        help="원천 데이터 루트(하위에 1.Training, 2.Validation)")
    parser.add_argument("--out_dir", type=str, default="./datasets")
    args = parser.parse_args()

    build(args.raw_dir, args.out_dir, "1.Training", "bert_train.json", "KorQuAD_v1.0_train")
    build(args.raw_dir, args.out_dir, "2.Validation", "bert_validation.json", "KorQuAD_v1.0_validation")
    print("[prepare_data] 완료")


if __name__ == "__main__":
    main()
