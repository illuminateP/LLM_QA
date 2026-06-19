# %% [markdown]
# ### 오염

# %%
import json
import random
import os
from tqdm import tqdm
import glob

# %%
# 설정
INPUT_FILE = "./datasets/bert_train.json"
OUTPUT_DIR = "./datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# 특수문자 풀
SPECIAL_CHARS = ['!', '@', '#', '$', '%', '^', '&', '*']
# 오염 비율 설정 (10%, 20%, 30%)
RATIOS = [0.1, 0.2, 0.3]

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- 1. Deletion (데이터 결측률) ---
def get_deletion_noise(text, answer_text, answer_start, ratio):
    answer_end = answer_start + len(answer_text)
    ans_text_original = text[answer_start:answer_end]
    
    pre_text = text[:answer_start]
    post_text = text[answer_end:]
    
    pre_tokens = pre_text.split(' ')
    post_tokens = post_text.split(' ')
    
    def delete_tokens(tokens, r):
        if not tokens: return []
        valid_indices = [i for i, t in enumerate(tokens) if t.strip()]
        n_delete = int(len(valid_indices) * r)
        
        if n_delete == 0: return tokens
        
        indices_to_delete = set(random.sample(valid_indices, n_delete))
        return [t for i, t in enumerate(tokens) if i not in indices_to_delete]

    new_pre_tokens = delete_tokens(pre_tokens, ratio)
    new_post_tokens = delete_tokens(post_tokens, ratio)
    
    new_pre_text = ' '.join(new_pre_tokens)
    if len(new_pre_text) > 0 and not new_pre_text.endswith(' ') and pre_text.endswith(' '):
        new_pre_text += ' '
        
    new_answer_start = len(new_pre_text)
    
    new_post_text = ' '.join(new_post_tokens)
    if len(new_post_text) > 0 and not new_post_text.startswith(' ') and post_text.startswith(' '):
        new_post_text = ' ' + new_post_text
        
    new_context = new_pre_text + ans_text_original + new_post_text
    
    return new_context, new_answer_start, ans_text_original

# --- 2. Context Insertion (구문 정확성 저하) ---
def get_context_insertion_noise(text, answer_text, answer_start, ratio):
    answer_end = answer_start + len(answer_text)
    
    pre_text = list(text[:answer_start])
    ans_text_original = text[answer_start:answer_end]
    post_text = list(text[answer_end:])
    
    def insert_noise(char_list, r):
        if not char_list: return ""
        n_insert = int(len(char_list) * r)
        for _ in range(n_insert):
            pos = random.randint(0, len(char_list))
            char = random.choice(SPECIAL_CHARS)
            char_list.insert(pos, char)
        return "".join(char_list)

    new_pre_text = insert_noise(pre_text, ratio)
    new_post_text = insert_noise(post_text, ratio)
    
    new_answer_start = len(new_pre_text)
    new_context = new_pre_text + ans_text_original + new_post_text
    
    return new_context, new_answer_start, ans_text_original

# --- 3. Answer Insertion (라벨링 정확성 저하) ---
def get_answer_insertion_noise(text, answer_text, answer_start, ratio):
    answer_end = answer_start + len(answer_text)
    
    pre_text = text[:answer_start]
    ans_chars = list(text[answer_start:answer_end])
    post_text = text[answer_end:]
    
    n_insert = int(len(ans_chars) * ratio)
    n_insert = max(1, n_insert)
    
    for _ in range(n_insert):
        pos = random.randint(1, len(ans_chars)-1) if len(ans_chars) > 1 else 0
        char = random.choice(SPECIAL_CHARS)
        ans_chars.insert(pos, char)
        
    new_ans_text = "".join(ans_chars)
    new_context = pre_text + new_ans_text + post_text
    
    return new_context, answer_start, new_ans_text

# --- 통합 생성 함수 ---
def generate_all_datasets():
    print(f"Loading input: {INPUT_FILE}")
    try:
        source_data = load_json(INPUT_FILE)
    except FileNotFoundError:
        print("Input file not found.")
        return

    # 데이터 저장소 초기화
    datasets = {
        "baseline": {"version": "KorQuAD_v1.0_baseline", "data": []}
    }
    
    # 비율별 키 생성 (예: deletion_10, deletion_20, deletion_30)
    for r in RATIOS:
        percent = int(r * 100)
        datasets[f"deletion_{percent}"] = {"version": f"KorQuAD_v1.0_deletion_{percent}", "data": []}
        datasets[f"insertion_que_{percent}"] = {"version": f"KorQuAD_v1.0_insertion_que_{percent}", "data": []}
        datasets[f"insertion_ans_{percent}"] = {"version": f"KorQuAD_v1.0_insertion_ans_{percent}", "data": []}

    print("Processing all noise types and ratios...")
    
    for doc in tqdm(source_data['data']):
        # 현재 문서에 대해 각 타입별 구조 생성
        doc_structs = {
            k: {"title": doc['title'], "paragraphs": []} for k in datasets.keys()
        }
        
        for para in doc['paragraphs']:
            original_context = para['context']
            
            for qa in para['qas']:
                ans = qa['answers'][0]
                a_text = ans['text']
                a_start = ans['answer_start']
                
                # 1. Baseline (오염 없음)
                doc_structs["baseline"]["paragraphs"].append({
                    "context": original_context,
                    "qas": [qa]
                })

                # 비율별 오염 데이터 생성
                for r in RATIOS:
                    percent = int(r * 100)
                    
                    # 2. 삭제
                    del_ctx, del_start, del_ans = get_deletion_noise(original_context, a_text, a_start, ratio=r)
                    doc_structs[f"deletion_{percent}"]["paragraphs"].append({
                        "context": del_ctx,
                        "qas": [{
                            "id": qa['id'], "question": qa['question'],
                            "answers": [{"text": del_ans, "answer_start": del_start}]
                        }]
                    })
                    
                    # 3. 지문 삽입
                    ins_q_ctx, ins_q_start, ins_q_ans = get_context_insertion_noise(original_context, a_text, a_start, ratio=r)
                    doc_structs[f"insertion_que_{percent}"]["paragraphs"].append({
                        "context": ins_q_ctx,
                        "qas": [{
                            "id": qa['id'], "question": qa['question'],
                            "answers": [{"text": ins_q_ans, "answer_start": ins_q_start}]
                        }]
                    })
                    
                    # 4. 정답 삽입
                    ins_a_ctx, ins_a_start, ins_a_ans = get_answer_insertion_noise(original_context, a_text, a_start, ratio=r)
                    doc_structs[f"insertion_ans_{percent}"]["paragraphs"].append({
                        "context": ins_a_ctx,
                        "qas": [{
                            "id": qa['id'], "question": qa['question'],
                            "answers": [{"text": ins_a_ans, "answer_start": ins_a_start}]
                        }]
                    })
        
        # 문서가 비어있지 않다면 각 데이터셋에 추가
        for k in datasets.keys():
            if doc_structs[k]["paragraphs"]:
                datasets[k]["data"].append(doc_structs[k])

    print("Saving files...")
    for key, data_content in datasets.items():
        # 파일명 생성: train_insertion_ans_10.json 형식
        filename = f"train_{key}.json"
        save_path = os.path.join(OUTPUT_DIR, filename)
        save_json(data_content, save_path)
        print(f"   - Saved: {save_path}")

# %%
# 실행
generate_all_datasets()

# %% [markdown]
# ### 검증

# %%
# 검증 대상 경로
TARGET_DIR = "./datasets"
BASELINE_FILE = os.path.join(TARGET_DIR, "train_baseline.json")
SPECIAL_CHARS = ['!', '@', '#', '$', '%', '^', '&', '*']

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Load Failed: {filepath} ({e})")
        return None

def build_reference_map(baseline_data):
    ref_map = {}
    for doc in baseline_data['data']:
        for para in doc['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                qid = qa['id']
                ans = qa['answers'][0]
                ref_map[qid] = {
                    "context": context,
                    "text": ans['text'],
                    "start": ans['answer_start']
                }
    return ref_map

def verify_dataset(filepath, mode_tag, ref_map):
    filename = os.path.basename(filepath)
    print(f"\nVerifying: {filename} (Tag: {mode_tag})")
    
    data = load_json(filepath)
    if not data: return

    stats = {
        "total": 0,
        "alignment_ok": 0,
        "alignment_fail": 0,
        "corruption_ok": 0,
        "corruption_fail": 0,
        "skipped_short": 0,
        "errors": []
    }

    for doc in tqdm(data['data'], desc=f"Checking {mode_tag}"):
        for para in doc['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                stats["total"] += 1
                qid = qa['id']
                
                curr_ans = qa['answers'][0]
                curr_text = curr_ans['text']
                curr_start = curr_ans['answer_start']

                # 1. 정렬(alignment) 검사: answer_start 가 정답을 정확히 가리키는지
                extracted = context[curr_start : curr_start + len(curr_text)]
                if extracted == curr_text:
                    stats["alignment_ok"] += 1
                else:
                    stats["alignment_fail"] += 1
                    if len(stats["errors"]) < 3:
                        stats["errors"].append(f"[Align Error] ID: {qid}")

                # 2. 오염(corruption) 검사: 실제로 오염이 적용됐는지
                if 'baseline' in mode_tag:
                    stats["corruption_ok"] += 1
                    continue

                if qid not in ref_map: continue
                ref = ref_map[qid]
                is_corrupted = False
                
                # 파일명에 포함된 태그에 따라 검증 로직 분기 ('==' 대신 'in' 사용)
                if 'deletion' in mode_tag:
                    if len(context) < len(ref['context']):
                        is_corrupted = True
                    else:
                        # 길이가 안 줄어든 경우: 원본 단어 수가 적어서 스킵된 케이스인지 확인
                        ref_ctx = ref['context']
                        ref_start = ref['start']
                        ref_end = ref_start + len(ref['text'])
                        
                        pre_text = ref_ctx[:ref_start]
                        post_text = ref_ctx[ref_end:]
                        
                        pre_tokens = [t for t in pre_text.split(' ') if t.strip()]
                        post_tokens = [t for t in post_text.split(' ') if t.strip()]
                        
                        if len(pre_tokens) < 7 and len(post_tokens) < 7:
                            is_corrupted = True 
                            stats["skipped_short"] += 1
                        else:
                            is_corrupted = False
                    
                elif 'insertion_que' in mode_tag:
                    has_special = any(c in context for c in SPECIAL_CHARS)
                    is_longer = len(context) > len(ref['context'])
                    if has_special and is_longer:
                        is_corrupted = True
                        
                elif 'insertion_ans' in mode_tag:
                    has_special = any(c in curr_text for c in SPECIAL_CHARS)
                    is_longer = len(curr_text) > len(ref['text'])
                    if has_special and is_longer:
                        is_corrupted = True
                
                if is_corrupted:
                    stats["corruption_ok"] += 1
                else:
                    stats["corruption_fail"] += 1

    print(f"   Results for {filename}")
    print(f"   - Total Samples: {stats['total']}")
    if stats['total'] > 0:
        print(f"   - Alignment OK: {stats['alignment_ok']} ({(stats['alignment_ok']/stats['total'])*100:.1f}%)")
        
        if 'baseline' not in mode_tag:
            print(f"   - Corruption Verified: {stats['corruption_ok']} ({(stats['corruption_ok']/stats['total'])*100:.1f}%)")
            if 'deletion' in mode_tag and stats['skipped_short'] > 0:
                 print(f"     (Includes {stats['skipped_short']} samples confirmed skipped due to < 7 words)")

            if stats['corruption_fail'] > 0:
                print(f"   - Unchanged Samples (Fail): {stats['corruption_fail']}")
    print("-" * 50)

# 메인 실행 로직
if not os.path.exists(BASELINE_FILE):
    print(f"Baseline file not found at {BASELINE_FILE}")
else:
    print("Loading Baseline Data...")
    baseline_data = load_json(BASELINE_FILE)
    ref_map = build_reference_map(baseline_data)
    
    files = glob.glob(os.path.join(TARGET_DIR, "train_*.json"))
    files.sort()

    for file_path in files:
        # 파일명에서 확장자와 train_ 접두사 제거 (예: deletion_10)
        mode_tag = os.path.basename(file_path).replace("train_", "").replace(".json", "")
        verify_dataset(file_path, mode_tag, ref_map)
