# %% [markdown]
# ### import

# %%
import json
import glob
import os
import shutil
import pandas as pd
from tqdm import tqdm

# %%
# 경로 설정
input_folder = './datasets/raw/1.Training'
save_folder = './datasets'
output_filename = 'merged_training.json'
output_file = os.path.join(save_folder, output_filename)

# %% [markdown]
# ### Training set 병합

# %%
# 파일 리스트 확보 및 병합
all_data = []
file_list = glob.glob(os.path.join(input_folder, '*.json'))

print(f"'{input_folder}' 경로에서 총 {len(file_list)}개의 JSON 파일을 찾았습니다.")

for file_path in tqdm(file_list, desc="Processing Training Files"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 리스트인 경우
            if isinstance(data, list):
                all_data.extend(data)
                
            # 내부에 리스트를 포함하는 경우
            elif isinstance(data, dict):
                found_list = False
                for key, value in data.items():
                    if isinstance(value, list):
                        all_data.extend(value)
                        found_list = True
                        break
                # 리스트가 없으면 딕셔너리 단일 항목으로 간주
                if not found_list:
                    all_data.append(data)
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# DataFrame 변환 및 결측치 검사
df_train = pd.DataFrame(all_data)

print("\n" + "="*50)
print(f"[Training] 데이터 병합 완료")
print(f"   - 총 데이터 건수: {len(df_train):,} 건")
print(f"   - 발견된 컬럼(Keys): {list(df_train.columns)}")
print("="*50)

# 결측치(Null) 확인
print("\n[결측치(Null) 보유 컬럼 및 개수]")
null_counts = df_train.isnull().sum()
if null_counts.sum() == 0:
    print("결측치가 발견되지 않았습니다 (All Clean).")
else:
    print(null_counts[null_counts > 0])

# 병합된 파일 저장
df_train.to_json(output_file, orient='records', force_ascii=False, indent=4)
print(f"\n병합된 파일이 '{output_file}' 로 저장되었습니다.")

# 데이터 샘플 확인
df_train.head(3)

# %% [markdown]
# ### Validation set 병합

# %%
# 데이터 경로 및 저장 파일명 설정
input_folder = './datasets/raw/2.Validation'
save_folder = './datasets'
output_filename = 'merged_validation.json'
output_file = os.path.join(save_folder, output_filename)

# %%
# 파일 리스트 확보 및 병합
all_data_val = []
file_list_val = glob.glob(os.path.join(input_folder, '*.json'))

print(f"'{input_folder}' 경로에서 총 {len(file_list_val)}개의 JSON 파일을 찾았습니다.")

for file_path in tqdm(file_list_val, desc="Processing Validation Files"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                all_data_val.extend(data)
            elif isinstance(data, dict):
                found_list = False
                for key, value in data.items():
                    if isinstance(value, list):
                        all_data_val.extend(value)
                        found_list = True
                        break
                if not found_list:
                    all_data_val.append(data)
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


# DataFrame 변환 및 결측치 검사
df_val = pd.DataFrame(all_data_val)

print("\n" + "="*50)
print(f"[Validation] 데이터 병합 완료")
print(f"   - 총 데이터 건수: {len(df_val):,} 건")
print(f"   - 발견된 컬럼(Keys): {list(df_val.columns)}")
print("="*50)

# 결측치(Null) 확인
print("\n[결측치(Null) 보유 컬럼 및 개수]")
null_counts_val = df_val.isnull().sum()
if null_counts_val.sum() == 0:
    print("결측치가 발견되지 않았습니다 (All Clean).")
else:
    print(null_counts_val[null_counts_val > 0])


# 병합된 파일 저장
df_val.to_json(output_file, orient='records', force_ascii=False, indent=4)
print(f"\n병합된 파일이 '{output_file}' 로 저장되었습니다.")

# 데이터 샘플 확인
df_val.head(3)

# %% [markdown]
# ### 생성된 파일 복사

# %%
# 출발지 폴더
source_folder = './datasets'

# 목적지 폴더
destination_folder = './datasets/interim'

# 복사할 파일명
files_to_copy = ['merged_training.json', 'merged_validation.json']

# %%
# 1. 목적지 폴더 생성
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"Created directory: {destination_folder}")

# 2. 파일 이동 실행
print(f"파일 복사 시작: {source_folder} -> {destination_folder}")

for file_name in files_to_copy:
    src_path = os.path.join(source_folder, file_name)
    dst_path = os.path.join(destination_folder, file_name)
    
    # 소스 파일 존재 확인
    if os.path.exists(src_path):
        try:
            # 파일 이동
            shutil.copy(src_path, dst_path)
            print(f"복사 완료: {file_name}")
        except Exception as e:
            print(f"복사 실패 ({file_name}): {e}")
    else:
        print(f"파일 없음: {src_path} (이전 단계가 정상적으로 완료되었는지 확인하세요)")

print("\n모든 작업이 완료되었습니다.")

# %%
def convert_custom_to_korquad(input_file, output_file):
    print(f"처리 시작: {input_file}")
    
    # 통계 변수 초기화
    stats = {
        "status": "processing", # processing, completed, error
        "error_msg": "",
        "total_docs": 0,
        "total_qas": 0,
        "valid_qas": 0,
        "skipped_no_answer_index": 0, # 본문에서 정답을 못 찾은 경우
        "skipped_empty_answer": 0     # 정답 텍스트 자체가 비어있는 경우
    }
    
    # 1. 입력 파일 존재 여부 확인
    if not os.path.exists(input_file):
        stats["status"] = "error"
        stats["error_msg"] = f"파일을 찾을 수 없음 ({input_file})"
        print(f"[오류] {stats['error_msg']}")
        return stats

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except Exception as e:
        stats["status"] = "error"
        stats["error_msg"] = f"JSON 로드 실패 ({str(e)})"
        print(f"[오류] {stats['error_msg']}")
        return stats

    stats["total_docs"] = len(source_data)
    
    korquad_data = {
        "version": "KorQuAD_v1.0_transformed",
        "data": []
    }

    # 2. 데이터 변환 루프
    for doc in source_data:
        k_doc = {
            "title": doc.get('doc_title', 'No Title'),
            "paragraphs": []
        }
        
        for para in doc['paragraphs']:
            context = para['context']
            qas_list = []
            
            for qa in para['qas']:
                stats["total_qas"] += 1
                
                # 정답 텍스트 추출
                ans_obj = qa['answers']
                answer_text = ans_obj.get('text', '').strip()
                
                # Case A: 정답 텍스트가 비어있는 경우
                if not answer_text:
                    stats["skipped_empty_answer"] += 1
                    continue

                # Case B: 정답 인덱스 검색 (본문 내 위치 찾기)
                start_idx = context.find(answer_text)
                
                if start_idx != -1:
                    # 성공: 유효한 데이터로 추가
                    new_qa = {
                        "id": qa['question_id'],
                        "question": qa['question'],
                        "answers": [
                            {
                                "text": answer_text,
                                "answer_start": start_idx
                            }
                        ]
                    }
                    qas_list.append(new_qa)
                    stats["valid_qas"] += 1
                else:
                    # 실패: 본문에 정답 텍스트가 없는 경우
                    stats["skipped_no_answer_index"] += 1
            
            # 유효한 질문이 하나라도 있는 문단만 저장
            if qas_list:
                k_doc['paragraphs'].append({
                    "context": context,
                    "qas": qas_list
                })
        
        # 유효한 문단이 하나라도 있는 문서만 저장
        if k_doc['paragraphs']:
            korquad_data['data'].append(k_doc)

    # 3. 결과 저장 (폴더 자동 생성 포함)
    try:
        # 출력 경로의 디렉토리(폴더) 부분만 추출하여 생성
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"폴더 생성됨: {output_dir}")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(korquad_data, f, ensure_ascii=False, indent=2)
            
        stats["status"] = "completed"
        print(f"변환 완료 및 저장: {output_file}")
        
    except Exception as e:
        stats["status"] = "error"
        stats["error_msg"] = f"파일 저장 실패 ({str(e)})"
        print(f"[오류] {stats['error_msg']}")

    return stats

def print_summary(file_label, stats):
    print(f"\n[{file_label}] 처리 결과 요약")
    print(f"   - 상태: {stats['status']}")
    if stats['status'] == 'error':
        print(f"   - 원인: {stats['error_msg']}")
    else:
        print(f"   - 총 문서 수: {stats['total_docs']}개")
        print(f"   - 총 질문 수: {stats['total_qas']}개")
        print(f"   - 변환 성공: {stats['valid_qas']}개")
        print(f"   - 제외됨 (정답 인덱스 못 찾음): {stats['skipped_no_answer_index']}개")
        print(f"   - 제외됨 (정답 텍스트 비어있음): {stats['skipped_empty_answer']}개")
    print("-" * 40)

# %%
# 입력 경로 (병합된 중간 산출물)
input_train = './datasets/merged_training.json'
input_val = './datasets/merged_validation.json'

# 출력 경로 (최종 KorQuAD 학습/검증셋)
output_train = './datasets/bert_train.json'
output_val = './datasets/bert_validation.json'

# 실행 및 통계 출력
stats_train = convert_custom_to_korquad(input_train, output_train)
print_summary("Training Set", stats_train)

stats_val = convert_custom_to_korquad(input_val, output_val)
print_summary("Validation Set", stats_val)
