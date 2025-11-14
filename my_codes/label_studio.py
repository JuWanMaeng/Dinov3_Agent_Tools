import os
import json
import shutil
from tqdm import tqdm

# -----------------------------------------------------------------
# ❗️ 1. 여기만 수정하세요
# -----------------------------------------------------------------

# Label Studio에서 내보낸 JSON 파일의 전체 경로
json_path = r"C:\workspace\dinov3\project-3-at-2025-11-13-15-15-6147107e.json"

# 이름이 뒤죽박죽인 마스크 PNG가 모여있는 폴더
mask_input_dir = r"C:\workspace\dinov3\project-3-at-2025-11-13-15-06-6147107e"

# -----------------------------------------------------------------
# ❗️ 2. 최종 저장할 경로 (이전의 annotation 폴더)
# -----------------------------------------------------------------
# 수정된 파일이 저장될 최종 목적지
mask_output_dir = r"C:/data/DinoCrack/annotations/training"
# -----------------------------------------------------------------


# 1. 출력 폴더 생성
os.makedirs(mask_output_dir, exist_ok=True)
print(f"JSON 파일을 읽는 중: {json_path}")

# 2. JSON 파일을 읽어 {Task ID : 원본 파일명} 매핑 생성
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"JSON 파일 로드 실패: {e}")
    print("파일 경로와 인코딩(UTF-8)을 확인하세요.")
    exit()

task_to_original_name_map = {}
for task in data:
    task_id = task['id']
    
    # 'data' 필드에서 'image' 키를 찾습니다.
    if 'image' not in task['data']:
        print(f"경고: Task {task_id}에 'image' 키가 없습니다. 건너뜁니다.")
        continue
        
    original_path = task['data']['image']
    # 원본 파일명만 추출 (예: /data/upload/1/[000001].png -> [000001].png)
    original_filename = os.path.basename(original_path)
    # 확장자를 뺀 이름 (예: [000001])
    original_stem = os.path.splitext(original_filename)[0]
    
    task_to_original_name_map[task_id] = original_stem

print(f"총 {len(task_to_original_name_map)}개의 Task 매핑을 생성했습니다.")

# 3. 마스크 폴더를 순회하며 파일명 변경/복사
print(f"마스크 파일 이름 변경 시작: {mask_input_dir}")

renamed_count = 0
for mask_file in tqdm(os.listdir(mask_input_dir), desc="파일 이름 변경 중"):
    if not mask_file.endswith('.png'):
        continue

    # Label Studio 파일명 (예: task-1-annotation-1-by-1-0.png)
    try:
        # 파일명에서 task ID 추출
        task_id_str = mask_file.split('-')[1]
        task_id = int(task_id_str)
    except Exception:
        # 예상과 다른 형식의 파일명 (예: [000001].png)
        # 이미 올바른 이름일 수 있으므로, 그냥 복사
        print(f"'{mask_file}' 파일은 이미 변환된 파일일 수 있습니다. 그대로 복사합니다.")
        shutil.copy(os.path.join(mask_input_dir, mask_file), 
                    os.path.join(mask_output_dir, mask_file))
        renamed_count += 1
        continue

    # 4. 매핑에서 새 이름 찾기
    if task_id in task_to_original_name_map:
        new_stem = task_to_original_name_map[task_id]
        new_filename = f"{new_stem}.png"
        
        old_path = os.path.join(mask_input_dir, mask_file)
        new_path = os.path.join(mask_output_dir, new_filename)
        
        # 파일 복사 (덮어쓰기)
        shutil.copy(old_path, new_path)
        renamed_count += 1
    else:
        tqdm.write(f"경고: {mask_file} (Task {task_id})에 해당하는 원본 파일명을 JSON에서 찾지 못했습니다.")

print(f"\n--- 작업 완료 ---")
print(f"총 {renamed_count}개의 마스크 파일을 {mask_output_dir}에 복사/이름 변경했습니다.")