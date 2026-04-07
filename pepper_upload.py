import os
import shutil

base_dir = 'C:/DL_DATA/peppers/val'
src_image_dir = os.path.join(base_dir, 'images')
src_label_dir = os.path.join(base_dir, 'labels')

dest_dir = 'C:/DL_DATA/mini_peppers_matched/val'
dest_image_dir = os.path.join(dest_dir, 'image')
dest_label_dir = os.path.join(dest_dir, 'labels')

os.makedirs(dest_image_dir, exist_ok=True)
os.makedirs(dest_label_dir, exist_ok=True)

print("하위 폴더를 포함하여 모든 이미지와 라벨을 스캔합니다.")

# 1. 모든 라벨 파일의 경로를 딕셔너리로 수집 (하위 폴더 포함)
label_dict = {}
for root, dirs, files in os.walk(src_label_dir):
    for f in files:
        if f.endswith('.json'):
            label_dict[f] = os.path.join(root, f)

print(f"탐색된 전체 라벨 파일 수: {len(label_dict)}개")

# 2. 이미지를 탐색하며 짝이 맞는 라벨 병렬 검색
target_count = 100
matched_count = 0

for root, dirs, files in os.walk(src_image_dir):
    for img_name in files:
        if not img_name.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            continue

        if matched_count >= target_count:
            break

        # 예상되는 라벨 파일명 두 가지 케이스
        label_name_1 = img_name + '.json'                  # 예: file.JPG.json
        label_name_2 = os.path.splitext(img_name)[0] + '.json' # 예: file.json

        # 딕셔너리에서 해당 파일명이 존재하는지 확인
        valid_label_path = label_dict.get(label_name_1) or label_dict.get(label_name_2)

        if valid_label_path:
            shutil.copy(os.path.join(root, img_name), os.path.join(dest_image_dir, img_name))

            # 라벨을 복사할 때는 원본과 동일한 이름으로 저장
            dest_label_name = label_name_1 if label_dict.get(label_name_1) else label_name_2
            shutil.copy(valid_label_path, os.path.join(dest_label_dir, dest_label_name))

            matched_count += 1

    if matched_count >= target_count:
        break

print(f"\n작업 완료: 총 {matched_count}쌍의 데이터가 '{dest_dir}'에 복사되었습니다.")

if matched_count == 0:
    print("경고: 압축 해제한 이미지 데이터와 라벨 데이터가 서로 다른 파트일 가능성이 높습니다. (예: 이미지는 파트1, 라벨은 파트2를 다운로드한 경우)")
