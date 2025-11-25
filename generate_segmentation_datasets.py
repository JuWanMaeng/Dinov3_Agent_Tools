import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans
import cv2
import json
import random

# -------------------------------------------------
# (A) 설정 및 경로 초기화
# -------------------------------------------------
task = 'crack'
input_dir = f"imgs/{task}"
base_save_dir = f"C:/workspace/dinov3/output/black_weight_mask/{task}_split"

# 1. Train/Val 비율 설정
val_ratio = 0.2
seed = 42
random.seed(seed)

# 2. 폴더 생성 (Train / Val 구조)
train_dir = os.path.join(base_save_dir, "train")
val_dir = os.path.join(base_save_dir, "val")
train_img_dir = os.path.join(train_dir, "images")
val_img_dir = os.path.join(val_dir, "images")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

# 3. COCO JSON 초기화 함수
def init_coco(desc):
    return {
        "info": {
            "description": desc,
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "DINOv3 Agent",
            "date_created": "2024-11-25"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "defect", "supercategory": "defect"}]
    }

coco_train = init_coco("DINOv3 Auto-labeled Train Set")
coco_val = init_coco("DINOv3 Auto-labeled Val Set")

# ID 카운터 (전역적으로 고유하게 증가)
global_ann_id = 1
global_img_id = 1

# 4. 모델 로드
REPO_DIR = 'C:/workspace/dinov3'
print("Loading model (Pretrain Backbone only)...")
model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights='weights/dinov3_vitb16_pretrain.pth')
model = model.half().to("cuda").eval()
print("✅ Model loaded.")

# -------------------------------------------------
# (B) 전처리 Transform 정의
# -------------------------------------------------
inference_size = 2048   # 추론 해상도
crop_size = 2048        # 추론 크롭 사이즈
target_size = 640       # 저장 해상도

transform_dino = T.Compose([
    T.Resize(inference_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(crop_size), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_color_transform(H, W):
    return T.Compose([
        T.Grayscale(),
        T.Resize((H, W), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor()
    ])

# -------------------------------------------------
# (C) 메인 루프 실행
# -------------------------------------------------
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
random.shuffle(file_list) # 파일 리스트 자체를 섞어서 순서 랜덤화

print(f"Processing {len(file_list)} images...")
print(f"Saving to: {base_save_dir} (Train/Val Split)")

color_weight = 20.0
k = 10
n_init = 10 
scale_factor = target_size / crop_size

for filename in file_list:
    img_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    
    # 🎲 Train vs Val 결정
    is_val = random.random() < val_ratio
    current_coco = coco_val if is_val else coco_train
    current_save_dir = val_img_dir if is_val else train_img_dir
    split_name = "VAL" if is_val else "TRAIN"

    print(f"[{split_name}] Processing: {filename} ...", end=" ")

    try:
        # 1. 이미지 로드 및 전처리
        img = Image.open(img_path).convert("RGB")
        x = transform_dino(img).unsqueeze(0).half().to("cuda")
        
        # 추론용 고해상도 크롭 (2048)
        img_cropped = img.resize((inference_size, inference_size)).crop((
            (inference_size - crop_size) // 2, 
            (inference_size - crop_size) // 2, 
            (inference_size + crop_size) // 2, 
            (inference_size + crop_size) // 2
        ))

        # 2. Feature 추출
        with torch.no_grad():
            feats = model.forward_features(x)

        feat_tokens = feats["x_norm_patchtokens"] 
        feat_tokens = feat_tokens[:,:,:256]
        features_flat = feat_tokens.squeeze(0).cpu().numpy() 

        B, N, C = feat_tokens.shape
        H = W = int(N ** 0.5)
        
        if N == 0: continue

        # 색상 피처 결합
        transform_color = get_color_transform(H, W)
        color_tensor = transform_color(img_cropped) 
        color_features_flat = color_tensor.permute(1, 2, 0).reshape(N, 1).cpu().numpy()

        scaled_color_features = color_features_flat * color_weight
        combined_features = np.concatenate([features_flat, scaled_color_features], axis=1)

        # 3. K-Means 마스크 생성
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init) 
        labels = kmeans.fit_predict(combined_features) 
        
        # Crack(검은색) 레이블 자동 탐색
        cluster_avg_colors = []
        for i in range(k):
            patches_in_cluster = (labels == i)
            avg_color = np.mean(color_features_flat[patches_in_cluster]) if np.sum(patches_in_cluster) > 0 else 1.0
            cluster_avg_colors.append(avg_color)
        
        crack_label = np.argmin(cluster_avg_colors)
        kmeans_img = labels.reshape(H, W)
        
        # 마스크 업스케일링 (저해상도 -> 2048)
        binary_mask_lowres = (kmeans_img == crack_label).astype(np.uint8) * 255
        binary_mask_pil = Image.fromarray(binary_mask_lowres)
        smooth_mask_pil = binary_mask_pil.resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)
        mask_2048 = (np.array(smooth_mask_pil) > 127).astype(np.uint8)

        # 4. 저장 (640 리사이즈) 및 COCO 등록
        
        # (1) 이미지 저장
        img_resized = img_cropped.resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
        image_filename = f"{base_name}.jpg"
        img_resized.save(os.path.join(current_save_dir, image_filename))
        
        # (2) 이미지 정보 등록
        current_coco["images"].append({
            "id": global_img_id,
            "width": target_size,
            "height": target_size,
            "file_name": image_filename,
            "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0
        })

        # (3) 어노테이션 등록
        mask_for_cv2 = mask_2048 * 255
        contours, _ = cv2.findContours(mask_for_cv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ann_count = 0
        for contour in contours:
            if cv2.contourArea(contour) < 20: continue # 노이즈 제거
            
            # 좌표 스케일링 (2048 -> 640)
            contour_scaled = (contour * scale_factor).astype(np.int32)
            area = cv2.contourArea(contour_scaled)
            if area < 1: continue

            x, y, w, h = cv2.boundingRect(contour_scaled)
            
            current_coco["annotations"].append({
                "id": global_ann_id,
                "image_id": global_img_id,
                "category_id": 1,
                "segmentation": [contour_scaled.flatten().tolist()],
                "area": area,
                "bbox": [int(x), int(y), int(w), int(h)],
                "iscrowd": 0
            })
            global_ann_id += 1
            ann_count += 1
            
        global_img_id += 1
        print(f"Done. (Anns: {ann_count})")

    except Exception as e:
        print(f"\n❗️ Error: {e}")

print("\n--- Saving JSON Files ---")

# Train JSON 저장
train_json_path = os.path.join(train_dir, "annotations.json") # 보통 train 폴더 안에 두거나 상위 폴더에 train.json으로 저장
# 여기서는 train/annotations.json 구조로 저장
with open(train_json_path, "w") as f:
    json.dump(coco_train, f, indent=4)

# Val JSON 저장
val_json_path = os.path.join(val_dir, "annotations.json")
with open(val_json_path, "w") as f:
    json.dump(coco_val, f, indent=4)

print(f"✅ Train Saved: {train_json_path} ({len(coco_train['images'])} images)")
print(f"✅ Val Saved: {val_json_path} ({len(coco_val['images'])} images)")