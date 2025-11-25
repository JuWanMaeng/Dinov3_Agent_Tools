import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans
# PCA, matplotlib 제거됨
import cv2
import json

# -------------------------------------------------
# (A) 경로 설정
# -------------------------------------------------
task = 'crack'
# 1. 입력 이미지 폴더
input_dir = f"imgs/{task}"

# 2. 결과 저장 폴더 (COCO 데이터셋 구조)
save_dir = f"C:/workspace/dinov3/output/black_weight_mask/{task}"  
images_save_dir = os.path.join(save_dir, "images") # 이미지가 저장될 곳
os.makedirs(images_save_dir, exist_ok=True)

# COCO Dataset 구조 초기화
coco_output = {
    "info": {
        "description": "DINOv3 Auto-labeled Dataset",
        "url": "",
        "version": "1.0",
        "year": 2024,
        "contributor": "DINOv3 Agent",
        "date_created": "2024-11-25"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "defect", "supercategory": "defect"}
    ]
}
annotation_id = 1
image_id = 1

# 3. 처리할 이미지 확장자
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

# -------------------------------------------------
# 1. 모델 생성 + weight 로드 (Backbone)
# -------------------------------------------------
REPO_DIR = 'C:/workspace/dinov3'
print("Loading model (Pretrain Backbone only)...")
# 로컬 캐시나 소스 경로가 맞는지 확인 필요
model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights='weights/dinov3_vitb16_pretrain.pth')
model = model.half().to("cuda").eval()
print("✅ Model loaded to cuda with .half()")

# -------------------------------------------------
# 2. 전처리용 Transform 정의
# -------------------------------------------------
img_size = 2048
crop_size = 2048 

# (1) DINOv3 입력용
transform_dino = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(crop_size), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# (2) 색상 피처 추출용
def get_color_transform(H, W):
    return T.Compose([
        T.Grayscale(),
        T.Resize((H, W), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor()
    ])

# -------------------------------------------------
# 메인 루프 시작
# -------------------------------------------------
print(f"Processing images from: {input_dir}")
print(f"Saving COCO dataset to: {save_dir}")

color_weight = 20.0
k = 10
n_init = 10 

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(valid_extensions):
        continue

    img_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    
    print(f"Processing: {filename} ...", end=" ")

    try:
        # -------------------------------------------------
        # 2. 이미지 로드 + 전처리
        # -------------------------------------------------
        img = Image.open(img_path).convert("RGB")

        # DINO 입력
        x = transform_dino(img).unsqueeze(0).half().to("cuda")

        # 원본 크롭 (저장용 및 색상 추출용)
        img_cropped = img.resize((img_size, img_size)).crop((
            (img_size - crop_size) // 2, 
            (img_size - crop_size) // 2, 
            (img_size + crop_size) // 2, 
            (img_size + crop_size) // 2
        ))

        # -------------------------------------------------
        # 3. Feature 추출 (DINOv3 + Color)
        # -------------------------------------------------
        with torch.no_grad():
            feats = model.forward_features(x)

        feat_tokens = feats["x_norm_patchtokens"] 
        feat_tokens = feat_tokens[:,:,:256]
        features_flat = feat_tokens.squeeze(0).cpu().numpy() 

        B, N, C = feat_tokens.shape
        H = W = int(N ** 0.5)
        
        if N == 0:
            print("Skipping: No features.")
            continue

        # 색상 피처 결합
        transform_color = get_color_transform(H, W)
        color_tensor = transform_color(img_cropped) 
        color_features_flat = color_tensor.permute(1, 2, 0).reshape(N, 1).cpu().numpy()

        scaled_color_features = color_features_flat * color_weight
        combined_features = np.concatenate([features_flat, scaled_color_features], axis=1)

        # -------------------------------------------------
        # 4. K-Means (마스크 생성)
        # -------------------------------------------------

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init) 
        labels = kmeans.fit_predict(combined_features) 
        kmeans_img = labels.reshape(H, W) 

        # Crack(검은색) 레이블 자동 탐색
        cluster_avg_colors = []
        for i in range(k):
            patches_in_cluster = (labels == i)
            if np.sum(patches_in_cluster) > 0:
                avg_color = np.mean(color_features_flat[patches_in_cluster])
            else:
                avg_color = 1.0 
            cluster_avg_colors.append(avg_color)

        crack_label = np.argmin(cluster_avg_colors)
        
        # 마스크 생성 및 후처리
        binary_mask_lowres = (kmeans_img == crack_label).astype(np.uint8) * 255
        binary_mask_pil = Image.fromarray(binary_mask_lowres)
        
        # 원본 크기로 리사이즈 (부드럽게)
        smooth_mask_pil = binary_mask_pil.resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)
        smooth_mask_np = np.array(smooth_mask_pil)
        threshold = 127 
        final_mask_smooth = (smooth_mask_np > threshold).astype(np.uint8) # 0 or 1 (if used directly) or 255 for cv2

        # -------------------------------------------------
        # 5. COCO 데이터셋 저장 (시각화 제거됨)
        # -------------------------------------------------
        
        # (1) 이미지 파일 저장
        image_filename = f"{base_name}.jpg"
        image_save_path = os.path.join(images_save_dir, image_filename)
        img_cropped.save(image_save_path)
        
        # (2) COCO Images 정보 등록
        coco_output["images"].append({
            "id": image_id,
            "width": crop_size,
            "height": crop_size,
            "file_name": image_filename,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })

        # (3) 마스크 -> 폴리곤 변환 (Annotation 등록)
        # cv2.findContours를 위해 255 스케일로 확실하게 변환
        mask_for_cv2 = final_mask_smooth * 255 
        contours, _ = cv2.findContours(mask_for_cv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotation_count = 0
        for contour in contours:
            if cv2.contourArea(contour) < 20: # 노이즈 제거 (면적 임계값 약간 상향)
                continue
            
            # Segmentation (Polygon)
            segmentation = contour.flatten().tolist()
            
            # Bounding Box
            x_box, y_box, w_box, h_box = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [segmentation],
                "area": area,
                "bbox": [x_box, y_box, w_box, h_box],
                "iscrowd": 0
            })
            annotation_id += 1
            annotation_count += 1
            
        image_id += 1
        print(f"Done. (Annotations: {annotation_count})")

    except Exception as e:
        print(f"\n❗️ Error processing {filename}: {e}")

print("\n--- All processing complete. ---")

# COCO JSON 파일 저장
coco_json_path = os.path.join(save_dir, "annotations.json")
with open(coco_json_path, "w") as f:
    json.dump(coco_output, f, indent=4)
    
print(f"✅ COCO annotations saved to: {coco_json_path}")