import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 

# -------------------------------------------------
# (A) 헬퍼 함수: PCA 시각화를 위한 정규화
# -------------------------------------------------
def min_max_scale(img):
    """PCA 결과를 [0, 1] 범위로 정규화하는 함수"""
    img_min = img.min(axis=(0, 1))
    img_max = img.max(axis=(0, 1))
    return (img - img_min) / (img_max - img_min + 1e-6)

# -------------------------------------------------
# (A-2) ⬅️ (추가) 경로 설정
# -------------------------------------------------
task = 'ink'
# ❗️ 1. 이미지가 모여있는 폴더 경로
input_dir = f"imgs/{task}"
# ❗️ 2. Annotation 마스크를 저장할 폴더 경로
save_dir = f"C:/workspace/dinov3/output/black_weight_mask/{task}"  
# ❗️ 3. 처리할 이미지 확장자
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

# -------------------------------------------------
# 1. 모델 생성 + weight 로드 (순수 Backbone)
# -------------------------------------------------
REPO_DIR = 'C:/workspace/dinov3'
print("Loading model (Pretrain Backbone only)...")
model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights='weights/dinov3_vitb16_pretrain.pth')

model = model.half().to("cuda").eval()
print("✅ Model loaded to cuda with .half()")

# -------------------------------------------------
# 2. 전처리용 Transform 정의
# -------------------------------------------------
img_size = 2048
crop_size = 2048 

# (1) DINOv3 입력을 위한 정규화 트랜스폼
transform_dino = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(crop_size), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# (2) 색상 피처 및 시각화용 트랜스폼
def get_color_transform(H, W):
    return T.Compose([
        T.Grayscale(), # 1. 흑백으로 변환
        T.Resize((H, W), interpolation=T.InterpolationMode.LANCZOS), # 2. (H, W)로 평균 리사이즈
        T.ToTensor() # 3. [0, 1] 범위의 텐서 (0=검정, 1=흰색)
    ])

# -------------------------------------------------
# ⬅️ (추가) 메인 루프 시작
# -------------------------------------------------
os.makedirs(save_dir, exist_ok=True)
print(f"Processing images from: {input_dir}")
print(f"Saving results to: {save_dir}")

# "검은색" 특징을 얼마나 중요하게 볼지 결정하는 가중치
color_weight = 20.0
# K-Means 클러스터 개수
k = 10
# K-Means 반복 실행 횟수
n_init = 30 # ⬅️ (수정) 10 -> 20 (기존 코드 기준)

for filename in os.listdir(input_dir):
    # 1. 확장자 검사
    if not filename.lower().endswith(valid_extensions):
        continue

    # 2. 파일명 및 경로 설정
    img_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0] 
    
    print(f"\n--- Processing: {filename} ---")

    try:
        # -------------------------------------------------
        # 2. 이미지 로드 + 전처리 (루프 내 실행)
        # -------------------------------------------------
        img = Image.open(img_path).convert("RGB")

        # (1) DINOv3 입력용 (정규화됨)
        x = transform_dino(img).unsqueeze(0).half().to("cuda")

        # (2) 시각화 및 색상 피처 추출용 (원본 크롭)
        img_cropped = img.resize((img_size, img_size)).crop((
            (img_size - crop_size) // 2, 
            (img_size - crop_size) // 2, 
            (img_size + crop_size) // 2, 
            (img_size + crop_size) // 2
        ))

        # -------------------------------------------------
        # 3. Feature 추출 (DINOv3)
        # -------------------------------------------------
        with torch.no_grad():
            feats = model.forward_features(x)

        feat_tokens = feats["x_norm_patchtokens"] 
        feat_tokens = feat_tokens[:,:,:256]
        features_flat = feat_tokens.squeeze(0).cpu().numpy() 

        B, N, C = feat_tokens.shape
        H = W = int(N ** 0.5) 
        
        if N == 0:
            print(f"Skipping {filename}: No features extracted.")
            continue
        print(f"Feature map H, W: {H}x{W}") 

        # -------------------------------------------------
        # 3.1 "검은색" 피처 추출 (하이라이트)
        # -------------------------------------------------
        transform_color = get_color_transform(H, W)
        color_tensor = transform_color(img_cropped) 
        color_features_flat = color_tensor.permute(1, 2, 0).reshape(N, 1).cpu().numpy()

        # -------------------------------------------------
        # 3.2 DINO + Color 피처 결합
        # -------------------------------------------------
        scaled_color_features = color_features_flat * color_weight
        combined_features = np.concatenate([features_flat, scaled_color_features], axis=1)
        # combined_features = features_flat

        # -------------------------------------------------
        # 4. PCA 피처맵 생성 (시각화용)
        # -------------------------------------------------
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features_flat) 
        pca_img = pca_features.reshape(H, W, 3) 
        pca_img = min_max_scale(pca_img) 

        # -------------------------------------------------
        # 5. K-Means 클러스터링
        # -------------------------------------------------
        print(f"Running K-Means (k={k}) on combined features...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init) 
        labels = kmeans.fit_predict(combined_features) 
        kmeans_img = labels.reshape(H, W) 

        # -------------------------------------------------
        # 5.1 Crack 레이블 자동 탐색
        # -------------------------------------------------
        cluster_avg_colors = []
        for i in range(k):
            patches_in_cluster = (labels == i)
            if np.sum(patches_in_cluster) > 0:
                avg_color = np.mean(color_features_flat[patches_in_cluster])
            else:
                avg_color = 1.0 
            cluster_avg_colors.append(avg_color)

        crack_label = np.argmin(cluster_avg_colors)
        print(f"✅ Crack automatically identified as Label: {crack_label} (Avg. Color: {cluster_avg_colors[crack_label]:.4f})")

        # -------------------------------------------------
        # 5.2 매끈한 마스크 후처리
        # -------------------------------------------------
        binary_mask_lowres = (kmeans_img == crack_label).astype(np.uint8) * 255
        binary_mask_pil = Image.fromarray(binary_mask_lowres)
        smooth_mask_pil = binary_mask_pil.resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)
        smooth_mask_np = np.array(smooth_mask_pil)
        threshold = 127 
        final_mask_smooth = (smooth_mask_np > threshold).astype(np.uint8)

        # -------------------------------------------------
        # 6. 시각화 및 저장 (Crack 오버레이)
        # -------------------------------------------------
        save_image_path = os.path.join(save_dir, f"{base_name}_DINO_Color_AutoLabel_Overlay.png") 

        plt.figure(figsize=(15, 6)) 

        # --- 1번: 원본 이미지 ---
        plt.subplot(1, 3, 1) 
        plt.imshow(img_cropped) 
        plt.axis("off")
        plt.title(f"Input image, {crop_size}x{crop_size}") # ⬅️ 파일명 표시

        # --- 2번: PCA 피처맵 (DINO-only) ---
        plt.subplot(1, 3, 2)
        plt.imshow(pca_img) 
        plt.axis("off")
        plt.title(f"DINOv3 PCA Feature Map, {H}x{W})") 

        # --- 3번: Crack 오버레이 이미지 ---
        plt.subplot(1, 3, 3) 
        overlay_image = np.array(img_cropped).copy()
        crack_color = [255, 0, 0] # 빨간색 (RGB)
        alpha = 0.5               # 투명도
        mask_3channel = final_mask_smooth[:, :, np.newaxis] 
        overlay_image = (overlay_image * (1 - alpha) + mask_3channel * np.array(crack_color) * alpha).astype(np.uint8)

        plt.imshow(overlay_image)
        plt.axis("off")
        plt.title(f"Mask Overlay, {crop_size}x{crop_size}")

        plt.savefig(save_image_path, bbox_inches='tight', dpi=300)
        plt.close() # ⬅️ (중요) 메모리 누수 방지를 위해 figure를 닫습니다.

        print(f"✅ 최종 Crack 오버레이 이미지가 저장되었습니다: {save_image_path}")
        
    except Exception as e:
        # ⬅️ (추가) 에러 처리
        print(f"❗️ FAILED to process {filename}: {e}")

print("\n--- All processing complete. ---")