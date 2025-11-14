import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from tqdm import tqdm
# ⬅️ (F.min_pool2d는 사용하지 않으므로 F 삭제)

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
# ❗️ 1. 이미지가 모여있는 폴더 경로
input_dir = r"C:\workspace\dinov3\imgs\particles" 
# ❗️ 2. 결과 오버레이 그리드를 저장할 폴더 경로 (이름 변경)
save_dir = "output/particles_masks" 
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
patch_size = model.patch_embed.patch_size[0] # (16)

# -------------------------------------------------
# 2. 전처리용 Transform 정의
# -------------------------------------------------
img_size = 1024
crop_size = 1024 

# (1) DINOv3 입력을 위한 정규화 트랜스폼
transform_dino = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(crop_size), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# (2) ⬅️ (유지) Average Color 피처용 트랜스폼 (LANCZOS)
def get_color_transform(H, W):
    return T.Compose([
        T.Grayscale(), # 1. 흑백으로 변환
        T.Resize((H, W), interpolation=T.InterpolationMode.LANCZOS), # 2. (H, W)로 평균 리사이즈
        T.ToTensor() # 3. [0, 1] 범위의 텐서 (0=검정, 1=흰색)
    ])

# -------------------------------------------------
# ⬅️ (수정) 메인 루프 시작
# -------------------------------------------------
os.makedirs(save_dir, exist_ok=True)
print(f"Processing images from: {input_dir}") 
print(f"Saving results to: {save_dir}")       

# "검은색" 특징을 얼마나 중요하게 볼지 결정하는 가중치
color_weight = 10.0
# K-Means 클러스터 개수
k = 6
# K-Means 반복 실행 횟수
n_init = 10
# 오버레이 설정
crack_color = [255, 0, 0] # 빨간색 (RGB)
alpha = 0.5               # 투명도

for filename in tqdm(os.listdir(input_dir), desc="Processing images", ncols=100):
    # 1. 확장자 검사
    if not filename.lower().endswith(valid_extensions):
        continue

    # 2. 파일명 및 경로 설정
    img_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0] 
    
    tqdm.write(f"\n--- Processing: {filename} ---") 

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
        img_cropped_np = np.array(img_cropped) # ⬅️ (추가) 오버레이용 numpy 배열

        # -------------------------------------------------
        # 3. Feature 추출 (DINOv3)
        # -------------------------------------------------
        with torch.no_grad():
            feats = model.forward_features(x)

        feat_tokens = feats["x_norm_patchtokens"] 
        features_flat = feat_tokens.squeeze(0).cpu().numpy() 

        B, N, C = feat_tokens.shape
        H = W = int(N ** 0.5) 
        
        if N == 0:
            tqdm.write(f"Skipping {filename}: No features extracted.") 
            continue

        # -------------------------------------------------
        # 3.1 ⬅️ (유지) "검은색" 피처 추출 (Average - LANCZOS)
        # -------------------------------------------------
        transform_color = get_color_transform(H, W)
        color_tensor = transform_color(img_cropped) 
        color_features_flat = color_tensor.permute(1, 2, 0).reshape(N, 1).cpu().numpy()

        # -------------------------------------------------
        # 3.2 ⬅️ (유지) DINO + Color 피처 결합
        # -------------------------------------------------
        scaled_color_features = color_features_flat * color_weight
        combined_features = np.concatenate([features_flat, scaled_color_features], axis=1)

        # -------------------------------------------------
        # 4. PCA 피처맵 생성 (시각화용)
        # -------------------------------------------------
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features_flat) 
        pca_img = pca_features.reshape(H, W, 3) 
        pca_img = min_max_scale(pca_img) 

        # -------------------------------------------------
        # 5. K-Means 클러스터링 (k=10)
        # -------------------------------------------------
        tqdm.write(f"Running K-Means (k={k}) for {filename}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init) 
        labels = kmeans.fit_predict(combined_features) 
        kmeans_img = labels.reshape(H, W) # (128, 128)
        tqdm.write(f"✅ K-Means complete for {filename}")
        # -------------------------------------------------

        # -------------------------------------------------
        # 6. ⬅️ (핵심 수정) 10개 마스크 개별 시각화 및 저장
        # -------------------------------------------------
        save_image_path = os.path.join(save_dir, f"{base_name}_K10_All_Masks_Overlay.png") 

        # 3x4 그리드 (총 12칸)
        rows = 2
        cols = 4
        plt.figure(figsize=(cols * 5, rows * 5)) # 전체 Figure 크기 (20, 15)

        # --- 1번: 원본 이미지 ---
        plt.subplot(rows, cols, 1) 
        plt.imshow(img_cropped) 
        plt.axis("off")
        plt.title(f"Original: {filename}") 

        # --- 2번: PCA 피처맵 (DINO-only) ---
        plt.subplot(rows, cols, 2)
        plt.imshow(pca_img) # 128x128
        plt.axis("off")
        plt.title(f"PCA Feature Map ({H}x{W})") 

        # --- 3~12번: K-Means의 10개 레이블(0~9) 개별 오버레이 ---
        for i in range(k): # i = 0 부터 9 까지
            subplot_idx = i + 3 # 3번 칸부터 12번 칸까지
            
            # 1. 128x128 흑백 마스크 생성 (i번 레이블만 255)
            binary_mask_lowres = (kmeans_img == i).astype(np.uint8) * 255
            
            # 2. 매끈한 마스크 후처리
            binary_mask_pil = Image.fromarray(binary_mask_lowres)
            smooth_mask_pil = binary_mask_pil.resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)
            smooth_mask_np = np.array(smooth_mask_pil)
            threshold = 127 
            final_mask_smooth = (smooth_mask_np > threshold).astype(np.uint8)

            # 3. 오버레이 이미지 생성
            overlay_image = img_cropped_np.copy()
            mask_3channel = final_mask_smooth[:, :, np.newaxis] 
            overlay_image = (overlay_image * (1 - alpha) + mask_3channel * np.array(crack_color) * alpha).astype(np.uint8)

            # 4. 시각화
            plt.subplot(rows, cols, subplot_idx)
            plt.imshow(overlay_image) 
            plt.axis("off")
            plt.title(f"Overlay for Label: {i}")
            
        plt.tight_layout()
        plt.savefig(save_image_path, bbox_inches='tight', dpi=300)
        plt.close() # ⬅️ (중요) 메모리 누수 방지를 위해 figure를 닫습니다.

        
    except Exception as e:
        # ⬅️ (추가) 에러 처리
        tqdm.write(f"❗️ FAILED to process {filename}: {e}") # ⬅️ 수정

print("\n--- All processing complete. ---")