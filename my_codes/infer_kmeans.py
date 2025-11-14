import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # ⬅️ K-Means import 확인
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
# 1. 모델 생성 + weight 로드 (순수 Backbone)
# -------------------------------------------------
REPO_DIR = 'C:/workspace/dinov3'
print("Loading model (Pretrain Backbone only)...")
# (ViT-Base 로드 확인)
model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights='weights/dinov3_vitb16_pretrain.pth')
# model = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights='weights/dinov3_vit7b16_pretrain.pth')

model = model.half().to("cuda").eval()
print("✅ Model loaded to cuda with .half()")

# -------------------------------------------------
# 2. 이미지 로드 + 전처리
# -------------------------------------------------
img_size = 2048
crop_size = 2048 # ⬅️ 1024 리사이즈 후 512 크롭
img_path = r"C:\data\251102\crack\[000246]R15xC24-2503-89pxl.png"
img = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(crop_size), # ⬅️ (핵심 수정) RandomCrop -> CenterCrop
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

x = transform(img).unsqueeze(0).half().to("cuda")

# -------------------------------------------------
# 3. Feature 추출 (Backbone만)
# -------------------------------------------------
print("Extracting features...")
with torch.no_grad():
    feats = model.forward_features(x)

# (B, N, C) - 예: (1, 1024, 768) (512px / 16 = 32x32=1024)
feat_tokens = feats["x_norm_patchtokens"] 
print(f"Feature map shape: {feat_tokens.shape}")

# (B, N, C) -> (N, C)
features_flat = feat_tokens.squeeze(0).cpu().numpy()

# H, W 계산 (reshape용)
B, N, C = feat_tokens.shape
H = W = int(N ** 0.5) # 512/16 = 32
H = W = 512

# -------------------------------------------------
# 4. (추가) PCA 피처맵 생성
# -------------------------------------------------
print("Running PCA (for visualization)...")
pca = PCA(n_components=3)
pca_features = pca.fit_transform(features_flat) # (N, C) -> (N, 3)
pca_img = pca_features.reshape(H, W, 3) # (N, 3) -> (H, W, 3)
pca_img = min_max_scale(pca_img) # [0, 1] 범위로 정규화
print("PCA visualization complete.")

# -------------------------------------------------
# 5. (추가) K-Means 클러스터링 
# -------------------------------------------------
k = 10 # ⬅️ (추가) 클러스터 개수 (원하는 값으로 수정)
print(f"Running K-Means (k={k})...")

# n_init=10을 설정하여 경고를 방지하고 안정적인 결과를 얻습니다.
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
labels = kmeans.fit_predict(features_flat) # (N, C) -> (N,)

# (N,) -> (H, W)
kmeans_img = labels.reshape(H, W)
print("K-Means complete.")


# -------------------------------------------------
# 6. 시각화 및 저장 (3개 패널로 수정)
# -------------------------------------------------
save_dir = "output/visualizations_base"
os.makedirs(save_dir, exist_ok=True)
base_name = os.path.splitext(os.path.basename(img_path))[0] 
# (파일명 변경)
save_image_path = os.path.join(save_dir, f"{base_name}_pca_kmeans_comparison.png") # ⬅️ (수정) 파일명

plt.figure(figsize=(15, 6)) # ⬅️ (수정) 3개 패널용 너비 (10 -> 15)

# --- 1번: 원본 이미지 ---
plt.subplot(1, 3, 1) # ⬅️ (수정) 1행 3열 중 1번째
# K-Means/PCA 입력과 동일하게 크롭된 이미지 표시
img_cropped = img.resize((img_size, img_size)).crop((
    (img_size - crop_size) // 2, 
    (img_size - crop_size) // 2, 
    (img_size + crop_size) // 2, 
    (img_size + crop_size) // 2
))
plt.imshow(img_cropped)
plt.axis("off")
plt.title(f"Original Image (Cropped {crop_size}x{crop_size})")

# --- 2번: PCA 피처맵 (추가) ---
plt.subplot(1, 3, 2) # ⬅️ (수정) 1행 3열 중 2번째
plt.imshow(pca_img)
plt.axis("off")
plt.title("PCA Feature Map (C -> 3)")

# --- 3번: K-Means 클러스터링 결과 ---
plt.subplot(1, 3, 3) # ⬅️ (추가) 1행 3열 중 3번째
plt.imshow(kmeans_img, cmap='tab10') # ⬅️ (추가) 범주형 colormap 사용
plt.axis("off")
plt.title(f"K-Means (k={k})")


plt.savefig(save_image_path, bbox_inches='tight', dpi=300)
plt.close() 


print(f"PCA + K-Means 시각화 이미지가 저장되었습니다: {save_image_path}")