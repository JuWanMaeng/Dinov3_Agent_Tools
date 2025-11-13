import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
import os

# -------------------------------------------------
# (A) 헬퍼 함수: PCA 시각화를 위한 정규화
# -------------------------------------------------
def min_max_scale(img):
    """PCA 결과를 [0, 1] 범위로 정규화하는 함수"""
    img_min = img.min(axis=(0, 1))
    img_max = img.max(axis=(0, 1))
    return (img - img_min) / (img_max - img_min + 1e-6)

# -------------------------------------------------
# 1. 경로 설정
# -------------------------------------------------
# ❗️ 1. 시각화할 .npy 피처 파일의 경로
#    (이전에 'output/features_npy'에 저장한 파일 중 하나)
npy_path = r"C:\workspace\dinov3\output\features_npy\[000002]R10xC8-1-43pxl.npy" 

# ❗️ 2. PCA 시각화 이미지를 저장할 폴더
save_dir = "output/pca_visualizations_from_npy"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------------------------
# 2. NPY 로드
# -------------------------------------------------
print(f"Loading features from: {npy_path}")
try:
    # NumPy 파일을 로드합니다.
    features_flat = np.load(npy_path) # (N, C), 예: (1024, 768)
except Exception as e:
    print(f"❗️ FAILED to load {npy_path}: {e}")
    exit()

print(f"Features loaded successfully. Shape: {features_flat.shape}")

# -------------------------------------------------
# 3. H, W 계산
# -------------------------------------------------
N, C = features_flat.shape
H = W = int(N ** 0.5) # 1024 -> 32

if H * W != N:
    print(f"❗️ Error: Cannot reshape {N} patches into a square grid.")
    exit()

print(f"Feature map size (H, W): {H}x{W}")

# -------------------------------------------------
# 4. PCA 피처맵 생성
# -------------------------------------------------
print("Running PCA (for visualization)...")
pca = PCA(n_components=3)
pca_features = pca.fit_transform(features_flat) # (N, C) -> (N, 3)
pca_img = pca_features.reshape(H, W, 3) # (N, 3) -> (H, W, 3)
pca_img = min_max_scale(pca_img) # [0, 1] 범위로 정규화
print("PCA visualization complete.")

# -------------------------------------------------
# 5. 시각화 및 저장
# -------------------------------------------------
base_name = os.path.splitext(os.path.basename(npy_path))[0] 
save_image_path = os.path.join(save_dir, f"{base_name}_PCA_from_npy.png") 

plt.figure(figsize=(8, 8))
plt.imshow(pca_img)
plt.axis("off")
plt.title(f"PCA Feature Map (from {base_name}.npy)")

plt.savefig(save_image_path, bbox_inches='tight', dpi=300)
plt.close() 

print(f"✅ PCA 시각화 이미지가 저장되었습니다: {save_image_path}")