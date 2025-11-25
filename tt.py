import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 

# ❗️ (추가) skimage 및 scipy 임포트
from skimage.filters import sobel_h, sobel_v
from skimage.transform import resize
from skimage import color as ski_color
from scipy.ndimage import convolve

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
task = 'crack_hardcases'
# ❗️ 1. 이미지가 모여있는 폴더 경로
input_dir = f"C:/workspace/dinov3/imgs/{task}" 
# ❗️ 2. Annotation 마스크를 저장할 폴더 경로
save_dir = f"C:/workspace/dinov3/output/hetero_diag_mask/{task}" # ❗️ 저장 폴더명 변경
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

# ❗️ (삭제) get_color_transform 함수 (더 이상 사용하지 않음)

# -------------------------------------------------
# ⬅️ (추가) 메인 루프 시작
# -------------------------------------------------
os.makedirs(save_dir, exist_ok=True)
print(f"Processing images from: {input_dir}")
print(f"Saving results to: {save_dir}")

# ❗️ (수정) 가중치 설정 (Color 가중치 제거)
diagonal_weight = 10.0   # 대각선 가중치 (조정 가능)
horizontal_weight = 10.0 # 가로선 가중치 (조정 가능)

# K-Means 클러스터 개수
k = 10 
# K-Means 반복 실행 횟수
n_init = 10 

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

        # (2) 시각화 및 피처 추출용 (원본 크롭)
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
        feat_tokens = feat_tokens[:,:,:128]
        features_flat = feat_tokens.squeeze(0).cpu().numpy() # (N, 128)

        B, N, C = feat_tokens.shape
        H = W = int(N ** 0.5) 
        
        if N == 0:
            print(f"Skipping {filename}: No features extracted.")
            continue
        print(f"Feature map H, W: {H}x{W}") 

        # -------------------------------------------------
        # 3.1 (수정) 피처 추출 (1: 대각선, 2: 가로선)
        # -------------------------------------------------
        img_gray_np = ski_color.rgb2gray(np.array(img_cropped))

        # --- 1. "대각선" 피처 (직접 커널 방식) ---
        kernel_45 = np.array([[ 2,  1,  0],[ 1,  0, -1],[ 0, -1, -2]], dtype=np.float32)
        kernel_135 = np.array([[ 0,  1,  2],[-1,  0,  1],[-2, -1,  0]], dtype=np.float32)
        response_45 = convolve(img_gray_np, kernel_45)
        response_135 = convolve(img_gray_np, kernel_135)
        diagonal_map = np.maximum(np.abs(response_45), np.abs(response_135))
        diagonal_map_resized = resize(diagonal_map, (H, W), anti_aliasing=True, mode='reflect')
        diagonal_features_flat = diagonal_map_resized.reshape(N, 1)

        # --- 2. "가로선" 피처 (sobel_v 사용) ---
        horizontal_map = np.abs(sobel_v(img_gray_np))
        horizontal_map_resized = resize(horizontal_map, (H, W), anti_aliasing=True, mode='reflect')
        horizontal_features_flat = horizontal_map_resized.reshape(N, 1)

        # -------------------------------------------------
        # 3.2 (수정) DINO + 대각선/가로선 피처 결합 (Color 제외)
        # -------------------------------------------------
        scaled_diagonal_features = diagonal_features_flat * diagonal_weight
        scaled_horizontal_features = horizontal_features_flat * horizontal_weight
        
        combined_features = np.concatenate([
            features_flat,              # DINO 피처 (128-dim)
            scaled_diagonal_features,   # 대각선 피처 (1-dim)
            scaled_horizontal_features  # 가로선 피처 (1-dim)
        ], axis=1)

        # -------------------------------------------------
        # 4. PCA 피처맵 생성 (시각화용 - DINO Only)
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
        # 5.1 (수정) Crack 레이블 자동 탐색 (DINO 이질성 + 대각선)
        # -------------------------------------------------
        cluster_avg_diagonals = []
        cluster_avg_horizontals = []
        cluster_sizes = np.bincount(labels, minlength=k) # ❗️ 클러스터 크기 계산 (minlength=k 추가)

        # ❗️ (수정) DINO 피처(128) 및 가중치 피처에 대한 클러스터 중심점
        centroids = kmeans.cluster_centers_ 
        centroids_dino = centroids[:, :128] # (0 ~ 127) DINO
        # (Combined_features 순서에 맞게 인덱싱)
        centroids_diag_scaled = centroids[:, 128] # (128) Scaled Diagonal
        centroids_horiz_scaled = centroids[:, 129] # (129) Scaled Horizontal

        # 1. (수정) 각 클러스터의 "평균 가중치" 점수 계산 (K-Means 중심점 사용)
        # (참고: K-Means가 이미 가중치(e.g., 10.0)가 곱해진 피처로 학습했으므로, 
        # 중심점(centroid) 값을 가중치로 나눠서 [0, 1] 범위로 되돌립니다.)
        for i in range(k):
            # ❗️ (개선) 평균을 다시 계산하는 대신 K-Means 중심점(Centroid)을 사용
            # (가중치로 나눠서 0~1 스케일로 복원)
            avg_diagonal = (centroids_diag_scaled[i] / diagonal_weight) if diagonal_weight > 1e-6 else 0.0
            avg_horizontal = (centroids_horiz_scaled[i] / horizontal_weight) if horizontal_weight > 1e-6 else 0.0
            
            # (0 미만 값 방지 - Sobel/Convolve는 음수가 없음)
            cluster_avg_diagonals.append(max(0, avg_diagonal))
            cluster_avg_horizontals.append(max(0, avg_horizontal))

        # 2. "가중치" 점수 정규화
        scores_diagonal = np.array(cluster_avg_diagonals)
        if np.max(scores_diagonal) > 1e-6:
            scores_diagonal = scores_diagonal / np.max(scores_diagonal)

        scores_horizontal = np.array(cluster_avg_horizontals)
        if np.max(scores_horizontal) > 1e-6:
            scores_horizontal = scores_horizontal / np.max(scores_horizontal)

        # 3. ❗️ (신규) "배경과 이질적인 것" (Heterogeneity) 점수 계산
        if np.sum(cluster_sizes) == 0: 
            print("❗️ K-Means 레이블이 비어있습니다. 스킵합니다.")
            continue

        # (1) 가장 큰 클러스터를 "메인 배경"으로 간주
        main_background_label = np.argmax(cluster_sizes)
        centroid_bg_dino = centroids_dino[main_background_label]

        # (2) 모든 클러스터와 "메인 배경"간의 DINO 피처 거리 계산
        distances_from_bg = [np.linalg.norm(centroids_dino[i] - centroid_bg_dino) for i in range(k)]

        # (3) 거리 점수(이질성) 정규화 (0~1)
        scores_heterogeneity = np.array(distances_from_bg)
        if np.max(scores_heterogeneity) > 1e-6:
            scores_heterogeneity = scores_heterogeneity / np.max(scores_heterogeneity)
        # (메인 배경 클러스터의 이질성 점수는 0이 됩니다)

        # --- 레이블 선정 (핵심 로직: 이질성 + 대각선) ---
        hetero_weight = 0.5        # 이질성 점수 가중치
        diag_weight = 0.5          # 대각선 점수 가중치
        horiz_penalty_weight = 1.0 # 가로선 페널티 강도

        final_scores = (scores_heterogeneity * hetero_weight) + (scores_diagonal * diag_weight)
        final_scores_penalized = final_scores - (scores_horizontal * horiz_penalty_weight) # 가로선 점수만큼 뺌

        # (디버깅용 상세 점수 출력)
        print("\n--- Cluster Scores (Label: Hetero, Diagonal, Horizontal) -> Final ---")
        for i in range(k):
            is_bg = "(BG)" if i == main_background_label else ""
            print(f"  Label {i}: Hetero={scores_heterogeneity[i]:.2f}, Diag={scores_diagonal[i]:.2f}, Horiz={scores_horizontal[i]:.2f} -> Final={final_scores_penalized[i]:.2f} {is_bg}")
        print("---------------------------------------------------------")

        # ❗️ (변경) 최종 점수가 특정 임계값(e.g., 상위 80% percentile) 이상인 것들을 모두 선택
        PERCENTILE_THRESHOLD = 80 

        positive_scores = final_scores_penalized[final_scores_penalized > 0]
        if len(positive_scores) > 0:
            if len(positive_scores) == 1:
                thresh_val = positive_scores[0] * 0.9 
            else:
                thresh_val = np.percentile(positive_scores, PERCENTILE_THRESHOLD)
        else:
            thresh_val = 0.5 

        # ❗️ 0점(배경)보다 크고, 임계값보다 높은 모든 레이블 선택
        final_crack_labels = list(np.where((final_scores_penalized > 0) & (final_scores_penalized >= thresh_val))[0])

        print(f"✅ Crack (Hetero+Diag) identified as Labels: {final_crack_labels} (Thresh > {thresh_val:.2f})")
        
        # -------------------------------------------------
        # 5.2 매끈한 마스크 후처리
        # -------------------------------------------------
        # (np.isin을 사용하여 다중 레이블 마스크 생성)
        binary_mask_lowres = np.isin(kmeans_img, final_crack_labels).astype(np.uint8) * 255
        binary_mask_pil = Image.fromarray(binary_mask_lowres)
        smooth_mask_pil = binary_mask_pil.resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)
        smooth_mask_np = np.array(smooth_mask_pil)
        threshold = 127 
        final_mask_smooth = (smooth_mask_np > threshold).astype(np.uint8)

        # -------------------------------------------------
        # 6. 시각화 및 저장 (Crack 오버레이)
        # -------------------------------------------------
        # ❗️ (수정) 저장 파일명 변경
        save_image_path = os.path.join(save_dir, f"{base_name}_DINO_HeteroDiag_Overlay.png") 

        plt.figure(figsize=(15, 6)) 

        # --- 1번: 원본 이미지 ---
        plt.subplot(1, 3, 1) 
        plt.imshow(img_cropped) 
        plt.axis("off")
        plt.title(f"Input image, {crop_size}x{crop_size}")

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