import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
# from tqdm import tqdm # (삭제)
import torch.nn.functional as F 

# -------------------------------------------------
# (A) 헬퍼 함수: PCA 시각화를 위한 정규화
# -------------------------------------------------
def min_max_scale(img):
    """PCA 결과를 [0, 1] 범위로 정규화하는 함수"""
    img_min = img.min(axis=(0, 1))
    img_max = img.max(axis=(0, 1))
    return (img - img_min) / (img_max - img_min + 1e-6)

# -------------------------------------------------
# (A-2) ⬅️ (수정) 경로 및 프롬프트 설정
# -------------------------------------------------
# ❗️ 1. 테스트할 단일 이미지의 전체 경로
img_path = r"C:\data\251102\crack\crack_score(0.95)_B16AAA_POSTDICE_885.jpg"
# ❗️ 2. 결과 오버레이를 저장할 폴더 경로
save_dir = "output/visualizations_point_prompt"

# ❗️ 3. (삭제) PROMPT_Y, PROMPT_X (아래에서 ginput으로 받음)

# ❗️ 4. (유지) 유사도 임계값 (튜닝용)
PROMPT_THRESHOLD = 1.5

# -------------------------------------------------
# 1. 모델 생성 + weight 로드
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

# -------------------------------------------------
# ⬅️ (수정) 단일 이미지 처리 시작
# -------------------------------------------------
os.makedirs(save_dir, exist_ok=True)
base_name = os.path.splitext(os.path.basename(img_path))[0]
filename = os.path.basename(img_path) 

print(f"Processing image: {img_path}")
print(f"Saving results to: {save_dir}") 

# 오버레이 설정
crack_color = [255, 0, 0] # 빨간색 (RGB)
alpha = 0.5               # 투명도

try:
    # -------------------------------------------------
    # 2. 이미지 로드 + 전처리
    # -------------------------------------------------
    img = Image.open(img_path).convert("RGB")
    x = transform_dino(img).unsqueeze(0).half().to("cuda")
    img_cropped = img.resize((img_size, img_size)).crop((
        (img_size - crop_size) // 2, 
        (img_size - crop_size) // 2, 
        (img_size + crop_size) // 2, 
        (img_size + crop_size) // 2
    ))
    img_cropped_np = np.array(img_cropped) 

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
        print(f"Skipping {filename}: No features extracted.") 
        raise ValueError(f"No features extracted for {filename}")

    # -------------------------------------------------
    # 4. PCA 피처맵 생성
    # -------------------------------------------------
    print("Running PCA (for Similarity Search and visualization)...") 
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_flat) 
    pca_img = pca_features.reshape(H, W, 3) 
    pca_img = min_max_scale(pca_img) # ⬅️ 시각화용 (0~1 정규화)

    # -------------------------------------------------
    # 4.1 ⬅️ (핵심 추가) plt.ginput()로 좌표 입력받기
    # -------------------------------------------------
    print("\n--- ❗️ 사용자 입력 대기 ❗️ ---")
    print("PCA 맵이 팝업됩니다. Crack 지점을 '클릭'한 후, 팝업 창을 닫아주세요.")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(pca_img)
    plt.title("Click on a Crack patch, then close this window")
    plt.axis('off')
    
    # 1번 클릭할 때까지 스크립트를 중지하고 대기
    # timeout=0은 무한 대기
    click_coords = plt.ginput(1, timeout=0) 
    
    plt.close() # 팝업 창 닫기
    
    if not click_coords:
        print("클릭하지 않고 창을 닫았습니다. 스크립트를 중단합니다.")
        raise SystemExit

    # ginput은 (x, y)로 반환 -> (col, row)
    # (x, y) 좌표를 정수로 변환
    x_coord, y_coord = click_coords[0]
    PROMPT_X = int(np.round(x_coord))
    PROMPT_Y = int(np.round(y_coord))

    print(f"✅ 좌표 입력 완료: Y={PROMPT_Y}, X={PROMPT_X}")

    # -------------------------------------------------
    # 5. ⬅️ (수정) "Point-Click" 유사도 탐색
    # -------------------------------------------------
    print(f"Searching for similar patches based on point ({PROMPT_Y}, {PROMPT_X})...")
    
    # 1. 2D 좌표 -> 1D 인덱스 변환
    prompt_index = PROMPT_Y * W + PROMPT_X
    
    # 2. "프롬프트 벡터" 추출 (주의: 정규화 안된 pca_features 원본 사용)
    prompt_vector = pca_features[prompt_index] # (3,)
    
    # 3. 모든 피처(16384, 3)와 프롬프트 벡터(3,)간의 유클리드 거리 계산
    distances = np.linalg.norm(pca_features - prompt_vector, axis=1)
    
    # 4. 임계값(Threshold)을 기준으로 마스크 생성
    labels = (distances < PROMPT_THRESHOLD).astype(np.uint8)
    
    # 5. (128, 128) 흑백 마스크 생성
    binary_mask_lowres = labels.reshape(H, W) * 255
    
    print(f"✅ Similarity search complete. Found {np.sum(labels)} matching patches.")

    # -------------------------------------------------
    # 6. 매끈한 마스크 후처리
    # -------------------------------------------------
    binary_mask_pil = Image.fromarray(binary_mask_lowres)
    smooth_mask_pil = binary_mask_pil.resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)
    smooth_mask_np = np.array(smooth_mask_pil)
    threshold = 127 
    final_mask_smooth = (smooth_mask_np > threshold).astype(np.uint8)

    # -------------------------------------------------
    # 7. 오버레이 이미지 생성
    # -------------------------------------------------
    overlay_image = img_cropped_np.copy()
    mask_3channel = final_mask_smooth[:, :, np.newaxis] 
    overlay_image = (overlay_image * (1 - alpha) + mask_3channel * np.array(crack_color) * alpha).astype(np.uint8)

    # -------------------------------------------------
    # 8. ⬅️ 1x3 그리드 시각화 (최종 결과 저장)
    # -------------------------------------------------
    save_image_path = os.path.join(save_dir, f"{base_name}_Point_Prompt_Overlay.png") 

    plt.figure(figsize=(15, 6)) # 1x3 그리드

    # --- 1번: 원본 이미지 ---
    plt.subplot(1, 3, 1) 
    plt.imshow(img_cropped) 
    plt.axis("off")
    plt.title(f"Original: {filename}") 

    # --- 2번: PCA 피처맵 (DINO-only) ---
    plt.subplot(1, 3, 2)
    plt.imshow(pca_img) # 128x128
    # ⬅️ (수정) 클릭한 지점(Prompt)을 'x' 마커로 표시
    plt.scatter([PROMPT_X], [PROMPT_Y], c='red', marker='x', s=100) 
    plt.axis("off")
    plt.title(f"PCA Map (Prompt at {PROMPT_Y}, {PROMPT_X})") 

    # --- 3번: 최종 오버레이 결과 ---
    plt.subplot(1, 3, 3) 
    plt.imshow(overlay_image) 
    plt.axis("off")
    plt.title(f"Overlay (Threshold: {PROMPT_THRESHOLD})")
        
    plt.tight_layout()
    plt.savefig(save_image_path, bbox_inches='tight', dpi=300)
    plt.close() 
    
    print(f"✅ 최종 그리드 이미지가 저장되었습니다: {save_image_path}") 

except Exception as e:
    print(f"❗️ FAILED to process {filename}: {e}") 

print("\n--- All processing complete. ---")