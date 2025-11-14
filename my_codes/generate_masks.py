import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
# ⬅️ (PCA, plt 삭제)
from sklearn.cluster import KMeans
from tqdm import tqdm # ⬅️ (tqdm 복원)

# -------------------------------------------------
# (A) 헬퍼 함수 ⬅️ (min_max_scale 삭제)
# -------------------------------------------------

# -------------------------------------------------
# (A-2) ⬅️ (수정) 경로 설정
# -------------------------------------------------
# ❗️ 1. 이미지가 모여있는 폴더 경로
input_dir = r"C:/data/DinoCrack/images/training" 
# ❗️ 2. Annotation 마스크를 저장할 폴더 경로
save_dir = r"output/annotations" 
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
img_size = 2048
crop_size = 2048 

# (1) DINOv3 입력을 위한 정규화 트랜스폼
transform_dino = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(crop_size), 
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225]),
])

# (2) ⬅️ (복원) Average Color 피처용 트랜스폼 (LANCZOS)
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
print(f"Saving [0, 255] Masks (Darkest Cluster Logic) to: {save_dir}")       

# "검은색" 특징을 얼마나 중요하게 볼지 결정하는 가중치
color_weight = 10.0 # ⬅️ (붙여넣으신 코드 기준 10.0)
# K-Means 클러스터 개수
k = 10 # ⬅️ (붙여넣으신 코드 기준 15)
# K-Means 반복 실행 횟수
n_init = 10 # ⬅️ (붙여넣으신 코드 기준 20)

for filename in tqdm(os.listdir(input_dir), desc="Generating Masks (0, 255)", ncols=100):
    # 1. 확장자 검사
    if not filename.lower().endswith(valid_extensions):
        continue

    # 2. 파일명 및 경로 설정
    img_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0] 
    save_path = os.path.join(save_dir, f"{base_name}.png")

    if os.path.exists(save_path):
        continue
        
    tqdm.write(f"\n--- Processing: {filename} ---") 

    try:
        # -------------------------------------------------
        # 2. 이미지 로드 + 전처리 (루프 내 실행)
        # -------------------------------------------------
        img = Image.open(img_path).convert("RGB")
        x = transform_dino(img).unsqueeze(0).half().to("cuda")
        img_cropped = img.resize((img_size, img_size)).crop((
            (img_size - crop_size) // 2, (img_size - crop_size) // 2,
            (img_size + crop_size) // 2, (img_size + crop_size) // 2
        ))

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
        # 3.1 "검은색" 피처 추출 (Average - LANCZOS)
        # -------------------------------------------------
        transform_color = get_color_transform(H, W)
        color_tensor = transform_color(img_cropped) 
        color_features_flat = color_tensor.permute(1, 2, 0).reshape(N, 1).cpu().numpy()
        
        # -------------------------------------------------
        # 3.2 DINO + Color(Average) 피처 결합
        # -------------------------------------------------
        scaled_color_features = color_features_flat * color_weight
        combined_features = np.concatenate([features_flat, scaled_color_features], axis=1) # (N, 769)

        # -------------------------------------------------
        # 5. K-Means 클러스터링 (769-dim)
        # -------------------------------------------------
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init) 
        labels = kmeans.fit_predict(combined_features) 
        kmeans_img = labels.reshape(H, W) 

        # -------------------------------------------------
        # 5.1 "가장 어두운" Crack 레이블 자동 탐색
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
        tqdm.write(f"✅ Darkest Label identified as: {crack_label} for {filename}")

        # -------------------------------------------------
        # 5.2 매끈한 마스크 후처리
        # -------------------------------------------------
        binary_mask_lowres = (kmeans_img == crack_label).astype(np.uint8) 
        binary_mask_pil = Image.fromarray(binary_mask_lowres) 
        smooth_mask_pil = binary_mask_pil.resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC) 
        smooth_mask_np = np.array(smooth_mask_pil)
        # threshold = 127 
        # final_mask_smooth = (smooth_mask_np > threshold).astype(np.uint8) # (0 또는 1)

        # -------------------------------------------------
        # 6. 마스크 저장 (0과 255)
        # -------------------------------------------------
        final_mask_255 = smooth_mask_np * 255
        mask_image = Image.fromarray(final_mask_255.astype(np.uint8), mode='L') 
        mask_image.save(save_path)
        
    except Exception as e:
        tqdm.write(f"❗️ FAILED to process {filename}: {e}") 

print("\n--- All processing complete. ---")