import torch
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm 
import numpy as np # ⬅️ (추가) NumPy import

# -------------------------------------------------
# (A-2) ⬅️ (추가) 경로 설정
# -------------------------------------------------
# ❗️ 1. 원본 이미지가 모여있는 폴더 경로
input_dir = r"C:\workspace\dinov3\imgs" # ⬅️ (사용자님의 이미지 폴더)
# ❗️ 2. 추출된 피처를 저장할 폴더 경로
output_dir = r"output/features_npy" # ⬅️ (수정) npy 폴더
# ❗️ 3. 처리할 이미지 확장자
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------
# 1. 모델 생성 + weight 로드 (순수 Backbone)
# -------------------------------------------------
REPO_DIR = 'C:/workspace/dinov3'
print("Loading model (Pretrain Backbone only)...")
# (ViT-Base 로드 확인)
model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights='weights/dinov3_vitb16_pretrain.pth')

model = model.half().to("cuda").eval()
print("✅ Model loaded to cuda with .half()")

# -------------------------------------------------
# 2. 전처리용 Transform 정의 (루프 밖에서 한 번만)
# -------------------------------------------------
img_size = 512
crop_size = 512 

transform = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(crop_size), # ⬅️ (핵심 수정) RandomCrop -> CenterCrop
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# -------------------------------------------------
# 3. ⬅️ (핵심 수정) 폴더 순회 및 피처 추출/저장
# -------------------------------------------------
print(f"Starting feature extraction from: {input_dir}")

# 처리할 파일 리스트 생성
file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

for filename in tqdm(file_list, desc="Extracting features"):
    img_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    # ❗️ 피처 저장 경로 (.npy로 변경)
    save_path = os.path.join(output_dir, f"{base_name}.npy") # ⬅️ (수정)

    # (선택) 이미 추출된 피처가 있으면 건너뛰기
    if os.path.exists(save_path):
        continue

    try:
        # 2. 이미지 로드 + 전처리
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).half().to("cuda")

        # 3. Feature 추출 (Backbone만)
        with torch.no_grad():
            feats = model.forward_features(x)

        # (B, N, C) - 예: (1, 1024, 768)
        feat_tokens = feats["x_norm_patchtokens"] 
        
        # 4. ⬅️ (핵심 수정) NumPy로 피처 저장
        
        # (1, 1024, 768) -> (1024, 768)로 만들고, 
        # GPU -> CPU -> NumPy Array로 변환
        feature_to_save = feat_tokens.squeeze(0).cpu().numpy() # ⬅️ (수정) .numpy() 추가
        
        # NumPy를 사용해 .npy 파일로 저장
        np.save(save_path, feature_to_save) # ⬅️ (수정) np.save()

    except Exception as e:
        tqdm.write(f"❗️ FAILED to process {filename}: {e}")

print(f"\n✅ Feature extraction complete. Features saved to: {output_dir}")