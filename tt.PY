import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision import transforms as T
from PIL import Image
import numpy as np
import warnings

# DINOv3 (사용자 코드)에서 사용하는 기본값
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# --- 1. 모델 정의 (제공된 LinearHead 클래스) ---

class LinearHead(nn.Module):
    """Linear layer ."""

    def __init__(
        self,
        in_channels,
        n_output_channels,
        use_batchnorm=True,
        use_cls_token=False,
    ):
        super().__init__()
        # config에 따라 in_channels는 [768] 같은 리스트가 됩니다.
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        if use_cls_token:
            self.channels *= 2  # concatenate CLS to patch tokens
        self.n_output_channels = n_output_channels
        self.use_cls_token = use_cls_token
        # config의 use_batchnorm: True 이므로 SyncBatchNorm 사용
        self.batchnorm_layer = nn.BatchNorm2d(self.channels) if use_batchnorm else nn.Identity(self.channels)
        # (추론 시에는 SyncBatchNorm 대신 BatchNorm2d를 사용해도 동일합니다)
        self.conv = nn.Conv2d(self.channels, self.n_output_channels, kernel_size=1, padding=0, stride=1)
        self.dropout = nn.Dropout2d(0.1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder."""
        # config의 backbone_out_layers: "LAST"는 피처맵이 1개이므로
        # 이 함수는 사실상 입력을 그대로 반환합니다.
        inputs = [
            torch.nn.functional.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for x in inputs
        ]
        inputs = torch.cat(inputs, dim=1)
        return inputs

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel."""
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if self.use_cls_token:
                # (config = False 이므로 이 부분은 실행되지 않음)
                assert len(x) == 2, "Missing class tokens"
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                # (config = False 이므로 이 부분이 실행됨)
                if len(x.shape) == 2: # (B, N, C) 같은 형태일 경우
                    x = x[:, :, None, None] # (B, N, C, 1) -> (이러면 안됨, Reshape 필요)
                inputs[i] = x
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):
        """Forward function (Training)."""
        output = self._forward_feature(inputs)
        output = self.dropout(output) # 학습 시 드롭아웃 적용
        output = self.batchnorm_layer(output)
        output = self.conv(output)
        return output

    def predict(self, x, rescale_to=(512, 512)):
        """
        Predict function used in evaluation. (No dropout)
        """
        x = self._forward_feature(x)
        x = self.batchnorm_layer(x) # 드롭아웃 없음
        x = self.conv(x)
        x = F.interpolate(input=x, size=rescale_to, mode="bilinear") # 업샘플링
        return x

# --- 2. 백본 + 헤드 결합 모델 ---
class SimpleSegModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.patch_size = backbone.patch_embed.patch_size[0]

    def forward(self, img):
        # 1. 백본에서 피처 추출
        with torch.no_grad(): # 백본은 고정
            features = self.backbone.forward_features(img)
        
        # config의 "LAST" 레이어, "use_cls_token: False"
        patch_tokens = features['x_norm_patchtokens'] # (B, N, C_feat)

        # 2. 패치 토큰을 이미지 2D 형태로 복원
        B, N, C_feat = patch_tokens.shape
        H_feat = img.shape[2] // self.patch_size
        W_feat = img.shape[3] // self.patch_size
        
        # (B, N, C_feat) -> (B, C_feat, H_feat, W_feat)
        patch_tokens_reshaped = patch_tokens.permute(0, 2, 1).reshape(B, C_feat, H_feat, W_feat)

        # 3. LinearHead의 'predict' 메서드 호출
        #    LinearHead는 입력을 리스트 [ ] 로 받아야 합니다.
        head_inputs = [patch_tokens_reshaped]
        
        #    head.predict가 업샘플링까지 모두 처리합니다.
        logits = self.head.predict(head_inputs, rescale_to=(img.shape[2], img.shape[3]))
        
        return logits

# --- 3. 오버레이 시각화 함수 ---
def visualize_overlay(original_pil_img, mask_numpy, color=(255, 0, 0), alpha=0.5):
    """
    원본 PIL 이미지에 세그멘테이션 마스크를 오버레이합니다.
    """
    mask_pil = Image.fromarray((mask_numpy * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(original_pil_img.size, Image.NEAREST)
    color_mask_pil = Image.new("RGB", original_pil_img.size, color)
    img_rgba = original_pil_img.convert("RGBA")
    color_mask_rgba = color_mask_pil.convert("RGBA")
    color_mask_rgba.putalpha(Image.fromarray((np.array(mask_pil) > 0) * int(255 * alpha), 'L'))
    combined_img = Image.alpha_composite(img_rgba, color_mask_rgba)
    return combined_img.convert("RGB")

# --- 4. 메인 추론 함수 ---
def main_inference():
    # --- 설정 ---
    HEAD_WEIGHTS_PATH = "weights/dinov3_vitb16_pretrain.pth" # ⚠️ 여기에 헤드 가중치 경로 입력
    IMAGE_PATH = "imgs/seg_test/particle_score(0.91)_B16AAA_POSTDICE_874.jpg"      # ⚠️ 여기에 테스트 이미지 경로 입력
    OUTPUT_PATH = "overlay_result.png"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 512
    NUM_CLASSES = 2
    EMBED_DIM = 768 # ViT-B/16 (DINOv3)
    
    mean = [m * 255 for m in IMAGENET_DEFAULT_MEAN]
    std = [s * 255 for s in IMAGENET_DEFAULT_STD]

    # --- 1. eval_transforms 정의 ---
    inference_transforms = v2.Compose([
        v2.Resize(size=(IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BILINEAR),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(mean=mean, std=std)
    ])

    # --- 2. 모델 로드 ---
    print("모델을 로드합니다...")
    REPO_DIR = 'C:/workspace/dinov3'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backbone = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights = HEAD_WEIGHTS_PATH)
    backbone.eval() # 백본은 항상 eval 모드
    
    # 2) LinearHead 정의 (config 기반)
    head = LinearHead(
        in_channels=[EMBED_DIM],  # config의 "LAST" 레이어 1개
        n_output_channels=NUM_CLASSES,
        use_batchnorm=True,       # config에서 True
        use_cls_token=False       # config에서 False
    )
    
    # 3) 모델 결합
    model = SimpleSegModel(backbone, head).to(DEVICE).eval()
    
    # 4) 학습된 LinearHead 가중치 로드
    try:
        # ⚠️ 중요: LinearHead 전체의 state_dict를 로드합니다.
        model.head.load_state_dict(torch.load(HEAD_WEIGHTS_PATH, map_location=DEVICE))
        print(f"헤드 가중치 로드 성공: {HEAD_WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"경고: '{HEAD_WEIGHTS_PATH}' 에서 가중치 파일을 찾을 수 없습니다. 랜덤 가중치로 추론합니다.")
    except Exception as e:
        print(f"가중치 로드 오류: {e}")
        print("    (팁: 저장된 state_dict가 'LinearHead' 클래스 구조와 일치하는지 확인하세요.)")
        return

    # --- 3. 이미지 로드 및 전처리 ---
    print(f"이미지 로드: {IMAGE_PATH}")
    original_pil_img = Image.open(IMAGE_PATH).convert("RGB")
    input_tensor = inference_transforms(original_pil_img).unsqueeze(0).to(DEVICE)

    # --- 4. 추론 (Dropout 비활성화를 위해 model.eval() 상태) ---
    print("추론을 실행합니다...")
    with torch.no_grad():
        logits = model(input_tensor) # (1, 2, 512, 512)
    
    # --- 5. 후처리 (마스크 생성) ---
    # 모델은 0(배경), 1(객체)을 예측하도록 학습되었습니다.
    pred_mask_tensor = torch.argmax(logits, dim=1).squeeze(0) # (H, W)
    pred_mask_numpy = pred_mask_tensor.cpu().numpy().astype(np.uint8)

    # --- 6. 시각화 및 저장 ---
    print("오버레이 이미지를 생성합니다...")
    overlay_img = visualize_overlay(original_pil_img, pred_mask_numpy, color=(255, 0, 0), alpha=0.5)
    
    overlay_img.save(OUTPUT_PATH)
    print(f"성공! 결과가 '{OUTPUT_PATH}'에 저장되었습니다.")

if __name__ == "__main__":
    main_inference()