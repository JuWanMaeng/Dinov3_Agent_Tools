import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from tqdm import tqdm
import torch.nn.functional as F

from skimage.transform import resize 
import skimage.morphology as morph

# -------------------------------------------------
# (A) 헬퍼 함수: PCA 시각화를 위한 정규화
# -------------------------------------------------
def min_max_scale(img):
    """PCA 결과를 [0, 1] 범위로 정규화하는 함수"""
    img_min = img.min(axis=(0, 1))
    img_max = img.max(axis=(0, 1))
    return (img - img_min) / (img_max - img_min + 1e-6)

# -------------------------------------------------
# ⬇️ (수정) 펜/지우개 편집 기능이 추가된 헬퍼 클래스
# -------------------------------------------------
class PaintToolHandler:
    def __init__(self, axes_to_label_map, ax_combined, img_cropped_np, stored_binary_masks, crack_color, alpha):
        self.axes_to_label_map = axes_to_label_map
        self.ax_combined = ax_combined 
        self.img_cropped_np = img_cropped_np
        self.stored_binary_masks = stored_binary_masks
        self.crack_color = crack_color
        self.alpha = alpha
        
        self.selected_labels = set() 
        self.is_done = False
        self.final_mask = np.zeros_like(img_cropped_np[:, :, 0], dtype=np.uint8) 

        # ❗️ (NEW) 편집기 상태 변수
        self.mode = 'select'  # 'select' (레이블 선택) vs 'edit' (그리기/지우기)
        self.tool = 'paint'   # 'paint' vs 'erase'
        self.brush_size = 10  # 픽셀 단위 브러시 크기
        self.is_dragging = False # 마우스 드래그 중인지

        # 초기 Combined Result 플롯 설정
        self._update_combined_plot()
        
    def _get_title(self):
        """(NEW) 현재 상태에 맞는 제목 반환"""
        if self.mode == 'select':
            return f"Combined Result ({len(self.selected_labels)} selected)\n[Press 'f' to Fix & Edit]"
        else: # self.mode == 'edit'
            tool_str = self.tool.upper()
            return f"EDIT MODE (Tool: {tool_str})\n[Drag to {self.tool}. 'p'/'e'/'s'. ENTER to save]"

    def _update_combined_plot(self, update_mask_from_labels=False):
        """'Combined Result' 서브플롯을 실시간으로 업데이트하는 함수"""
        
        self.ax_combined.clear() 

        # ❗️ (NEW) 'select' 모드일 때만 레이블로부터 마스크를 재계산
        if self.mode == 'select' and update_mask_from_labels:
            if not self.selected_labels:
                self.final_mask.fill(0)
            else:
                any_label = next(iter(self.stored_binary_masks))
                merged_mask = np.zeros_like(self.stored_binary_masks[any_label], dtype=bool)
                for label in self.selected_labels:
                    if label in self.stored_binary_masks:
                        merged_mask = np.logical_or(merged_mask, self.stored_binary_masks[label])
                self.final_mask = merged_mask.astype(np.uint8)
        
        # 'edit' 모드이거나, 'select' 모드에서 드래그 중이 아닐 때는
        # 현재 'self.final_mask'를 기반으로 오버레이를 생성
        final_overlay = self.img_cropped_np.copy()
        mask_3channel = self.final_mask[:, :, np.newaxis]
        final_overlay[mask_3channel.squeeze() == 1] = \
            (final_overlay[mask_3channel.squeeze() == 1] * (1 - self.alpha) + \
             np.array(self.crack_color) * self.alpha).astype(np.uint8)
        
        self.ax_combined.imshow(final_overlay)
        self.ax_combined.set_title(self._get_title())
        self.ax_combined.axis('off')
        self.ax_combined.figure.canvas.draw()


    def on_button_press(self, event):
        """(NEW) 마우스 버튼 클릭 이벤트"""
        if event.button != 1: return # 좌클릭만
        
        ax = event.inaxes
        if self.mode == 'select' and ax in self.axes_to_label_map:
            # [선택 모드] + 마스크 클릭 -> 레이블 선택/해제
            label = self.axes_to_label_map[ax]
            if label in self.selected_labels:
                self.selected_labels.remove(label)
                tqdm.write(f"➡️ Deselected Label: {label}.")
                ax.set_title(f"Label: {label}", color='black', fontweight='normal')
            else:
                self.selected_labels.add(label)
                tqdm.write(f"➡️ Selected Label: {label}.")
                ax.set_title(f"Label: {label} (CLICKED)", color='red', fontweight='bold')
            
            # ❗️ (중요) 레이블에서 마스크 업데이트
            self._update_combined_plot(update_mask_from_labels=True) 

        elif self.mode == 'edit' and ax == self.ax_combined:
            # [편집 모드] + Combined 뷰 클릭 -> 드래그 시작
            self.is_dragging = True
            self.on_motion(event) # 클릭 지점에도 1회 적용
            
    def on_button_release(self, event):
        """(NEW) 마우스 버튼 해제 이벤트"""
        if event.button != 1: return
        self.is_dragging = False

    def on_motion(self, event):
        """(NEW) 마우스 이동 (드래그) 이벤트"""
        # (편집 모드 + 드래그 중 + Combined 뷰 안)일 때만 작동
        if not self.is_dragging or self.mode != 'edit' or event.inaxes != self.ax_combined:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None: return # 뷰 밖으로 나감

        x, y = int(x), int(y)
        
        # 브러시 값 (1: 칠하기, 0: 지우기)
        paint_value = 1 if self.tool == 'paint' else 0
        
        # 브러시 크기만큼 self.final_mask 수정
        y_min = max(0, y - self.brush_size)
        y_max = min(self.final_mask.shape[0], y + self.brush_size)
        x_min = max(0, x - self.brush_size)
        x_max = min(self.final_mask.shape[1], x + self.brush_size)
        
        self.final_mask[y_min:y_max, x_min:x_max] = paint_value
        
        # ❗️ (중요) 편집된 마스크로 즉시 뷰 업데이트 (레이블 재계산 안함)
        self._update_combined_plot(update_mask_from_labels=False)

    def on_key_press(self, event):
        """키보드 입력 이벤트"""
        
        if self.mode == 'select':
            if event.key == 'f':
                # 'f' (Fix) -> 편집 모드로 전환
                self.mode = 'edit'
                tqdm.write("--- Mode: EDIT ---")
                tqdm.write("Drag on 'Combined' view to draw/erase.")
                tqdm.write("Keys: [p] Paint | [e] Erase | [s] Select Mode | [Enter] Save & Close")
                self._update_combined_plot() # 제목 갱신
            
            elif event.key == 'escape':
                tqdm.write("--- Selection Canceled (Esc). Skipping image. ---")
                self.selected_labels.clear()
                self.is_done = True
                plt.close(event.canvas.figure)
        
        elif self.mode == 'edit':
            if event.key == 'e':
                self.tool = 'erase'
                tqdm.write("Tool: ERASE")
                self._update_combined_plot() # 제목 갱신
            
            elif event.key == 'p':
                self.tool = 'paint'
                tqdm.write("Tool: PAINT")
                self._update_combined_plot() # 제목 갱신
            
            elif event.key == 's':
                # 's' (Select) -> 선택 모드로 복귀
                self.mode = 'select'
                tqdm.write("--- Mode: SELECT ---")
                # ❗️ (중요) 현재 편집본을 놔두고, 레이블로부터 마스크를 다시 계산
                self._update_combined_plot(update_mask_from_labels=True) 
            
            elif event.key == 'enter':
                tqdm.write(f"✅ Confirming EDITED mask. Saving...")
                self.is_done = True
                plt.close(event.canvas.figure) 
            
            elif event.key == 'escape':
                # 'Esc' -> 선택 모드로 복귀 (편집 취소)
                self.mode = 'select'
                tqdm.write("--- Mode: SELECT (Edits Canceled) ---")
                self._update_combined_plot(update_mask_from_labels=True)


# -------------------------------------------------
# (A-2) 경로 설정
# -------------------------------------------------
input_dir = r"C:\workspace\dinov3\imgs\ink" 
save_dir = "output/visualizations_mask_live_EDITOR" # ⬅️ 결과 폴더명 변경
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

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
img_size = 512
crop_size = 512

transform_dino = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(crop_size), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# -------------------------------------------------
# 메인 루프 시작
# -------------------------------------------------
os.makedirs(save_dir, exist_ok=True)
print(f"Processing images from: {input_dir}") 
print(f"Saving results to: {save_dir}")       

# --- 하이퍼파라미터 (상단 고정) ---
k = 15 # 마스크 개수
n_init = 10
crack_color = [255, 0, 0] 
alpha = 0.5

for filename in tqdm(os.listdir(input_dir), desc="Processing images", ncols=100):
    if not filename.lower().endswith(valid_extensions):
        continue

    img_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0] 
    
    tqdm.write(f"\n--- Processing: {filename} ---") 

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
            tqdm.write(f"Skipping {filename}: No features extracted.") 
            continue

        # -------------------------------------------------
        # 5. K-Means 클러스터링
        # -------------------------------------------------
        tqdm.write(f"Running K-Means (k={k}) on {N} Patches...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init) 
        kmeans.fit(features_flat)
        kmeans_img_hw = kmeans.labels_.reshape(H, W)

        # -------------------------------------------------
        # 5.2 k개의 모든 오버레이 마스크 생성 (팝업 전)
        # -------------------------------------------------
        tqdm.write(f"👀 Generating all {k} overlay masks for review...")
        
        stored_overlays = {}
        stored_binary_masks = {}

        for i in range(k):
            binary_mask_lowres = (kmeans_img_hw == i)
            mask_no_noise = morph.remove_small_objects(binary_mask_lowres, min_size=3)
            mask_closed = morph.binary_closing(mask_no_noise, morph.disk(1))
            
            if mask_closed.sum() == 0: 
                continue 

            binary_mask_pil = Image.fromarray(mask_closed.astype(np.uint8) * 255)
            smooth_mask_pil = binary_mask_pil.resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)
            final_mask_smooth = (np.array(smooth_mask_pil) > 127).astype(np.uint8)

            overlay_image = img_cropped_np.copy()
            mask_3channel = final_mask_smooth[:, :, np.newaxis] 
            overlay_image[mask_3channel.squeeze() == 1] = \
                (overlay_image[mask_3channel.squeeze() == 1] * (1 - alpha) + \
                 np.array(crack_color) * alpha).astype(np.uint8)

            stored_overlays[i] = overlay_image
            stored_binary_masks[i] = final_mask_smooth
        
        # -------------------------------------------------
        # 5.3 팝업창 레이아웃 설정 (GridSpec 사용)
        # -------------------------------------------------
        fig = plt.figure(figsize=(20, 12)) 
        gs = fig.add_gridspec(3, 6, width_ratios=[1,1,1,1,1, 1.5]) 
        
        axes_to_label_map = {}
        
        # K-Means 마스크 (3x5 그리드)
        for i in range(k):
            row = i // 5
            col = i % 5
            ax = fig.add_subplot(gs[row, col])
            
            if i in stored_binary_masks:
                ax.imshow(stored_overlays[i]) 
                ax.set_title(f"Label: {i}")
                axes_to_label_map[ax] = i 
            else:
                ax.set_title(f"Label: {i} (Empty)")
            ax.axis("off")

        # Combined Result (우측 3칸 합쳐서 1칸)
        ax_combined = fig.add_subplot(gs[:, 5])
        
        fig.suptitle(f"Review: {filename} - [SELECT Mode] CLICK masks. Press 'f' to Edit.", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # -------------------------------------------------
        # 5.4 다중 클릭 핸들러 연결
        # -------------------------------------------------
        handler = PaintToolHandler( # ❗️ (수정) 새 핸들러 클래스
            axes_to_label_map, 
            ax_combined, 
            img_cropped_np, 
            stored_binary_masks, 
            crack_color, 
            alpha
        )
        
        # ❗️ (NEW) 4개의 이벤트를 연결
        fig.canvas.mpl_connect('button_press_event', handler.on_button_press)
        fig.canvas.mpl_connect('button_release_event', handler.on_button_release)
        fig.canvas.mpl_connect('motion_notify_event', handler.on_motion)
        fig.canvas.mpl_connect('key_press_event', handler.on_key_press)


        tqdm.write(f"Waiting for user interaction...")
        plt.show() # 핸들러가 닫힐 때까지 멈춤

        # -------------------------------------------------
        # 5.5 클릭 결과 확인
        # -------------------------------------------------
        if not handler.is_done or not handler.final_mask.any(): 
            # (is_done이 False = 비정상 종료) or (final_mask가 모두 0 = 스킵)
            tqdm.write(f"--- No mask saved. Skipping {filename} ---")
            plt.close(fig)
            continue 

        # -------------------------------------------------
        # 6. '최종 병합/편집된' 마스크 저장
        # -------------------------------------------------
        tqdm.write(f"✅ Saving final edited mask...")

        final_mask_to_save = handler.final_mask 
        
        # 파일명 조합 (선택된 레이블 기준 or 그냥 "Edited")
        label_str_list = [str(label) for label in sorted(list(handler.selected_labels))]
        label_filename_part = "+".join(label_str_list)
        if not label_filename_part:
            label_filename_part = "Edited" # 레이블 선택 없이 바로 편집한 경우
        
        # 저장용 최종 오버레이 생성
        final_overlay_to_save = img_cropped_np.copy()
        mask_3channel = final_mask_to_save[:, :, np.newaxis]
        final_overlay_to_save[mask_3channel.squeeze() == 1] = \
            (final_overlay_to_save[mask_3channel.squeeze() == 1] * (1 - alpha) + \
             np.array(crack_color) * alpha).astype(np.uint8)

        # 오버레이 이미지 저장
        save_overlay_path = os.path.join(save_dir, f"{base_name}_Overlay_L{label_filename_part}.png")
        Image.fromarray(final_overlay_to_save).save(save_overlay_path)

        # 바이너리 마스크(흑백)도 별도 저장
        save_mask_path = os.path.join(save_dir, f"{base_name}_Mask_L{label_filename_part}.png")
        Image.fromarray(final_mask_to_save * 255).save(save_mask_path)
        
        tqdm.write(f"   Saved Final Overlay: {save_overlay_path}")
        tqdm.write(f"   Saved Final Mask: {save_mask_path}")

        plt.close(fig) 

    except Exception as e:
        tqdm.write(f"❗️ FAILED to process {filename}: {e}") 
        if plt.get_fignums(): 
            plt.close('all')

print("\n--- All processing complete. ---")