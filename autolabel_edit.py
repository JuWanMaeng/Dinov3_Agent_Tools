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

# ⬇️ Matplotlib 기본 키 충돌 방지
plt.rcParams['keymap.pan'] = []       # 'p' 키 (Pan) 기본 기능 끄기
plt.rcParams['keymap.fullscreen'] = []  # 'f' 키 (Fullscreen) 기본 기능 끄기
plt.rcParams['keymap.save'] = []      # 's' 키 (Save) 기본 기능 끄기
plt.rcParams['keymap.zoom'] = ['o']     # 'o' 키 (Zoom)
plt.rcParams['keymap.back'] = ['z']     # 'z' 키 (Zoom Back)

# -------------------------------------------------
# (A) 헬퍼 함수: PCA 시각화를 위한 정규화
# -------------------------------------------------
def min_max_scale(img):
    """PCA 결과를 [0, 1] 범위로 정규화하는 함수"""
    img_min = img.min(axis=(0, 1))
    img_max = img.max(axis=(0, 1))
    return (img - img_min) / (img_max - img_min + 1e-6)

# -------------------------------------------------
# ⬇️ (수정) [1단계: 선택 창] 헬퍼 클래스
# -------------------------------------------------
class MultiClickHandler:
    def __init__(self, axes_to_label_map, ax_combined, img_cropped_np, stored_binary_masks, crack_color, alpha):
        self.axes_to_label_map = axes_to_label_map
        self.ax_combined = ax_combined 
        self.img_cropped_np = img_cropped_np
        self.stored_binary_masks = stored_binary_masks
        self.crack_color = crack_color
        self.alpha = alpha
        
        self.selected_labels = set() 
        self.final_mask = np.zeros_like(img_cropped_np[:, :, 0], dtype=np.uint8) 

        # ❗️ (NEW) 상태 플래그
        self.action = 'skip' # 'skip', 'save', 'edit'
        
        self._update_combined_plot() # 초기화

    def _update_combined_plot(self, update_mask_from_labels=True):
        """'Combined Result' 서브플롯을 실시간으로 업데이트하는 함수"""
        self.ax_combined.clear() 

        if update_mask_from_labels:
            if not self.selected_labels:
                self.final_mask.fill(0)
            else:
                any_label = next(iter(self.stored_binary_masks))
                merged_mask = np.zeros_like(self.stored_binary_masks[any_label], dtype=bool)
                for label in self.selected_labels:
                    if label in self.stored_binary_masks:
                        merged_mask = np.logical_or(merged_mask, self.stored_binary_masks[label])
                self.final_mask = merged_mask.astype(np.uint8)
        
        final_overlay = self.img_cropped_np.copy()
        mask_3channel = self.final_mask[:, :, np.newaxis]
        final_overlay[mask_3channel.squeeze() == 1] = \
            (final_overlay[mask_3channel.squeeze() == 1] * (1 - self.alpha) + \
             np.array(self.crack_color) * self.alpha).astype(np.uint8)
        
        self.ax_combined.imshow(final_overlay)
        self.ax_combined.set_title(f"Combined ({len(self.selected_labels)} selected)\n[ENTER] Save | [f] Edit | [ESC] Skip")
        self.ax_combined.axis('off')
        self.ax_combined.figure.canvas.draw()

    def on_click(self, event):
        """클릭 이벤트 (선택/해제 토글)"""
        if event.button != 1: return
        ax = event.inaxes
        if ax in self.axes_to_label_map:
            label = self.axes_to_label_map[ax]
            if label in self.selected_labels:
                self.selected_labels.remove(label)
                tqdm.write(f"➡️ Deselected Label: {label}.")
                ax.set_title(f"Label: {label}", color='black', fontweight='normal')
            else:
                self.selected_labels.add(label)
                tqdm.write(f"➡️ Selected Label: {label}.")
                ax.set_title(f"Label: {label} (CLICKED)", color='red', fontweight='bold')
            
            self._update_combined_plot(update_mask_from_labels=True) 

    def on_key_press(self, event):
        """키보드 입력 이벤트 (Enter/f/Esc)"""
        if event.key == 'enter':
            tqdm.write(f"✅ Confirming selection: {sorted(list(self.selected_labels))}")
            self.action = 'save'
            plt.close(event.canvas.figure) 
        
        elif event.key == 'f':
            tqdm.write(f"➡️ Fixing mask for EDITING: {sorted(list(self.selected_labels))}")
            self.action = 'edit'
            plt.close(event.canvas.figure)
        
        elif event.key == 'escape':
            tqdm.write("--- Selection Canceled (Esc). Skipping image. ---")
            self.action = 'skip'
            plt.close(event.canvas.figure)

# -------------------------------------------------
# ⬇️ (NEW) [2단계: 편집 창] 헬퍼 클래스
# -------------------------------------------------
class EditWindowHandler:
    def __init__(self, fig, ax, img_cropped_np, initial_mask, crack_color, alpha):
        self.fig = fig
        self.ax = ax
        self.img_cropped_np = img_cropped_np
        self.final_mask = initial_mask.copy() # ❗️ 전달받은 마스크로 시작
        self.crack_color = crack_color
        self.alpha = alpha

        self.tool = 'paint'
        self.brush_size = 10
        self.is_dragging = False
        self.is_done = False # 'Enter' (True) or 'Esc' (False)

        self._update_plot() # 초기화

    def _get_title(self):
        tool_str = self.tool.upper()
        return f"EDIT MODE (Tool: {tool_str}, Brush: {self.brush_size})\n[Drag] {self.tool} | [p/e] Tool | [+/-] Brush | [ENTER] Save | [ESC] Cancel"

    def _update_plot(self):
        """편집 뷰를 업데이트하는 함수"""
        self.ax.clear()

        final_overlay = self.img_cropped_np.copy()
        mask_3channel = self.final_mask[:, :, np.newaxis]
        final_overlay[mask_3channel.squeeze() == 1] = \
            (final_overlay[mask_3channel.squeeze() == 1] * (1 - self.alpha) + \
             np.array(self.crack_color) * self.alpha).astype(np.uint8)
        
        self.ax.imshow(final_overlay)
        self.ax.set_title(self._get_title())
        self.ax.axis('off')
        self.fig.canvas.draw()

    def on_button_press(self, event):
        if event.button != 1 or event.inaxes != self.ax: return
        self.is_dragging = True
        self.on_motion(event)

    def on_button_release(self, event):
        if event.button != 1: return
        self.is_dragging = False

    def on_motion(self, event):
        if not self.is_dragging or event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None: return

        x, y = int(x), int(y)
        paint_value = 1 if self.tool == 'paint' else 0
        
        y_min = max(0, y - self.brush_size)
        y_max = min(self.final_mask.shape[0], y + self.brush_size)
        x_min = max(0, x - self.brush_size)
        x_max = min(self.final_mask.shape[1], x + self.brush_size)
        
        self.final_mask[y_min:y_max, x_min:x_max] = paint_value
        
        self._update_plot() # 드래그 중 실시간 업데이트

    def on_key_press(self, event):
        """편집 창 키보드 이벤트"""
        
        key_pressed = event.key
        
        if key_pressed == 'e':
            self.tool = 'erase'
            tqdm.write("Tool: ERASE")
        elif key_pressed == 'p':
            self.tool = 'paint'
            tqdm.write("Tool: PAINT (p)")
            
        elif key_pressed == '=' or key_pressed == '+':
            self.brush_size = min(50, self.brush_size + 1)
            tqdm.write(f"Brush Size: {self.brush_size}")
            
        elif key_pressed == '-':
            self.brush_size = max(1, self.brush_size - 1)
            tqdm.write(f"Brush Size: {self.brush_size}")
            
        elif key_pressed == 'enter':
            tqdm.write("✅ Confirming EDITED mask. Saving...")
            self.is_done = True
            plt.close(event.canvas.figure)
            
        elif key_pressed == 'escape':
            tqdm.write("--- Edit Canceled (Esc). ---")
            self.is_done = False # 저장 안 함 플래그
            plt.close(event.canvas.figure)
        
        # Enter/Esc가 아니면, 제목 갱신 (브러시 크기, 툴 이름 등)
        if key_pressed not in ['enter', 'escape']:
            self._update_plot()

# -------------------------------------------------
# (A-2) 경로 설정
# -------------------------------------------------
input_dir = r"C:\workspace\dinov3\imgs\tmp"
save_dir = "output/interactive_masks" # ⬅️ 결과 폴더명 변경
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
img_size = 1024
crop_size = 1024

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
        # 5.3 [1단계: 선택 창] 팝업
        # -------------------------------------------------
        fig_select = plt.figure(figsize=(20, 12)) 
        gs = fig_select.add_gridspec(3, 6, width_ratios=[1,1,1,1,1, 1.5]) 
        
        axes_to_label_map = {}
        
        for i in range(k):
            row = i // 5
            col = i % 5
            ax = fig_select.add_subplot(gs[row, col])
            if i in stored_binary_masks:
                ax.imshow(stored_overlays[i]) 
                ax.set_title(f"Label: {i}")
                axes_to_label_map[ax] = i 
            else:
                ax.set_title(f"Label: {i} (Empty)")
            ax.axis("off")

        ax_combined = fig_select.add_subplot(gs[:, 5])
        
        fig_select.suptitle(f"Review: {filename} - [SELECT Mode]", fontsize=16)
        fig_select.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        grid_handler = MultiClickHandler(
            axes_to_label_map, 
            ax_combined, 
            img_cropped_np, 
            stored_binary_masks, 
            crack_color, 
            alpha
        )
        
        fig_select.canvas.mpl_connect('button_press_event', grid_handler.on_click)
        fig_select.canvas.mpl_connect('key_press_event', grid_handler.on_key_press)

        tqdm.write(f"Waiting for user interaction in [SELECT] window...")
        # ⬇️⬇️⬇️ (추가) 창 위치 및 크기 조절 (WxH+X+Y) ⬇️⬇️⬇️
        try:
            manager = plt.get_current_fig_manager()
            # ❗️ (수정) "가로x세로+X좌표+Y좌표" (e.g., "1600x900+50+50")
            manager.window.geometry("1800x1000+250+5") 
        except Exception as e:
            tqdm.write(f"Warning: Could not set window position ({e})")
        # ⬆️⬆️⬆️ (추가) 여기까지 ⬆️⬆️⬆️
        plt.show() # 1단계 선택 창 팝업 (닫힐 때까지 대기)

        # -------------------------------------------------
        # 5.4 [1단계] 결과 처리
        # -------------------------------------------------
        
        if grid_handler.action == 'skip':
            tqdm.write(f"--- Skipping {filename} ---")
            if plt.fignum_exists(fig_select.number):
                plt.close(fig_select)
            continue
        
        # 'save' 또는 'edit'
        final_mask_to_save = grid_handler.final_mask
        selected_labels = grid_handler.selected_labels
        
        if plt.fignum_exists(fig_select.number):
            plt.close(fig_select)

        # -------------------------------------------------
        # 5.5 (NEW) [2단계: 편집 창] 팝업
        # -------------------------------------------------
        
        if grid_handler.action == 'edit':
            tqdm.write(f"--- Opening [EDIT] window for {filename} ---")
            
            fig_edit, ax_edit = plt.subplots(figsize=(10, 10)) # 1x1 팝업
            
            edit_handler = EditWindowHandler(
                fig_edit,
                ax_edit,
                img_cropped_np,
                final_mask_to_save, # 1단계에서 조합한 마스크 전달
                crack_color,
                alpha
            )
            
            fig_edit.canvas.mpl_connect('button_press_event', edit_handler.on_button_press)
            fig_edit.canvas.mpl_connect('button_release_event', edit_handler.on_button_release)
            fig_edit.canvas.mpl_connect('motion_notify_event', edit_handler.on_motion)
            fig_edit.canvas.mpl_connect('key_press_event', edit_handler.on_key_press)
            
            plt.show() # 2단계 편집 창 팝업 (닫힐 때까지 대기)

            # -------------------------------------------------
            # 5.6 [2단계] 결과 처리
            # -------------------------------------------------
            if not edit_handler.is_done: # 편집 창에서 'Esc'
                tqdm.write(f"--- Edit Canceled. Skipping {filename} ---")
                if plt.fignum_exists(fig_edit.number):
                    plt.close(fig_edit)
                continue
            
            # 'Enter' 누름. 최종 마스크를 덮어씀
            final_mask_to_save = edit_handler.final_mask
            
            if plt.fignum_exists(fig_edit.number):
                plt.close(fig_edit)


        # -------------------------------------------------
        # 6. 최종 마스크 저장 (1단계 'save' 또는 2단계 'edit' 완료)
        # -------------------------------------------------
        
        if not final_mask_to_save.any():
            tqdm.write(f"--- Final mask is empty. Skipping {filename} ---")
            continue

        tqdm.write(f"✅ Saving final mask...")

        # 파일명 조합
        label_str_list = [str(label) for label in sorted(list(selected_labels))]
        label_filename_part = "+".join(label_str_list)
        if grid_handler.action == 'edit':
             label_filename_part = "Edited_" + (label_filename_part if label_filename_part else "Manual")
        
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

    except Exception as e:
        tqdm.write(f"❗️ FAILED to process {filename}: {e}") 
        if plt.get_fignums(): 
            plt.close('all')

print("\n--- All processing complete. ---")