import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import skimage.morphology as morph
from editor import MultiClickHandler, EditWindowHandler
# -------------------------------------------------
task = 'crack_hardcases'
input_dir = f"C:/data/251102/test/particle"
save_dir = f"output/interactive_masks/{task}" # ⬅️ 결과 폴더명 변경
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
img_size = 2048
crop_size = 2048

transform_dino = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(crop_size), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

os.makedirs(save_dir, exist_ok=True)
print(f"Processing images from: {input_dir}") 
print(f"Saving results to: {save_dir}")       

# --- 하이퍼파라미터 (상단 고정) ---
k = 15 # 마스크 개수
n_init = 10
crack_color = [255, 0, 0] 
alpha = 0.5

# -------------------------------------------------
# (A) 헬퍼 함수: PCA 시각화를 위한 정규화
# -------------------------------------------------
def min_max_scale(img):
    """PCA 결과를 [0, 1] 범위로 정규화하는 함수"""
    img_min = img.min(axis=(0, 1))
    img_max = img.max(axis=(0, 1))
    return (img - img_min) / (img_max - img_min + 1e-6)


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
        feat_tokens = feat_tokens[:,:,:256] 
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
        save_overlay_path = os.path.join(save_dir, f"{base_name}_Overlay.png")
        Image.fromarray(final_overlay_to_save).save(save_overlay_path)

        # 바이너리 마스크(흑백)도 별도 저장
        save_mask_path = os.path.join(save_dir, f"{base_name}_Mask_L.png")
        Image.fromarray(final_mask_to_save * 255).save(save_mask_path)
        
        tqdm.write(f"   Saved Final Overlay: {save_overlay_path}")
        tqdm.write(f"   Saved Final Mask: {save_mask_path}")

    except Exception as e:
        tqdm.write(f"❗️ FAILED to process {filename}: {e}") 
        if plt.get_fignums(): 
            plt.close('all')

print("\n--- All processing complete. ---")