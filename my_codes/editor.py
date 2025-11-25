import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm


# ⬇️ Matplotlib 기본 키 충돌 방지
plt.rcParams['keymap.pan'] = []       # 'p' 키 (Pan) 기본 기능 끄기
plt.rcParams['keymap.fullscreen'] = []  # 'f' 키 (Fullscreen) 기본 기능 끄기
plt.rcParams['keymap.save'] = []      # 's' 키 (Save) 기본 기능 끄기
plt.rcParams['keymap.zoom'] = ['o']     # 'o' 키 (Zoom)
plt.rcParams['keymap.back'] = ['z']     # 'z' 키 (Zoom Back)

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