import random
from typing import List, Tuple, Union, Dict

import numpy as np
import torch
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageSpawner:
    def __init__(
        self,
        image_paths: List[str],
        canvas_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.3, 1.4),
        fade_range: Tuple[float, float] = (0.03, 0.12),
        pixelate_size: Union[int, Tuple[int, int]] = None
    ):
        self.canvas_w, self.canvas_h = canvas_size
        self.scale_range = scale_range
        self.fade_range = fade_range
        self.pixelate_size = pixelate_size
        self.spawn_queue: List[Dict] = []
        self.active: List[Dict] = []
        self.pil_images: List[Image.Image] = [Image.open(p).convert("L") for p in image_paths]
        self.zoom_factor = 1.0
        
        # Extended spawn area for zoom out effect (increased for larger zoom range)
        self.extended_w = int(self.canvas_w * 5.0)  # Increased from 2.5 to 5.0
        self.extended_h = int(self.canvas_h * 5.0)  # Increased from 2.5 to 5.0

    def set_zoom_factor(self, zoom_factor: float):
        """Set zoom factor for camera effect. zoom_factor > 1.0 means zoom out."""
        self.zoom_factor = zoom_factor

    def _prepare_image(self) -> Dict:
        pil = random.choice(self.pil_images)
        orig_w, orig_h = pil.size
        
        # Use extended area for spawning
        max_scale = min(self.extended_w / orig_w, self.extended_h / orig_h, 1.0)
        scale = random.uniform(self.scale_range[0], self.scale_range[1]) * max_scale
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        pil_resized = pil.resize((new_w, new_h), resample=Image.LANCZOS)
        
        pixel_n = (
            random.randint(self.pixelate_size[0], self.pixelate_size[1])
            if isinstance(self.pixelate_size, (tuple, list))
            else self.pixelate_size
        )
        if pixel_n and pixel_n > 1:
            small_w = max(1, new_w // pixel_n)
            small_h = max(1, new_h // pixel_n)
            pil_resized = pil_resized.resize((small_w, small_h), resample=Image.NEAREST)
            pil_resized = pil_resized.resize((new_w, new_h), resample=Image.NEAREST)
            
        arr = np.array(pil_resized, dtype=np.float32) / 255.0
        tex = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).to(DEVICE).unsqueeze(0)
        
        # Spawn in extended area (centered around canvas)
        offset_x = (self.extended_w - self.canvas_w) // 2
        offset_y = (self.extended_h - self.canvas_h) // 2
        
        # ensure non-negative range for spawning coordinates
        max_x = max(self.extended_w - new_w, 0)
        max_y = max(self.extended_h - new_h, 0)
        x = random.randint(0, max_x) - offset_x
        y = random.randint(0, max_y) - offset_y
        
        fade = random.uniform(self.fade_range[0], self.fade_range[1])
        return {"tex": tex, "pos": (x, y), "alpha": 1.0, "fade": fade, "delay": random.randint(0, 7)}

    def spawn(self, count: int):
        for _ in range(count):
            self.spawn_queue.append(self._prepare_image())

    def update(self):
        remaining = []
        for img in self.spawn_queue:
            if img["delay"] <= 0:
                self.active.append(img)
            else:
                img["delay"] -= 1
                remaining.append(img)
        self.spawn_queue = remaining

    def render(self, canvas: torch.Tensor) -> torch.Tensor:
        self.update()
        out = canvas.clone()
        
        # Calculate zoom transformation
        center_x = self.canvas_w // 2
        center_y = self.canvas_h // 2
        
        to_remove = []
        for i, obj in enumerate(self.active):
            tex, (x, y), a = obj["tex"], obj["pos"], obj["alpha"]
            _, _, th, tw = tex.shape
            
            # Apply zoom transformation (scale and translate)
            if self.zoom_factor != 1.0:
                # Transform position based on zoom
                scaled_x = center_x + (x - center_x) / self.zoom_factor
                scaled_y = center_y + (y - center_y) / self.zoom_factor
                scaled_w = tw / self.zoom_factor
                scaled_h = th / self.zoom_factor
            else:
                scaled_x, scaled_y = x, y
                scaled_w, scaled_h = tw, th
            
            # Convert to integer coordinates
            scaled_x = int(scaled_x)
            scaled_y = int(scaled_y)
            scaled_w = max(1, int(scaled_w))
            scaled_h = max(1, int(scaled_h))
            
            # Check if visible in canvas
            if (scaled_x + scaled_w < 0 or scaled_y + scaled_h < 0 or 
                scaled_x >= self.canvas_w or scaled_y >= self.canvas_h):
                obj["alpha"] -= obj["fade"]
                if obj["alpha"] <= 0:
                    to_remove.append(i)
                continue
            
            # Clip to canvas bounds
            clip_x1 = max(0, scaled_x)
            clip_y1 = max(0, scaled_y)
            clip_x2 = min(self.canvas_w, scaled_x + scaled_w)
            clip_y2 = min(self.canvas_h, scaled_y + scaled_h)
            
            if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
                obj["alpha"] -= obj["fade"]
                if obj["alpha"] <= 0:
                    to_remove.append(i)
                continue
            
            # Resize texture if needed
            if scaled_w != tw or scaled_h != th:
                tex_resized = torch.nn.functional.interpolate(
                    tex, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False
                )
            else:
                tex_resized = tex
            
            # Calculate texture coordinates
            tex_x1 = clip_x1 - scaled_x
            tex_y1 = clip_y1 - scaled_y
            tex_x2 = tex_x1 + (clip_x2 - clip_x1)
            tex_y2 = tex_y1 + (clip_y2 - clip_y1)
            
            # Blend texture
            if tex_x2 > tex_x1 and tex_y2 > tex_y1:
                tex_region = tex_resized[..., tex_y1:tex_y2, tex_x1:tex_x2]
                out[..., clip_y1:clip_y2, clip_x1:clip_x2] = (
                    out[..., clip_y1:clip_y2, clip_x1:clip_x2] * (1 - a) + tex_region * a
                )
            
            obj["alpha"] -= obj["fade"]
            if obj["alpha"] <= 0:
                to_remove.append(i)
                
        for idx in sorted(to_remove, reverse=True):
            self.active.pop(idx)
        return out

    def centers(self) -> List[Tuple[int, int]]:
        centers_list = []
        center_x = self.canvas_w // 2
        center_y = self.canvas_h // 2
        
        for obj in self.active:
            x, y = obj["pos"]
            tex = obj["tex"]
            tw, th = tex.shape[3], tex.shape[2]
            
            # Apply zoom transformation
            if self.zoom_factor != 1.0:
                scaled_x = center_x + (x - center_x) / self.zoom_factor
                scaled_y = center_y + (y - center_y) / self.zoom_factor
                scaled_w = tw / self.zoom_factor
                scaled_h = th / self.zoom_factor
            else:
                scaled_x, scaled_y = x, y
                scaled_w, scaled_h = tw, th
            
            center_x_pos = int(scaled_x + scaled_w // 2)
            center_y_pos = int(scaled_y + scaled_h // 2)
            
            # Only include centers visible in canvas
            if (0 <= center_x_pos < self.canvas_w and 
                0 <= center_y_pos < self.canvas_h):
                centers_list.append((center_x_pos, center_y_pos))
                
        return centers_list 