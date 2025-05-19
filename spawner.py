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
        scale_range: Tuple[float, float] = (0.2, 0.8),
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

    def _prepare_image(self) -> Dict:
        pil = random.choice(self.pil_images)
        orig_w, orig_h = pil.size
        max_scale = min(self.canvas_w / orig_w, self.canvas_h / orig_h, 1.0)
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
        max_x = self.canvas_w - new_w
        max_y = self.canvas_h - new_h
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
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
        to_remove = []
        for i, obj in enumerate(self.active):
            tex, (x, y), a = obj["tex"], obj["pos"], obj["alpha"]
            _, _, th, tw = tex.shape
            if x < 0 or y < 0 or x + tw > self.canvas_w or y + th > self.canvas_h:
                to_remove.append(i)
                continue
            region_h = min(th, out.shape[2] - y)
            region_w = min(tw, out.shape[3] - x)
            if region_h != th or region_w != tw:
                to_remove.append(i)
                continue
            out[..., y:y+th, x:x+tw] = out[..., y:y+th, x:x+tw] * (1 - a) + tex * a
            obj["alpha"] -= obj["fade"]
            if obj["alpha"] <= 0:
                to_remove.append(i)
        for idx in sorted(to_remove, reverse=True):
            self.active.pop(idx)
        return out

    def centers(self) -> List[Tuple[int, int]]:
        return [
            (obj["pos"][0] + obj["tex"].shape[3] // 2,
             obj["pos"][1] + obj["tex"].shape[2] // 2)
            for obj in self.active
        ] 