import os
import glob
from typing import Tuple

import numpy as np
from PIL import Image
import torch

from spawner import ImageSpawner
from connector import LineConnector
from feedback import TorchFeedback

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(
    image_folder: str,
    output_folder: str,
    canvas_size: Tuple[int, int] = (2048, 2048),
    num_frames: int = 300,
    interval: int = 8,
    per_spawn: int = 60,
    fade_range=(0.03, 0.12),
    visual_style="modern_grid",
    enable_effects=True,
    bg_color=(30, 30, 30),
    pixelate_size=None,
    use_3d=False,
    perspective_vertical=0.7
):
    paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    os.makedirs(output_folder, exist_ok=True)
    spawner = ImageSpawner(paths, canvas_size, fade_range=fade_range, pixelate_size=pixelate_size)
    if visual_style == "modern_grid":
        connector = LineConnector(color=(230, 230, 230), width=2, highlight_color=(255, 60, 60),
                                  grid_color=(160, 160, 160), show_grid=True, grid_spacing=96)
    elif visual_style == "green_data":
        connector = LineConnector(color=(180, 230, 180), width=2, highlight_color=(50, 200, 50),
                                  grid_color=(120, 160, 120), show_grid=True, grid_spacing=128)
    else:
        connector = LineConnector()
    feedback = TorchFeedback(size=canvas_size, enable_effects=enable_effects)
    bg_tensor = torch.tensor(bg_color).float() / 255.0
    base_canvas = bg_tensor.view(1, 3, 1, 1).repeat(1, 1, *canvas_size).to(DEVICE)
    for i in range(num_frames):
        canvas = base_canvas.clone()
        if i % interval == 0:
            spawner.spawn(per_spawn)
        canvas = spawner.render(canvas)
        pil = Image.fromarray((canvas[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        connector.draw(pil, spawner.centers())
        arr = np.array(pil, dtype=np.float32) / 255.0
        canvas = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        out = feedback.apply(canvas)
        frame = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(frame).save(os.path.join(output_folder, f"frame_{i:04d}.png"))
        print(f"[{i+1}/{num_frames}] Saved frame_{i:04d}.png")
    print("Done.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main(
        image_folder=os.path.join(script_dir, "suwon_budget_images"),
        output_folder=os.path.join(script_dir, "torch_feedback_output"),
        use_3d=True,
        canvas_size=(2048, 2048),
        num_frames=300,
        interval=8,
        per_spawn=60,
        fade_range=(0.03, 0.12),
        visual_style="modern_grid",
        enable_effects=True,
        bg_color=(20, 20, 22),
        pixelate_size=(4, 16),
        perspective_vertical=0.5
    ) 