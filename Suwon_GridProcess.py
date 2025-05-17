import os
import glob
import math
import random
from typing import Tuple, List

from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from noise import pnoise2

# GPU 세팅
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_perlin(h: int, w: int, scale: float) -> np.ndarray:
    """CPU에서 Perlin noise를 float32 [0,1] 범위로 생성"""
    n = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            n[y, x] = pnoise2(x * scale, y * scale)
    return (n + 1) * 0.5


class TorchFeedback:
    """
    A simple feedback processor that currently returns the canvas unchanged.
    """
    def __init__(self, size: Tuple[int, int]):
        # Store canvas size for future use if needed
        self.canvas_size = size

    def apply(self, canvas: torch.Tensor) -> torch.Tensor:
        # No feedback effect applied; just return the original canvas
        return canvas


class ImageSpawner:
    """랜덤 스폰되는 이미지(텍스처)들을 관리 → 매번 랜덤 스케일 후 캔버스 바운드 체크"""
    def __init__(
        self,
        image_paths: List[str],
        canvas_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.2, 0.8),
    ):
        self.canvas_w, self.canvas_h = canvas_size
        self.scale_range = scale_range

        # 원본 PIL 그레이 이미지 로드
        self.pil_images: List[Image.Image] = [
            Image.open(p).convert("L") for p in image_paths
        ]
        self.active: List[dict] = []

    def spawn(self, count: int):
        for _ in range(count):
            pil = random.choice(self.pil_images)
            orig_w, orig_h = pil.size

            # 캔버스보다 크지 않도록 최대 스케일 계산
            max_scale = min(self.canvas_w / orig_w, self.canvas_h / orig_h, 1.0)
            # 유저 정의 스케일 범위와도 곱해서 최종 스케일 결정
            scale = random.uniform(self.scale_range[0], self.scale_range[1]) * max_scale

            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            pil_resized = pil.resize((new_w, new_h), resample=Image.LANCZOS)

            # NumPy → Tensor 변환 (3×H×W)
            arr = np.array(pil_resized, dtype=np.float32) / 255.0
            tex = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1).to(DEVICE).unsqueeze(0)

            # 위치 랜덤: 항상 0 ≤ x ≤ canvas_w-new_w
            max_x = self.canvas_w - new_w
            max_y = self.canvas_h - new_h
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            self.active.append({
                "tex": tex,          # 1×3×new_h×new_w
                "pos": (x, y),
                "alpha": 1.0,
                "fade": 0.08,
            })

    def render(self, canvas: torch.Tensor) -> torch.Tensor:
        out = canvas.clone()
        for obj in self.active[:]:
            tex, (x, y), a = obj["tex"], obj["pos"], obj["alpha"]
            _, _, th, tw = tex.shape
            out[..., y:y+th, x:x+tw] = (
                out[..., y:y+th, x:x+tw] * (1 - a)
                + tex * a
            )
            obj["alpha"] -= obj["fade"]
            if obj["alpha"] <= 0:
                self.active.remove(obj)
        return out

    def centers(self) -> List[Tuple[int, int]]:
        return [
            (obj["pos"][0] + obj["tex"].shape[3] // 2,
             obj["pos"][1] + obj["tex"].shape[2] // 2)
            for obj in self.active
        ]


class LineConnector:
    """
    Draws lines connecting texture centers on the canvas and marks each center.
    """
    def __init__(self, color: Tuple[int, int, int] = (255, 255, 255), width: int = 2):
        self.color = color
        self.width = width

    def draw(self, image: Image.Image, centers: List[Tuple[int, int]]) -> None:
        # Draw lines between centers if there are at least two points
        draw = ImageDraw.Draw(image)
        if len(centers) > 1:
            draw.line(centers, fill=self.color, width=self.width)
        # Draw small circles at each center
        radius = self.width
        for x, y in centers:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=self.color)


class VideoGenerator:
    """전체 프레임 생성 및 저장을 담당 (변동 없음)"""
    def __init__(self, image_folder: str, output_folder: str = "output_frames", **kwargs):
        paths = glob.glob(os.path.join(image_folder, "*.jpg"))
        os.makedirs(output_folder, exist_ok=True)

        self.spawner = ImageSpawner(paths, kwargs.get("canvas_size", (2048,2048)))
        self.connector = LineConnector()
        self.feedback  = TorchFeedback(size=kwargs.get("canvas_size", (2048,2048)))
        self.base_canvas = torch.zeros(1, 3, *kwargs.get("canvas_size", (2048,2048)), device=DEVICE)

        self.num_frames = kwargs.get("num_frames", 300)
        self.interval   = kwargs.get("interval", 8)
        self.per_spawn  = kwargs.get("per_spawn", 60)
        self.output     = output_folder

    def run(self):
        for i in range(self.num_frames):
            canvas = self.base_canvas.clone()

            if i % self.interval == 0:
                self.spawner.spawn(self.per_spawn)

            canvas = self.spawner.render(canvas)

            # PIL 드로잉
            pil = Image.fromarray(
                (canvas[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            )
            self.connector.draw(pil, self.spawner.centers())

            # 텐서로 복귀
            arr = np.array(pil, dtype=np.float32)/255.0
            canvas = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)

            out = self.feedback.apply(canvas)

            # 파일 저장
            frame = (out[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(frame).save(os.path.join(self.output, f"frame_{i:04d}.png"))
            print(f"[{i+1}/{self.num_frames}] Saved frame_{i:04d}.png")
        print("Done.")


if __name__ == "__main__":
    gen = VideoGenerator(
        image_folder=r"D:\HB\P.UrbanArchitecture\SRC\Suwon_Image_Crawl\suwon_budget_images",
        output_folder="torch_feedback_output",
        canvas_size=(2048, 2048),
        num_frames=300,
        interval=8,
        per_spawn=60,
    )
    gen.run()
