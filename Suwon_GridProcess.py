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
        fade_range: Tuple[float, float] = (0.03, 0.12),  # 페이드 속도 범위 추가
    ):
        self.canvas_w, self.canvas_h = canvas_size
        self.scale_range = scale_range
        self.fade_range = fade_range  # 페이드 속도 범위 저장
        self.spawn_queue = []  # 나중에 스폰될 이미지들을 위한 큐 추가

        # 원본 PIL 그레이 이미지 로드
        self.pil_images: List[Image.Image] = [
            Image.open(p).convert("L") for p in image_paths
        ]
        self.active: List[dict] = []

    def _prepare_image(self):
        """이미지 하나를 준비하고 정보를 반환"""
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
        
        # 랜덤 페이드 속도 설정
        fade = random.uniform(self.fade_range[0], self.fade_range[1])
        
        return {
            "tex": tex,          # 1×3×new_h×new_w
            "pos": (x, y),
            "alpha": 1.0,
            "fade": fade,
            "delay": random.randint(0, 7)  # 추가된 필드: 0-7 프레임 사이의 지연
        }

    def spawn(self, count: int):
        # 이미지들을 큐에 추가
        for _ in range(count):
            self.spawn_queue.append(self._prepare_image())

    def update(self):
        # 지연 시간이 0이 된 이미지들을 큐에서 active로 이동
        remaining_queue = []
        for img in self.spawn_queue:
            if img["delay"] <= 0:
                self.active.append(img)
            else:
                img["delay"] -= 1
                remaining_queue.append(img)
        self.spawn_queue = remaining_queue

    def render(self, canvas: torch.Tensor) -> torch.Tensor:
        # 먼저 큐에 있는 이미지들 업데이트
        self.update()
        
        # 기존 렌더링 코드 동일하게 유지
        out = canvas.clone()
        to_remove = []  # 제거할 객체 목록
        
        for i, obj in enumerate(self.active):
            try:
                tex, (x, y), a = obj["tex"], obj["pos"], obj["alpha"]
                _, _, th, tw = tex.shape
                
                # 캔버스 경계 체크
                if x < 0 or y < 0 or x + tw > self.canvas_w or y + th > self.canvas_h:
                    # 경계를 벗어난 이미지는 제거 대상으로 표시
                    to_remove.append(i)
                    continue
                    
                # 잘못된 인덱싱 방지
                if y + th > out.shape[2] or x + tw > out.shape[3]:
                    print(f"잘못된 인덱싱: 이미지 크기 {th}x{tw}, 위치 ({x},{y}), 캔버스 크기 {out.shape}")
                    to_remove.append(i)
                    continue
                    
                # 텐서 크기 확인 - 실제 슬라이싱 전에 크기 계산하여 확인
                # out 텐서는 [batch, channel, height, width] 형태
                region_h = min(th, out.shape[2] - y)
                region_w = min(tw, out.shape[3] - x)
                
                if region_h != th or region_w != tw:
                    # 크기가 다르면 텍스처를 조정
                    print(f"텍스처 크기 조정: 원본 {th}x{tw} -> 조정 {region_h}x{region_w}")
                    # 이 경우 해당 영역을 건너뛰는 게 안전
                    to_remove.append(i)
                    continue
                
                # 이제 안전하게 텐서 조작 가능
                region = out[..., y:y+th, x:x+tw]
                # 최종 검증
                if region.shape != tex.shape:
                    print(f"최종 크기 불일치: 영역 {region.shape}, 텍스처 {tex.shape}")
                    to_remove.append(i)
                    continue
                    
                # 알파 블렌딩 적용
                out[..., y:y+th, x:x+tw] = region * (1 - a) + tex * a
                
                # 페이드 처리
                obj["alpha"] -= obj["fade"]
                if obj["alpha"] <= 0:
                    to_remove.append(i)
                
            except Exception as e:
                print(f"렌더링 오류: {e}")
                # 오류가 발생한 이미지는 제거 대상으로 표시
                to_remove.append(i)
        
        # 제거할 객체들을 인덱스 역순으로 제거 (인덱스 변화 방지)
        for idx in sorted(to_remove, reverse=True):
            if 0 <= idx < len(self.active):  # 인덱스 범위 확인
                self.active.pop(idx)
                
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
    Lines are drawn only as horizontal and vertical segments (90-degree angles).
    """
    def __init__(self, color: Tuple[int, int, int] = (255, 255, 255), width: int = 2):
        self.color = color
        self.width = width

    def draw(self, image: Image.Image, centers: List[Tuple[int, int]]) -> None:
        # 최소 두 개의 점이 있을 때만 처리
        if len(centers) < 2:
            return
            
        draw = ImageDraw.Draw(image)
        
        # 직선 대신 수직/수평 선분으로 연결
        for i in range(len(centers) - 1):
            p1 = centers[i]
            p2 = centers[i + 1]
            
            # 두 점 사이의 "ㄱ" 또는 "ㄴ" 형태로 연결
            # (x1, y1) -> (x2, y1) -> (x2, y2) 또는
            # (x1, y1) -> (x1, y2) -> (x2, y2)
            
            # 랜덤하게 경로 결정 (수직 먼저 또는 수평 먼저)
            if random.choice([True, False]):
                # 수평 선분 먼저, 그 다음 수직 선분
                mid_point = (p2[0], p1[1])  # (x2, y1)
                draw.line([p1, mid_point], fill=self.color, width=self.width)
                draw.line([mid_point, p2], fill=self.color, width=self.width)
            else:
                # 수직 선분 먼저, 그 다음 수평 선분
                mid_point = (p1[0], p2[1])  # (x1, y2)
                draw.line([p1, mid_point], fill=self.color, width=self.width)
                draw.line([mid_point, p2], fill=self.color, width=self.width)
        
        # 각 중심점에 작은 원 표시
        radius = self.width
        for x, y in centers:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=self.color)


class VideoGenerator:
    """전체 프레임 생성 및 저장을 담당 (변동 없음)"""
    def __init__(self, image_folder: str, output_folder: str = "output_frames", **kwargs):
        paths = glob.glob(os.path.join(image_folder, "*.jpg"))
        os.makedirs(output_folder, exist_ok=True)

        self.spawner = ImageSpawner(
            paths, 
            kwargs.get("canvas_size", (2048,2048)),
            fade_range=kwargs.get("fade_range", (0.03, 0.12))
        )
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
        image_folder="suwon_budget_images",
        output_folder="torch_feedback_output",
        canvas_size=(2048, 2048),
        num_frames=300,
        interval=8,
        per_spawn=60,
        fade_range=(0.03, 0.12)  # 페이드 속도 범위 추가
    )
    gen.run()
