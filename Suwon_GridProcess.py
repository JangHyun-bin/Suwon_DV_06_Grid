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
import cv2  # for perspective transform

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
    Enhanced feedback processor that can add various visual effects.
    """
    def __init__(self, size: Tuple[int, int], enable_effects: bool = True):
        # Store canvas size for future use
        self.canvas_size = size
        self.enable_effects = enable_effects
        self.frame_count = 0
        
        # 효과 설정
        self.vignette_strength = 0.3  # 비네팅 강도
        self.noise_amount = 0.3      # 노이즈 강도
        # color_tint removed to maintain greyscale
        
        # 비네팅 마스크 생성
        h, w = size
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w)
        )
        self.vignette = (x.pow(2) + y.pow(2)).sqrt().pow(0.7)  # 약간 완만하게
        self.vignette = self.vignette.clamp(0, 1).view(1, 1, h, w)
        # feedback loop accumulator for long-exposure effect
        self.feedback_decay = 0.9 # decay factor: closer to 1 => longer trails
        self.accumulator = torch.zeros(1, 3, h, w, device=DEVICE)
    
    def apply(self, canvas: torch.Tensor) -> torch.Tensor:
        self.frame_count += 1
        
        if not self.enable_effects:
            return canvas
            
        # 원본 캔버스 복사
        out = canvas.clone()
        
        # 비네팅 효과 적용 (중앙은 밝게, 가장자리는 어둡게)
        vignette_mask = (1.0 - self.vignette * self.vignette_strength).to(out.device)
        out = out * vignette_mask
        
        # 랜덤 노이즈 추가 (약간의 거친 텍스처)
        if self.noise_amount > 0:
            noise = torch.randn_like(out) * self.noise_amount
            out = (out + noise).clamp(0, 1)
            
        # feedback loop: accumulate frames for long-exposure effect
        self.accumulator = self.accumulator * self.feedback_decay + out * (1 - self.feedback_decay)
        return self.accumulator


class ImageSpawner:
    """랜덤 스폰되는 이미지(텍스처)들을 관리 → 매번 랜덤 스케일 후 캔버스 바운드 체크"""
    def __init__(
        self,
        image_paths: List[str],
        canvas_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.2, 0.8),
        fade_range: Tuple[float, float] = (0.03, 0.12),  # 페이드 속도 범위 추가
        pixelate_size: int = None  # 픽셀화 강도 (None 이면 비활성)
    ):
        self.canvas_w, self.canvas_h = canvas_size
        self.scale_range = scale_range
        self.fade_range = fade_range  # 페이드 속도 범위 저장
        self.pixelate_size = pixelate_size  # 픽셀화 강도 저장
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
        # 픽셀화 적용 (nearest neighbor 축소-확대)
        if self.pixelate_size and self.pixelate_size > 1:
            small_w = max(1, new_w // self.pixelate_size)
            small_h = max(1, new_h // self.pixelate_size)
            pil_resized = pil_resized.resize((small_w, small_h), resample=Image.NEAREST)
            pil_resized = pil_resized.resize((new_w, new_h), resample=Image.NEAREST)

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
    def __init__(
        self, 
        color: Tuple[int, int, int] = (255, 255, 255), 
        width: int = 2,
        highlight_color: Tuple[int, int, int] = (255, 50, 50),  # 하이라이트 색상 (빨간색)
        grid_color: Tuple[int, int, int] = (180, 180, 180),     # 그리드 색상 (회색)
        show_grid: bool = True,                                 # 배경 그리드 표시 여부
        grid_spacing: int = 64,                                 # 그리드 간격
        grid_opacity: float = 0.15                              # 그리드 투명도
    ):
        self.color = color
        self.width = width
        self.highlight_color = highlight_color
        self.grid_color = grid_color
        self.show_grid = show_grid
        self.grid_spacing = grid_spacing
        self.grid_opacity = grid_opacity
        self.highlight_points = []  # 하이라이트할 특별 포인트

    def set_random_highlights(self, centers: List[Tuple[int, int]], count: int = 2):
        """랜덤하게 하이라이트할 포인트 선택"""
        if centers and count > 0:
            count = min(count, len(centers))
            self.highlight_points = random.sample(centers, count)
        else:
            self.highlight_points = []

    def draw_grid(self, image: Image.Image, canvas_size: Tuple[int, int]):
        """배경 그리드 그리기"""
        if not self.show_grid:
            return
            
        draw = ImageDraw.Draw(image, 'RGBA')  # RGBA 모드로 그리기
        w, h = canvas_size
        
        # 수직선
        for x in range(0, w, self.grid_spacing):
            alpha = int(255 * self.grid_opacity)
            draw.line([(x, 0), (x, h)], fill=self.grid_color + (alpha,), width=1)
            
        # 수평선
        for y in range(0, h, self.grid_spacing):
            alpha = int(255 * self.grid_opacity)
            draw.line([(0, y), (w, y)], fill=self.grid_color + (alpha,), width=1)

    def draw(self, image: Image.Image, centers: List[Tuple[int, int]]) -> None:
        # 최소 두 개의 점이 있을 때만 처리
        if len(centers) < 2:
            return
            
        # 배경 그리드 그리기
        self.draw_grid(image, image.size)
        
        # 랜덤 하이라이트 포인트 설정 (필요하면)
        if not self.highlight_points:
            self.set_random_highlights(centers)
            
        draw = ImageDraw.Draw(image)
        
        # 직선 대신 수직/수평 선분으로 연결
        for i in range(len(centers) - 1):
            p1 = centers[i]
            p2 = centers[i + 1]
            
            # 선 색상 - 하이라이트 점이면 강조 색상, 아니면 기본 색상
            line_color = self.highlight_color if (p1 in self.highlight_points or p2 in self.highlight_points) else self.color
            line_width = self.width + 1 if (p1 in self.highlight_points or p2 in self.highlight_points) else self.width
            
            # 두 점 사이의 "ㄱ" 또는 "ㄴ" 형태로 연결
            if random.choice([True, False]):
                # 수평 선분 먼저, 그 다음 수직 선분
                mid_point = (p2[0], p1[1])  # (x2, y1)
                draw.line([p1, mid_point], fill=line_color, width=line_width)
                draw.line([mid_point, p2], fill=line_color, width=line_width)
                
                # 교차점에 작은 점 추가
                if random.random() < 0.7:  # 70% 확률로 교차점 강조
                    r = max(1, line_width//2)
                    draw.ellipse([mid_point[0]-r, mid_point[1]-r, mid_point[0]+r, mid_point[1]+r], fill=line_color)
            else:
                # 수직 선분 먼저, 그 다음 수평 선분
                mid_point = (p1[0], p2[1])  # (x1, y2)
                draw.line([p1, mid_point], fill=line_color, width=line_width)
                draw.line([mid_point, p2], fill=line_color, width=line_width)
                
                # 교차점에 작은 점 추가
                if random.random() < 0.7:  # 70% 확률로 교차점 강조
                    r = max(1, line_width//2)
                    draw.ellipse([mid_point[0]-r, mid_point[1]-r, mid_point[0]+r, mid_point[1]+r], fill=line_color)
        
        # 각 중심점에 원 표시
        for x, y in centers:
            # 하이라이트 포인트인지 확인
            is_highlight = (x, y) in self.highlight_points
            point_color = self.highlight_color if is_highlight else self.color
            radius = self.width * 2 if is_highlight else self.width
            
            # 중심점 표시
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=point_color)
            
            # 하이라이트 포인트에는 추가 효과
            if is_highlight:
                # 외부 원 추가
                outer_radius = radius * 2
                draw.ellipse(
                    [x-outer_radius, y-outer_radius, x+outer_radius, y+outer_radius], 
                    outline=point_color,
                    width=2
                )


class VideoGenerator:
    """전체 프레임 생성 및 저장을 담당"""
    def __init__(self, image_folder: str, output_folder: str = "output_frames", **kwargs):
        paths = glob.glob(os.path.join(image_folder, "*.jpg"))
        os.makedirs(output_folder, exist_ok=True)

        self.spawner = ImageSpawner(
            paths, 
            kwargs.get("canvas_size", (2048,2048)),
            fade_range=kwargs.get("fade_range", (0.03, 0.12)),
            pixelate_size=kwargs.get("pixelate_size", None)
        )
        
        # 시각적 스타일 설정
        visual_style = kwargs.get("visual_style", "modern_grid")
        if visual_style == "modern_grid":
            self.connector = LineConnector(
                color=(230, 230, 230),                # 연결선 색상 (밝은 회색)
                width=2,                              # 선 두께
                highlight_color=(255, 60, 60),        # 하이라이트 색상 (빨간색)
                grid_color=(160, 160, 160),           # 그리드 색상 (어두운 회색)
                show_grid=True,                       # 그리드 표시
                grid_spacing=96                       # 그리드 간격
            )
        elif visual_style == "green_data":
            self.connector = LineConnector(
                color=(180, 230, 180),                # 연결선 색상 (연한 녹색)
                width=2,                              # 선 두께
                highlight_color=(50, 200, 50),        # 하이라이트 색상 (녹색)
                grid_color=(120, 160, 120),           # 그리드 색상 (어두운 녹색)
                show_grid=True,                       # 그리드 표시
                grid_spacing=128                      # 그리드 간격
            )
        else:
            self.connector = LineConnector()          # 기본 스타일
        
        self.feedback = TorchFeedback(
            size=kwargs.get("canvas_size", (2048,2048)),
            enable_effects=kwargs.get("enable_effects", True)
        )
        
        # 배경 색상 설정 (약간 어두운 회색 배경)
        bg_color = kwargs.get("bg_color", (30, 30, 30))
        bg_tensor = torch.tensor(bg_color).float() / 255.0
        self.base_canvas = bg_tensor.view(1, 3, 1, 1).repeat(1, 1, *kwargs.get("canvas_size", (2048,2048))).to(DEVICE)

        # perspective transform initialization for 2.5D effect
        self.use_3d = kwargs.get("use_3d", False)
        if self.use_3d:
            h, w = kwargs.get("canvas_size", (2048,2048))
            alpha = kwargs.get("perspective_alpha", 0.2)
            src = np.float32([[0,0],[w,0],[w,h],[0,h]])
            dst = np.float32([[w*alpha,0],[w*(1-alpha),0],[w,h],[0,h]])
            self.persp_M = cv2.getPerspectiveTransform(src, dst)

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

            # apply perspective warp for 2.5D effect
            frame_np = (out[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            if self.use_3d:
                h, w = frame_np.shape[:2]
                warped = cv2.warpPerspective(frame_np, self.persp_M, (w, h))
                Image.fromarray(warped).save(os.path.join(self.output, f"frame_{i:04d}.png"))
                continue

            # 파일 저장
            frame = (out[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(frame).save(os.path.join(self.output, f"frame_{i:04d}.png"))
            print(f"[{i+1}/{self.num_frames}] Saved frame_{i:04d}.png")
        print("Done.")


if __name__ == "__main__":
    gen = VideoGenerator(
        image_folder="suwon_budget_images",
        output_folder="torch_feedback_output",
        use_3d=True,
        canvas_size=(2048, 2048),
        num_frames=300,
        interval=8,
        per_spawn=60,
        fade_range=(0.03, 0.12),       # 페이드 속도 범위
        visual_style="modern_grid",     # 시각적 스타일 ('modern_grid' 또는 'green_data')
        enable_effects=True,            # 시각적 효과 활성화
        bg_color=(20, 20, 22),          # 배경색 (어두운 색상)
        pixelate_size=8                # 픽셀화 강도 (예: 8으로 설정해서 8x8 블록 픽셀화)
    )
    gen.run()
