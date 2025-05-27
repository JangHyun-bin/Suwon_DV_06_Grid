import random
from typing import List, Tuple
from PIL import Image, ImageDraw

class LineConnector:
    def __init__(
        self,
        color: Tuple[int, int, int] = (255, 255, 255),
        width: int = 2,
        highlight_color: Tuple[int, int, int] = (255, 50, 50),
        grid_color: Tuple[int, int, int] = (180, 180, 180),
        show_grid: bool = True,
        grid_spacing: int = 64,
        grid_opacity: float = 0.15
    ):
        self.color = color
        self.width = width
        self.highlight_color = highlight_color
        self.grid_color = grid_color
        self.show_grid = show_grid
        self.grid_spacing = grid_spacing
        self.grid_opacity = grid_opacity
        self.highlight_points: List[Tuple[int, int]] = []

    def set_random_highlights(self, centers: List[Tuple[int, int]], count: int = 2):
        if centers and count > 0:
            self.highlight_points = random.sample(centers, min(count, len(centers)))
        else:
            self.highlight_points = []

    def draw_grid(self, image: Image.Image):
        if not self.show_grid:
            return
        draw = ImageDraw.Draw(image, 'RGBA')
        w, h = image.size
        alpha = int(255 * self.grid_opacity)
        for x in range(0, w, self.grid_spacing):
            draw.line([(x, 0), (x, h)], fill=self.grid_color + (alpha,), width=1)
        for y in range(0, h, self.grid_spacing):
            draw.line([(0, y), (w, y)], fill=self.grid_color + (alpha,), width=1)

    def draw(self, image: Image.Image, centers: List[Tuple[int, int]]):
        if len(centers) < 2:
            return
        self.draw_grid(image)
        if not self.highlight_points:
            self.set_random_highlights(centers)
        draw = ImageDraw.Draw(image)
        for i in range(len(centers) - 1):
            p1, p2 = centers[i], centers[i + 1]
            is_high = p1 in self.highlight_points or p2 in self.highlight_points
            line_color = self.highlight_color if is_high else self.color
            line_width = self.width + 1 if is_high else self.width
            # fixed midpoint for consistent line segments
            mid = (p2[0], p1[1])
            draw.line([p1, mid], fill=line_color, width=line_width)
            draw.line([mid, p2], fill=line_color, width=line_width)
        for x, y in centers:
            is_high = (x, y) in self.highlight_points
            r = self.width*2 if is_high else self.width
            fill = self.highlight_color if is_high else self.color
            draw.ellipse([x-r, y-r, x+r, y+r], fill=fill)
            if is_high:
                outer = r*2
                draw.ellipse([x-outer, y-outer, x+outer, y+outer], outline=fill, width=2) 