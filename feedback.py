import torch
from typing import Tuple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TorchFeedback:
    def __init__(self, size: Tuple[int, int], enable_effects: bool = True):
        self.canvas_size = size
        self.enable_effects = enable_effects
        self.frame_count = 0
        self.vignette_strength = 0.15
        self.noise_amount = 0.0
        h, w = size
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w)
        )
        self.vignette = (x.pow(2) + y.pow(2)).sqrt().pow(0.7)
        self.vignette = self.vignette.clamp(0, 1).view(1, 1, h, w)
        self.feedback_decay = 0.95
        self.accumulator = torch.zeros(1, 3, h, w, device=DEVICE)

    def apply(self, canvas: torch.Tensor) -> torch.Tensor:
        self.frame_count += 1
        if not self.enable_effects:
            return canvas
        out = canvas.clone()
        vignette_mask = (1.0 - self.vignette * self.vignette_strength).to(out.device)
        out = out * vignette_mask
        if self.noise_amount > 0:
            noise = torch.randn_like(out) * self.noise_amount
            out = (out + noise).clamp(0, 1)
        
        feedback_strength = 0.05
        self.accumulator = self.accumulator * self.feedback_decay + out * feedback_strength
        
        return out * 0.95 + self.accumulator * 0.05 