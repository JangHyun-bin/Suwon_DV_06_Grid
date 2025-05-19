import os
from generator import main

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
