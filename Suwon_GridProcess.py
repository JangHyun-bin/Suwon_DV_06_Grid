import os
import sys
import glob
import random
import numpy as np
from datetime import datetime
from PIL import Image, ImageFile

from generator import main
from open3d_mesh_baker import bake_meshes_from_images
from open3d_mesh_viewer import view_mesh_sequence

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ─── Create new grid_images folder ─────────────────────────────
    images_folder = os.path.join(script_dir, f"grid_images_{timestamp}")
    meshes_folder = os.path.join(script_dir, f"grid_meshes_{timestamp}")

    # ─── Step 1: Generate 2D Grid Visualization ─────────────────────────
    print("\n=== Step 1: Generating 2D Grid Visualization ===")
    main(
        image_folder=os.path.join(script_dir, "suwon_budget_images"),
        output_folder=images_folder,
        canvas_size=(2048, 2048),
        num_frames=500,
        interval=16,
        per_spawn=250,
        fade_range=(0.03, 0.12),
        visual_style="modern_grid",
        enable_effects=True,
        bg_color=(20, 20, 22),
        pixelate_size=None,
        use_taichi=False,
        use_extrude3d=False,
        extrude_scale=1.0
    )

    print("\n=== 2D Image Generation Complete! ===")
    print(f"Generated images are saved in: {images_folder}")
    

    # ─── Step 3: Generate heightmap image from keyword frequencies CSV ───
    csv_images_folder = os.path.join(script_dir, f"csv_images_{timestamp}")
    os.makedirs(csv_images_folder, exist_ok=True)
    img_array = (norm * 255).astype(np.uint8)
    heightmap = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            y0, y1 = i * cell_h, (i+1) * cell_h
            x0, x1 = j * cell_w, (j+1) * cell_w
            heightmap[y0:y1, x0:x1] = img_array[i, j]
    heightmap_img = Image.fromarray(heightmap).convert("RGB")
    draw = ImageDraw.Draw(heightmap_img)
    base_font_size = int(min(cell_w, cell_h) * 0.2)
    for idx, (kw, freq) in enumerate(zip(df_all.head(30)["keyword"], top_freqs)):
        i, j = divmod(idx, cols)
        norm_freq = (freq - freq_array.min()) / (freq_array.max() - freq_array.min())
        fs = base_font_size + int(norm_freq * base_font_size * 1.5)
        font = ImageFont.truetype(font_path, fs)
        bbox = draw.textbbox((0,0), kw, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x = j * cell_w + (cell_w - tw)//2
        y = i * cell_h + (cell_h - th)//2
        pv = heightmap[i*cell_h + cell_h//2, j*cell_w + cell_w//2]
        color = (255,255,255) if pv < 128 else (0,0,0)
        draw.text((x, y), kw, font=font, fill=color)
    heightmap_img.save(os.path.join(csv_images_folder, "heightmap.png"))

    # ─── Step 4: Bake 3D meshes from grid visualization images ─────────────
    print(f"\n=== Step 4: Baking 3D Meshes from Grid Visualization ===")
    mesh_count_grid = bake_meshes_from_images(
        image_folder=images_folder,
        output_folder=meshes_folder,
        mesh_resolution=(2048, 2048),
        heightfield_scale=0.4,
        grid_size=2.0,
        color_map='terrain',
        blur_radius=2
    )

    # ─── Step 5: Bake 3D meshes from CSV-driven heightmap ────────────────
    csv_meshes_folder = os.path.join(script_dir, f"csv_meshes_{timestamp}")
    print(f"\n=== Step 5: Baking 3D Meshes from CSV Heightmap ===")
    mesh_count_csv = bake_meshes_from_images(
        image_folder=csv_images_folder,
        output_folder=csv_meshes_folder,
        mesh_resolution=(2048, 2048),
        heightfield_scale=0.4,
        grid_size=2.0,
        color_map='terrain',
        blur_radius=2
    )

    # ─── Step 6: View the baked meshes ────────────────────────────────────
    if mesh_count_grid > 0:
        print(f"\n=== Step 6a: Launching 3D Mesh Viewer for Grid Visualization ===")
        view_mesh_sequence(
            mesh_folder=meshes_folder,
            grid_size=2.0,
            show_grid=True,
            show_axes=True
        )
    else:
        print("No grid meshes were created, cannot launch viewer.")

    if mesh_count_csv > 0:
        print(f"\n=== Step 6b: Launching 3D Mesh Viewer for CSV Heightmap ===")
        view_mesh_sequence(
            mesh_folder=csv_meshes_folder,
            grid_size=2.0,
            show_grid=True,
            show_axes=True
        )
    else:
        print("No CSV-driven meshes were created, cannot launch viewer.")