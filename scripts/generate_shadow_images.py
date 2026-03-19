"""
Generate images with random shadow variations for data augmentation.

This script applies randomized shadow effects to a set of input images and
saves the augmented copies to an output directory. It is useful for creating
training/evaluation data that is robust to varying illumination conditions.

Usage
-----
    python scripts/generate_shadow_images.py <input_dir> [options]

Examples
--------
    # Apply 5 random shadow variants per image at default settings
    python scripts/generate_shadow_images.py /path/to/images

    # Apply 3 variants with a stronger shadow and soft edges
    python scripts/generate_shadow_images.py /path/to/images \
        --num-augmentations 3 --shadow-intensity 0.8 --blur-radius 25
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Shadow augmentation helpers
# ---------------------------------------------------------------------------

def _random_convex_polygon(width: int, height: int, num_vertices: int = 6) -> np.ndarray:
    """Return a random convex polygon as an (N, 2) int array of (x, y) pairs.

    The polygon vertices are spread across the full image canvas so that
    shadows can cover any region including edges.
    """
    # Sample random angles and sort them so vertices are ordered around the
    # centre (cx, cy), which produces a convex polygon.
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
    # Random radii — using separate x/y scales produces non-circular shapes
    rx = np.random.uniform(width * 0.2, width * 0.7)
    ry = np.random.uniform(height * 0.2, height * 0.7)
    cx = np.random.uniform(rx * 0.3, width - rx * 0.3)
    cy = np.random.uniform(ry * 0.3, height - ry * 0.3)
    xs = np.clip((cx + rx * np.cos(angles)).astype(int), 0, width - 1)
    ys = np.clip((cy + ry * np.sin(angles)).astype(int), 0, height - 1)
    return np.stack([xs, ys], axis=1)


def _polygon_mask(width: int, height: int, polygon: np.ndarray) -> np.ndarray:
    """Rasterise *polygon* into a boolean mask of shape (height, width)."""
    from PIL import ImageDraw
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    pts = [tuple(p) for p in polygon]
    draw.polygon(pts, fill=255)
    return np.array(mask_img, dtype=np.float32) / 255.0


def apply_random_shadows(
    image: Image.Image,
    num_shadows: int = 1,
    shadow_intensity: float = 0.5,
    blur_radius: int = 15,
    num_vertices: int = 6,
    seed: int | None = None,
) -> Image.Image:
    """Apply *num_shadows* random shadow patches to *image*.

    Parameters
    ----------
    image:
        Input PIL image (RGB or RGBA).
    num_shadows:
        How many independent shadow polygons to overlay.
    shadow_intensity:
        Darkness of the shadow in [0, 1].  0 = fully transparent (no effect);
        1 = completely black shadow.
    blur_radius:
        Gaussian blur radius applied to the shadow mask edges.  Set to 0 for
        hard shadows.
    num_vertices:
        Number of vertices used for each shadow polygon.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    PIL.Image.Image
        Augmented image as an RGB image.
    """
    if seed is not None:
        rng_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)

    image = image.convert("RGB")
    img_array = np.array(image, dtype=np.float32)
    height, width = img_array.shape[:2]

    combined_mask = np.zeros((height, width), dtype=np.float32)

    for _ in range(num_shadows):
        polygon = _random_convex_polygon(width, height, num_vertices)
        mask = _polygon_mask(width, height, polygon)

        if blur_radius > 0:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask = np.array(mask_img, dtype=np.float32) / 255.0

        # Accumulate masks; clamp so overlapping shadows don't exceed 1
        combined_mask = np.clip(combined_mask + mask, 0.0, 1.0)

    # Darken pixels that fall inside the shadow region
    shadow_factor = 1.0 - shadow_intensity * combined_mask[:, :, np.newaxis]
    augmented = np.clip(img_array * shadow_factor, 0, 255).astype(np.uint8)

    if seed is not None:
        random.setstate(rng_state)
        np.random.set_state(np_state)

    return Image.fromarray(augmented)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images with random shadow variations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        help="Directory that contains the input images (*.jpg / *.JPG / *.jpeg / *.JPEG / *.png / *.PNG).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where augmented images are saved.  "
            "Defaults to <input_dir>/shadow_augmented."
        ),
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=5,
        help="Number of distinct shadow variants to generate per image.",
    )
    parser.add_argument(
        "--num-shadows",
        type=int,
        default=1,
        help="Number of shadow polygons to overlay per augmented image.",
    )
    parser.add_argument(
        "--shadow-intensity",
        type=float,
        default=0.5,
        help="Shadow darkness in [0, 1]. 0 = no shadow, 1 = solid black.",
    )
    parser.add_argument(
        "--blur-radius",
        type=int,
        default=15,
        help="Gaussian blur radius for soft shadow edges. 0 = hard edges.",
    )
    parser.add_argument(
        "--num-vertices",
        type=int,
        default=6,
        help="Number of vertices used for each shadow polygon.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed for reproducible augmentations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(input_dir, "shadow_augmented")
    os.makedirs(output_dir, exist_ok=True)

    # Collect images
    patterns = [
        os.path.join(input_dir, "*.jpg"),
        os.path.join(input_dir, "*.JPG"),
        os.path.join(input_dir, "*.jpeg"),
        os.path.join(input_dir, "*.JPEG"),
        os.path.join(input_dir, "*.png"),
        os.path.join(input_dir, "*.PNG"),
    ]
    image_paths: list[str] = []
    for pattern in patterns:
        image_paths.extend(glob.glob(pattern))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in: {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s). "
          f"Generating {args.num_augmentations} shadow variant(s) each …")

    if args.seed is not None:
        # Seed both RNGs once before the loop.  apply_random_shadows() will
        # draw from this shared state for each call, so the full sequence of
        # augmentations is reproducible without needing per-call seeding.
        random.seed(args.seed)
        np.random.seed(args.seed)

    for image_path in image_paths:
        name_base, name_ext = os.path.splitext(os.path.basename(image_path))
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"  [skip] {image_path}: {exc}")
            continue

        for aug_idx in range(args.num_augmentations):
            augmented = apply_random_shadows(
                image,
                num_shadows=args.num_shadows,
                shadow_intensity=args.shadow_intensity,
                blur_radius=args.blur_radius,
                num_vertices=args.num_vertices,
            )
            out_filename = f"{name_base}_shadow_{aug_idx:03d}{name_ext}"
            out_path = os.path.join(output_dir, out_filename)
            augmented.save(out_path)

        print(f"  {name_base}{name_ext} → {args.num_augmentations} variant(s)")

    print(f"Done. Augmented images saved to: {output_dir}")


if __name__ == "__main__":
    main()
