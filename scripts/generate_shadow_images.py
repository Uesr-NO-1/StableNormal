"""
Generate random shadow augmentations on road surface images.

Three shadow generation methods are provided:
  1. polygon  - Random convex polygon shadows simulating cast shadows from objects.
  2. gradient - Smooth directional gradient shadows simulating low-angle sunlight.
  3. noise    - Naturalistic noise-pattern shadows using layered sinusoidal functions.

Usage:
    python scripts/generate_shadow_images.py <input_dir> [options]

Examples:
    # Apply all three methods, 3 augmented images per input, saved to <input_dir>/shadows/
    python scripts/generate_shadow_images.py ./data

    # Apply only the polygon method, 5 augmentations per image
    python scripts/generate_shadow_images.py ./data --method polygon --num-augmentations 5

    # Use a fixed random seed for reproducibility
    python scripts/generate_shadow_images.py ./data --seed 42
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Shadow generation methods
# ---------------------------------------------------------------------------

def _apply_shadow_mask(image_np: np.ndarray, mask: np.ndarray, darkness: float) -> np.ndarray:
    """Darken pixels of *image_np* where *mask* > 0.

    Args:
        image_np: H×W×3 uint8 array.
        mask:     H×W float array with values in [0, 1]; 1 = full shadow.
        darkness: Scalar in (0, 1] controlling how dark the shadow is.
                  A value of 1.0 produces fully black shadows.

    Returns:
        Augmented uint8 array of the same shape.
    """
    shadow_factor = 1.0 - darkness * mask  # 1 = no change, <1 = darker
    result = (image_np.astype(np.float32) * shadow_factor[:, :, np.newaxis]).clip(0, 255)
    return result.astype(np.uint8)


def add_polygon_shadow(
    image: Image.Image,
    rng: np.random.Generator,
    num_shadows: int = 1,
    darkness: float = 0.5,
    num_vertices: int = 6,
) -> Image.Image:
    """Add random convex polygon shadows to *image*.

    Each shadow is a filled convex polygon drawn at a random position and
    scale.  This simulates hard-edged shadows cast by trees, buildings or
    overhead structures.

    Args:
        image:        Input PIL image.
        rng:          Seeded numpy random Generator for reproducibility.
        num_shadows:  How many separate polygon shadows to overlay.
        darkness:     Shadow darkness in (0, 1].
        num_vertices: Number of vertices for each polygon (>=3).

    Returns:
        Augmented PIL image.
    """
    image_np = np.array(image.convert("RGB"))
    H, W = image_np.shape[:2]
    combined_mask = np.zeros((H, W), dtype=np.float32)

    for _ in range(num_shadows):
        # Random centre and scale
        cx = rng.uniform(0.1, 0.9) * W
        cy = rng.uniform(0.3, 0.9) * H  # biased toward lower portion (road area)
        scale_x = rng.uniform(0.05, 0.35) * W
        scale_y = rng.uniform(0.05, 0.25) * H

        # Generate convex polygon vertices by sorting random angles
        angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
        xs = (cx + scale_x * np.cos(angles)).astype(int).clip(0, W - 1)
        ys = (cy + scale_y * np.sin(angles)).astype(int).clip(0, H - 1)
        vertices = list(zip(xs.tolist(), ys.tolist()))

        # Draw filled polygon onto a temporary mask
        poly_img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(poly_img).polygon(vertices, fill=255)

        # Optionally soften the shadow edge
        blur_radius = max(1, int(min(scale_x, scale_y) * 0.08))
        poly_img = poly_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        poly_mask = np.array(poly_img, dtype=np.float32) / 255.0
        combined_mask = np.maximum(combined_mask, poly_mask)

    return Image.fromarray(_apply_shadow_mask(image_np, combined_mask, darkness))


def add_gradient_shadow(
    image: Image.Image,
    rng: np.random.Generator,
    num_shadows: int = 1,
    darkness: float = 0.5,
) -> Image.Image:
    """Add smooth directional gradient shadows to *image*.

    Each shadow is a linear gradient band sweeping across the image at a
    random angle.  This simulates soft shadows produced by low-angle sunlight
    partially blocked by distant objects or rolling terrain.

    Args:
        image:       Input PIL image.
        rng:         Seeded numpy random Generator.
        num_shadows: How many gradient bands to overlay.
        darkness:    Shadow darkness in (0, 1].

    Returns:
        Augmented PIL image.
    """
    image_np = np.array(image.convert("RGB"))
    H, W = image_np.shape[:2]
    combined_mask = np.zeros((H, W), dtype=np.float32)

    ys, xs = np.mgrid[0:H, 0:W]
    xs_n = xs / W  # normalised coordinates in [0, 1]
    ys_n = ys / H

    for _ in range(num_shadows):
        # Random direction angle for the gradient
        angle = rng.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Project pixel coordinates onto the gradient direction
        proj = cos_a * xs_n + sin_a * ys_n

        # Random band centre and width (expressed in the same projection space)
        proj_min, proj_max = float(proj.min()), float(proj.max())
        band_centre = rng.uniform(proj_min + 0.1 * (proj_max - proj_min),
                                  proj_max - 0.1 * (proj_max - proj_min))
        band_width = rng.uniform(0.05, 0.3) * (proj_max - proj_min)

        # Smooth bell-shaped mask centred on the band
        dist = np.abs(proj - band_centre)
        gradient_mask = np.clip(1.0 - dist / (band_width / 2.0), 0.0, 1.0)
        combined_mask = np.maximum(combined_mask, gradient_mask.astype(np.float32))

    return Image.fromarray(_apply_shadow_mask(image_np, combined_mask, darkness))


def add_noise_shadow(
    image: Image.Image,
    rng: np.random.Generator,
    num_shadows: int = 1,
    darkness: float = 0.5,
    frequency: float = 4.0,
    num_octaves: int = 4,
) -> Image.Image:
    """Add naturalistic noise-pattern shadows to *image*.

    Shadows are created by summing layered sinusoidal waves (pseudo-Perlin
    noise) at random phases and orientations.  This mimics the dappled light
    patterns produced by foliage overhead.

    Args:
        image:       Input PIL image.
        rng:         Seeded numpy random Generator.
        num_shadows: How many independent noise layers to overlay.
        darkness:    Shadow darkness in (0, 1].
        frequency:   Base spatial frequency of the noise pattern.
        num_octaves: Number of frequency octaves to layer.

    Returns:
        Augmented PIL image.
    """
    image_np = np.array(image.convert("RGB"))
    H, W = image_np.shape[:2]
    combined_mask = np.zeros((H, W), dtype=np.float32)

    ys, xs = np.mgrid[0:H, 0:W]
    xs_n = xs / W
    ys_n = ys / H

    for _ in range(num_shadows):
        noise = np.zeros((H, W), dtype=np.float32)
        amplitude = 1.0
        total_amplitude = 0.0

        for octave in range(num_octaves):
            freq = frequency * (2 ** octave)
            phase_x = rng.uniform(0, 2 * np.pi)
            phase_y = rng.uniform(0, 2 * np.pi)
            angle = rng.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            rotated = cos_a * xs_n + sin_a * ys_n
            noise += amplitude * (np.sin(2 * np.pi * freq * rotated + phase_x) + 1.0) / 2.0
            total_amplitude += amplitude
            amplitude *= 0.5

        noise /= total_amplitude  # normalise to [0, 1]

        # Apply a threshold to create distinct shadow regions
        threshold = rng.uniform(0.4, 0.65)
        shadow_mask = np.where(noise > threshold, noise - threshold, 0.0)
        if shadow_mask.max() > 0:
            shadow_mask /= shadow_mask.max()

        combined_mask = np.maximum(combined_mask, shadow_mask.astype(np.float32))

    return Image.fromarray(_apply_shadow_mask(image_np, combined_mask, darkness))


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

SHADOW_METHODS = {
    "polygon": add_polygon_shadow,
    "gradient": add_gradient_shadow,
    "noise": add_noise_shadow,
}


# ---------------------------------------------------------------------------
# Core augmentation entry point
# ---------------------------------------------------------------------------

def generate_shadow_augmentations(
    image: Image.Image,
    rng: np.random.Generator,
    methods: list[str],
    num_augmentations: int,
    darkness: float,
    num_shadows: int,
) -> list[tuple[str, Image.Image]]:
    """Generate *num_augmentations* augmented copies of *image*.

    Args:
        image:             Input PIL image.
        rng:               Seeded numpy random Generator.
        methods:           List of method names to cycle through.
        num_augmentations: Total number of augmented images to produce.
        darkness:          Shadow darkness in (0, 1].
        num_shadows:       Number of individual shadow shapes per augmentation.

    Returns:
        List of ``(method_name, augmented_image)`` tuples.
    """
    results: list[tuple[str, Image.Image]] = []
    for i in range(num_augmentations):
        method_name = methods[i % len(methods)]
        fn = SHADOW_METHODS[method_name]
        augmented = fn(image, rng, num_shadows=num_shadows, darkness=darkness)
        results.append((method_name, augmented))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing input images (looked up inside an 'images' sub-folder).",
    )
    parser.add_argument(
        "--method",
        choices=list(SHADOW_METHODS) + ["all"],
        default="all",
        help="Shadow generation method.  Use 'all' to cycle through all methods (default: all).",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=3,
        metavar="N",
        help="Number of augmented images to produce per input image (default: 3).",
    )
    parser.add_argument(
        "--darkness",
        type=float,
        default=0.5,
        metavar="D",
        help="Shadow darkness in the range (0, 1].  1.0 = fully black shadows (default: 0.5).",
    )
    parser.add_argument(
        "--num-shadows",
        type=int,
        default=2,
        metavar="S",
        help="Number of individual shadow shapes to overlay per augmentation (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducibility.  If omitted, results are non-deterministic.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not (0 < args.darkness <= 1.0):
        print("Error: --darkness must be in the range (0, 1].", file=sys.stderr)
        sys.exit(1)
    if args.num_augmentations < 1:
        print("Error: --num-augmentations must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.num_shadows < 1:
        print("Error: --num-shadows must be >= 1.", file=sys.stderr)
        sys.exit(1)

    methods = list(SHADOW_METHODS) if args.method == "all" else [args.method]

    rng = np.random.default_rng(args.seed)

    # Collect input images
    input_dir = args.input_dir
    image_patterns = [
        os.path.join(input_dir, "images", "*.jpg"),
        os.path.join(input_dir, "images", "*.JPG"),
        os.path.join(input_dir, "images", "*.png"),
    ]
    image_paths: list[str] = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern))

    if not image_paths:
        print(
            f"No images found under '{os.path.join(input_dir, 'images')}'. "
            "Expected .jpg, .JPG or .png files.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = os.path.join(input_dir, "shadows")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(image_paths)} image(s). Generating augmentations → {output_dir}")

    for image_path in image_paths:
        name_base = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path).convert("RGB")

        augmented_pairs = generate_shadow_augmentations(
            image,
            rng,
            methods=methods,
            num_augmentations=args.num_augmentations,
            darkness=args.darkness,
            num_shadows=args.num_shadows,
        )

        for idx, (method_name, aug_image) in enumerate(augmented_pairs):
            out_name = f"{name_base}_shadow_{method_name}_{idx:02d}.png"
            out_path = os.path.join(output_dir, out_name)
            aug_image.save(out_path)

        print(f"  {name_base}: saved {len(augmented_pairs)} augmentation(s)")

    print("Done.")


if __name__ == "__main__":
    main()
