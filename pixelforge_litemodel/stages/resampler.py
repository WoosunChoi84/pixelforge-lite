"""
PixelForge Lite — Stage 2: Resampler

Direction-aware pixel-preserving resize.

  - up       → nearest-neighbor (crisp integer scaling)
  - down     → CIELAB + k=2 k-means mode resampling with C6 bg-aware gate
  - identity → copy / minor INTER_NEAREST adjust

The down path implements the Non-representative Interpolation Principle:
AA transition pixels are structurally between bg and feature-core; a simple
frequency vote on non-bg pixels often selects a transition color instead
of the core. k-means in CIELAB separates feature-core from transition
clusters, and the cluster farthest from bg (by centroid distance) with
sufficient size is chosen. Within the winning cluster the most-frequent
observed color is emitted, preserving the "no new colors" property
required by the palette-sufficient fast path downstream.

The bg-aware path activates only when:
  (dominant_color_ratio >= 0.25) AND (edge_dom_ratio >= 0.7)

This "C6" gate (REPORT_V5) rejects images with equi-populated colors or
with the dominant color spread through the interior rather than the
perimeter — in such cases no meaningful bg exists and the fallback plain
mode vote preserves all features.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..models import InputProfile, PipelineConfig, StageResult
from ..utils import pack_rgb, rgb_to_lab, resample_alpha_binary


MIN_CLUSTER_SIZE: int = 2


def process(result: StageResult, config: PipelineConfig) -> StageResult:
    if result.image is None:
        raise ValueError("Input image is None")

    orig_h, orig_w = result.image.shape[:2]
    grid_w, grid_h = config.grid_size

    profile: InputProfile | None = result.metadata.get("input_profile")
    if profile is not None:
        direction = profile.scaling_direction
    else:
        orig_pixels = orig_h * orig_w
        grid_pixels = grid_h * grid_w
        if grid_pixels < orig_pixels * 0.95:
            direction = "down"
        elif grid_pixels > orig_pixels * 1.05:
            direction = "up"
        else:
            direction = "identity"

    if direction == "up":
        method = "nearest"
        resampled = _nearest_resize(result.image, config.grid_size)
    elif direction == "down":
        method = "mode"
        resampled = _mode_resample(result.image, config.grid_size, profile=profile)
    else:
        method = "identity"
        resampled = _identity_resize(result.image, config.grid_size)

    assert resampled.shape == (grid_h, grid_w, 3), (
        f"Image shape mismatch: {resampled.shape} (expected {(grid_h, grid_w, 3)})"
    )
    assert resampled.dtype == np.uint8

    # Parallel alpha resampling — binary mask, same spatial method family
    resampled_alpha: np.ndarray | None = None
    if result.alpha is not None:
        resampled_alpha = resample_alpha_binary(
            result.alpha, config.grid_size, direction
        )
        assert resampled_alpha.shape == (grid_h, grid_w)
        assert resampled_alpha.dtype == np.uint8

    metadata = result.metadata.copy()
    metadata["resampler"] = {
        "original_shape": (orig_h, orig_w),
        "grid_size": config.grid_size,
        "output_shape": resampled.shape,
        "direction": direction,
        "method": method,
        "has_alpha": resampled_alpha is not None,
    }

    return StageResult(
        image=resampled,
        palette=result.palette,
        metadata=metadata,
        alpha=resampled_alpha,
    )


def _nearest_resize(image: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    grid_w, grid_h = grid_size
    return cv2.resize(image, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)


def _identity_resize(image: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    grid_w, grid_h = grid_size
    h, w = image.shape[:2]
    if h == grid_h and w == grid_w:
        return image.copy()
    return cv2.resize(image, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)


def _compute_edge_dom_ratio(
    image: np.ndarray, dominant_rgb: tuple[int, int, int]
) -> float:
    """
    Fraction of the outer perimeter occupied by the dominant color.

    DataFlow: _compute_edge_dom_ratio(image, dominant_rgb)
    IN:  image (H, W, 3) uint8; dominant_rgb (r, g, b) from input analysis
    CHAIN: choose ring width (2px default, 1px for small images)
           → build perimeter mask → intersect with dominant-color mask
           → ratio = dom∩edge / edge [Spatial Statistic]
    OUT: float in [0, 1]

    Rationale: a "true" background surrounds the content. High edge-dominance
    (≥0.7) indicates a meaningful bg; low values indicate the dominant color
    is spread through the interior (e.g., a striped pattern), which means
    bg-vs-feature partitioning is ill-defined.

    T2.5 guard: for images smaller than 5×5 the default 2-pixel ring would
    cover the entire array, making the test degenerate. We fall back to a
    1-pixel ring for images ≥3, and return 0.0 for anything smaller (no
    meaningful spatial distinction possible — gate will be OFF).
    """
    h, w = image.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    ring = 2 if (h >= 5 and w >= 5) else 1

    dr, dg, db = dominant_rgb
    dom_mask = np.all(image == np.array([dr, dg, db], dtype=np.uint8), axis=2)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:ring, :] = True
    edge_mask[-ring:, :] = True
    edge_mask[:, :ring] = True
    edge_mask[:, -ring:] = True
    edge_count = int(np.sum(edge_mask))
    dom_in_edge = int(np.sum(dom_mask & edge_mask))
    return dom_in_edge / edge_count if edge_count > 0 else 0.0


def _kmeans_lab(
    points_lab: np.ndarray, k: int = 2, max_iters: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic k-means in CIELAB space with L-axis-quantile initialization.

    DataFlow: _kmeans_lab(points_lab, k, max_iters)
    IN:  points_lab (n, 3) float — CIELAB coords of pixels to cluster
         k int — number of clusters
         max_iters int — iteration cap (early exit on convergence)
    CHAIN: sort by L → pick k evenly-spaced quantile init centers
           → iterate: assign by L2 distance → update centers as cluster means
           → break on unchanged assignments or centroid stability [k-means]
    OUT: (centroids (k, 3), assignments (n,))

    Deterministic init avoids k-means non-determinism that would make
    block-by-block output order-dependent.
    """
    n = len(points_lab)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64)
    if n < k:
        centroids = np.zeros((k, 3), dtype=np.float32)
        centroids[:n] = points_lab
        centroids[n:] = points_lab[-1]
        assignments = np.arange(n, dtype=np.int64)
        return centroids, assignments

    order = np.argsort(points_lab[:, 0])
    init_idx = np.linspace(0, n - 1, k, dtype=int)
    centroids = points_lab[order[init_idx]].astype(np.float32).copy()

    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(max_iters):
        dists = np.sum((points_lab[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_assignments = np.argmin(dists, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        new_centroids = centroids.copy()
        for i in range(k):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = points_lab[mask].mean(axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-3):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids, assignments


def _mode_resample(
    image: np.ndarray,
    grid_size: tuple[int, int],
    profile: "InputProfile | None" = None,
) -> np.ndarray:
    """
    Background-aware mode downsampling with CIELAB k=2 k-means refinement.

    DataFlow: _mode_resample(image, grid_size, profile)
    IN:  image (H, W, 3) uint8; grid_size (w, h); profile (optional)
    CHAIN: resolve bg color + dom ratio (from profile or recompute)
           → gate = (dom_ratio >= 0.25) AND (edge_dom_ratio >= 0.7) [C6]
           → per block:
               compute mode via packed-uint32 unique+counts
               if gate AND mode == bg AND block has non-bg:
                 non-bg pixels → rgb_to_lab → _kmeans_lab(k=2)
                 filter clusters with size >= MIN_CLUSTER_SIZE
                 winner = cluster farthest from bg_lab
                 mode_packed = most-frequent observed color within winner
           → emit mode_packed as output cell color
    OUT: (grid_h, grid_w, 3) uint8

    The "most-frequent observed color in winner cluster" rule preserves the
    invariant that output colors are a subset of input colors, keeping the
    palette_sufficient fast path in the quantizer stage intact for
    pre-quantized inputs.
    """
    orig_h, orig_w = image.shape[:2]
    grid_w, grid_h = grid_size

    if profile is not None and profile.dominant_color is not None:
        r, g, b = profile.dominant_color
        bg_packed = np.uint32(r) * 65536 + np.uint32(g) * 256 + np.uint32(b)
        bg_ratio = profile.dominant_color_ratio
    else:
        all_pixels = image.reshape(-1, 3)
        all_packed = pack_rgb(all_pixels)
        global_values, global_counts = np.unique(all_packed, return_counts=True)
        bg_packed = global_values[np.argmax(global_counts)]
        bg_ratio = global_counts.max() / len(all_packed)
        r = int((int(bg_packed) >> 16) & 0xFF)
        g = int((int(bg_packed) >> 8) & 0xFF)
        b = int(int(bg_packed) & 0xFF)

    # C6 gate: (dom ratio absolute threshold) AND (spatial edge dominance)
    edge_dom_ratio = _compute_edge_dom_ratio(image, (r, g, b))
    use_bg_aware = (bg_ratio >= 0.25) and (edge_dom_ratio >= 0.7)

    bg_rgb = np.array([r, g, b], dtype=np.float32)
    bg_lab = rgb_to_lab(bg_rgb.reshape(1, 3))[0]

    row_b = np.clip(np.round(np.linspace(0, orig_h, grid_h + 1)).astype(int), 0, orig_h)
    col_b = np.clip(np.round(np.linspace(0, orig_w, grid_w + 1)).astype(int), 0, orig_w)

    block_heights = np.diff(row_b)
    block_widths = np.diff(col_b)
    uniform = (
        block_heights.size > 0
        and block_widths.size > 0
        and bool(np.all(block_heights == block_heights[0]))
        and bool(np.all(block_widths == block_widths[0]))
    )

    if (
        uniform
        and int(block_heights[0]) * grid_h == orig_h
        and int(block_widths[0]) * grid_w == orig_w
        and int(block_heights[0]) > 0
        and int(block_widths[0]) > 0
    ):
        return _mode_resample_uniform(
            image, grid_h, grid_w,
            int(block_heights[0]), int(block_widths[0]),
            bg_packed, use_bg_aware, bg_lab,
        )

    # Non-uniform tiling fallback: scalar per-block loop.
    output = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i in range(grid_h):
        for j in range(grid_w):
            block = image[row_b[i] : row_b[i + 1], col_b[j] : col_b[j + 1]]
            if block.size == 0:
                continue
            pixels = block.reshape(-1, 3)
            packed = pack_rgb(pixels)
            values, counts = np.unique(packed, return_counts=True)
            mode_packed = values[np.argmax(counts)]

            if use_bg_aware and mode_packed == bg_packed and len(values) > 1:
                non_bg_pixel_mask = packed != bg_packed
                non_bg_pixels = pixels[non_bg_pixel_mask]

                if len(non_bg_pixels) >= 2:
                    non_bg_lab = rgb_to_lab(non_bg_pixels.astype(np.float32))
                    centroids, assignments = _kmeans_lab(non_bg_lab, k=2)

                    cluster_sizes = np.array(
                        [int((assignments == ci).sum()) for ci in range(2)]
                    )
                    valid = cluster_sizes >= MIN_CLUSTER_SIZE

                    if valid.any():
                        cent_dists = np.sum((centroids - bg_lab) ** 2, axis=1)
                        cent_dists_masked = np.where(valid, cent_dists, -1.0)
                        winner_ci = int(np.argmax(cent_dists_masked))

                        winner_pixels = non_bg_pixels[assignments == winner_ci]
                        w_packed = pack_rgb(winner_pixels)
                        w_vals, w_counts = np.unique(w_packed, return_counts=True)
                        mode_packed = w_vals[np.argmax(w_counts)]

            output[i, j] = [
                (int(mode_packed) >> 16) & 0xFF,
                (int(mode_packed) >> 8) & 0xFF,
                int(mode_packed) & 0xFF,
            ]

    return output


def _mode_resample_uniform(
    image: np.ndarray,
    grid_h: int,
    grid_w: int,
    bh: int,
    bw: int,
    bg_packed: np.uint32,
    use_bg_aware: bool,
    bg_lab: np.ndarray,
) -> np.ndarray:
    """
    Fast path for the common uniform-tiling case (orig_h % grid_h == 0, etc.).

    DataFlow: _mode_resample_uniform(image, grid_h, grid_w, bh, bw, bg_packed,
                                      use_bg_aware, bg_lab)
    IN:  image (H, W, 3) uint8; grid dims; block dims; precomputed bg info
    CHAIN: pack_rgb whole image → reshape/transpose to (grid_h, grid_w, bh*bw)
           → per-row sort + run-length scan for batch mode [Sorted-Run Mode]
           → if bg gate fires, loop only blocks whose mode==bg AND has_multi,
             running k-means on their non-bg pixels [Filtered k-means]
           → unpack uint32 mode back to (r, g, b) uint8 slices
    OUT: (grid_h, grid_w, 3) uint8 — byte-identical to the scalar loop

    The "sort → run lengths → argmax of run length" trick replicates
    `np.unique(return_counts=True)` tie-breaking: both return the smallest
    packed value among counts ties, because unique values sort ascending and
    argmax selects the first occurrence of the max.
    """
    orig_h, orig_w = image.shape[:2]

    # (H, W) uint32 packed; then block-tile.
    packed_full = pack_rgb(image.reshape(-1, 3)).reshape(orig_h, orig_w)
    tiled = packed_full.reshape(grid_h, bh, grid_w, bw).transpose(0, 2, 1, 3)
    tiled = np.ascontiguousarray(tiled).reshape(grid_h, grid_w, bh * bw)

    K = bh * bw
    G = grid_h * grid_w
    flat = tiled.reshape(G, K)

    if K == 1:
        mode_packed_arr = flat[:, 0].reshape(grid_h, grid_w).copy()
        has_multi = np.zeros((grid_h, grid_w), dtype=bool)
    else:
        sorted_flat = np.sort(flat, axis=1)
        # Because the row is sorted ascending, first != last iff >1 unique.
        has_multi_flat = sorted_flat[:, 0] != sorted_flat[:, -1]

        # Run-length encoding per row via cumulative fills.
        run_start_mark = np.empty((G, K), dtype=bool)
        run_start_mark[:, 0] = True
        run_start_mark[:, 1:] = sorted_flat[:, 1:] != sorted_flat[:, :-1]

        run_end_mark = np.empty((G, K), dtype=bool)
        run_end_mark[:, -1] = True
        run_end_mark[:, :-1] = sorted_flat[:, :-1] != sorted_flat[:, 1:]

        idx = np.arange(K)
        masked_start = np.where(run_start_mark, idx[None, :], -1)
        run_start = np.maximum.accumulate(masked_start, axis=1)
        masked_end = np.where(run_end_mark, idx[None, :], K)
        run_end = np.minimum.accumulate(masked_end[:, ::-1], axis=1)[:, ::-1]

        counts_per_pos = run_end - run_start + 1
        mode_pos = np.argmax(counts_per_pos, axis=1)
        mode_vals = sorted_flat[np.arange(G), mode_pos]

        mode_packed_arr = mode_vals.reshape(grid_h, grid_w)
        has_multi = has_multi_flat.reshape(grid_h, grid_w)

    if use_bg_aware:
        # Only blocks where mode == bg AND the block has >1 unique colours
        # need the k-means refinement (len(values) > 1 in the reference).
        cand_mask = (mode_packed_arr == bg_packed) & has_multi
        cand_idx = np.argwhere(cand_mask)
        for k in range(len(cand_idx)):
            i = int(cand_idx[k, 0])
            j = int(cand_idx[k, 1])
            block_packed = tiled[i, j]
            non_bg_mask = block_packed != bg_packed
            non_bg_packed = block_packed[non_bg_mask]
            if len(non_bg_packed) < 2:
                continue
            non_bg_pixels = np.empty((len(non_bg_packed), 3), dtype=np.uint8)
            non_bg_pixels[:, 0] = (non_bg_packed >> 16) & 0xFF
            non_bg_pixels[:, 1] = (non_bg_packed >> 8) & 0xFF
            non_bg_pixels[:, 2] = non_bg_packed & 0xFF

            non_bg_lab = rgb_to_lab(non_bg_pixels.astype(np.float32))
            centroids, assignments = _kmeans_lab(non_bg_lab, k=2)
            cluster_sizes = np.array(
                [int((assignments == ci).sum()) for ci in range(2)]
            )
            valid = cluster_sizes >= MIN_CLUSTER_SIZE
            if valid.any():
                cent_dists = np.sum((centroids - bg_lab) ** 2, axis=1)
                cent_dists_masked = np.where(valid, cent_dists, -1.0)
                winner_ci = int(np.argmax(cent_dists_masked))
                winner_packed = non_bg_packed[assignments == winner_ci]
                w_vals, w_counts = np.unique(winner_packed, return_counts=True)
                mode_packed_arr[i, j] = w_vals[np.argmax(w_counts)]

    output = np.empty((grid_h, grid_w, 3), dtype=np.uint8)
    mp = mode_packed_arr.astype(np.uint32)
    output[:, :, 0] = ((mp >> 16) & 0xFF).astype(np.uint8)
    output[:, :, 1] = ((mp >> 8) & 0xFF).astype(np.uint8)
    output[:, :, 2] = (mp & 0xFF).astype(np.uint8)
    return output
