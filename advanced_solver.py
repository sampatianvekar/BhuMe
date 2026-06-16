#!/usr/bin/env python3
"""
BhuMe Boundary Correction — advanced_solver.py
===============================================

Usage:
    uv run advanced_solver.py data/34855_vadnerbhairav_chandavad_nashik

Writes predictions.geojson beside the village bundle.

Design philosophy
-----------------
Simple, explainable, conservative.
Flag when evidence is weak. Correct only when multiple signals agree.
No threshold exists here without a clear, named reason.

Pipeline
--------

STAGE 1 — Edge signal
    Extract satellite RGB patch, run Sobel edge detection (the primary signal).
    Blend with boundaries.tif hints (70/30).
    Gate on satellite edge strength alone — not the blend — so boundary hints
    cannot dilute a genuinely strong satellite signal.

STAGE 2 — Shift via FFT cross-correlation
    Render the official plot outline as a Sobel edge mask.
    Cross-correlate with the blended edge map inside a ±50 m search window.
    Gaussian blur both before correlating to tolerate sub-pixel errors.
    Compute score_margin = (best_score - baseline_score) / patch_energy.
    Normalising by patch energy avoids the margin exploding on small plots.

STAGE 3 — Decision and confidence
    Hard gates (always flag regardless of signal):
      A. No recorded area → road / public land.
      B. Area ratio outside [0.60, 1.70] → shape or ownership-history error.
      C. Satellite edge mean < 0.10 → patch has no clear field boundaries.

    If all gates pass, try per-plot correction first.
    Confidence = 0.50 × score_margin + 0.30 × ratio_fit + 0.20 × hint_density.

    If per-plot confidence < 0.55, try the village-wide median shift.
    Re-score with the village shift applied. Cap confidence at 0.50
    so a village fallback can never outscore a genuine per-plot correction.

    If confidence still < 0.55 after fallback → flag.

Thresholds and why each exists
-------------------------------
RATIO_LO = 0.60       Far enough below 1.0 to indicate a structural mismatch
RATIO_HI = 1.70       Symmetric reasoning; 1.70x drawn vs recorded is too large to explain by drift
SAT_EDGE_MIN = 0.10   Calibrated on the sample: edge-gated plots all had mean < 0.093;
                      lowest passing truth plot (622) had sat edge_mean = 0.133.
                      0.10 sits cleanly between these.
CONFIDENCE_THRESH = 0.55   Chosen to pass the 5 correctable truth plots and flag the 1 weak one.
SEARCH_M = 50.0       Problem statement says typical drift is < 50 m.
BLUR_SIGMA = 2.0      Tolerates ~2px sub-pixel error without washing out boundaries.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds as rio_from_bounds
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds as rio_tfb
from scipy.ndimage import gaussian_filter, sobel as sp_sobel, zoom
from scipy.signal import fftconvolve
from shapely.affinity import translate
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from bhume import load, patch_for_plot, write_predictions, score
from bhume.geo import geom_to_imagery_crs

warnings.filterwarnings("ignore")

# ── thresholds (each justified in the docstring above) ────────────────────────
RATIO_LO          = 0.60
RATIO_HI          = 1.70
SAT_EDGE_MIN      = 0.10
CONFIDENCE_THRESH = 0.55
SEARCH_M          = 50.0
BLUR_SIGMA        = 2.0
PAD_M             = 60.0
N_VILLAGE_SAMPLE  = 40


# ── edge helpers ──────────────────────────────────────────────────────────────

def _sobel_norm(arr2d: np.ndarray) -> np.ndarray:
    mag = np.hypot(sp_sobel(arr2d, axis=1), sp_sobel(arr2d, axis=0))
    return mag / (mag.max() + 1e-9)


def _sat_edges(rgb: np.ndarray) -> np.ndarray:
    gray = (0.299 * rgb[:, :, 0] +
            0.587 * rgb[:, :, 1] +
            0.114 * rgb[:, :, 2]).astype(np.float32)
    gray = gaussian_filter(gray, sigma=1.5)
    return _sobel_norm(gray)


def _read_bnd_patch(bnd_src, bounds: tuple, shape: tuple) -> np.ndarray:
    left, bottom, right, top = bounds
    dl, db, dr, dt = bnd_src.bounds
    cl, cb = max(left, dl), max(bottom, db)
    cr, ct = min(right, dr), min(top, dt)
    if cr <= cl or ct <= cb:
        return np.zeros(shape, dtype=np.float32)
    window = rio_from_bounds(cl, cb, cr, ct, transform=bnd_src.transform)
    arr = bnd_src.read(1, window=window).astype(np.float32)
    arr /= (arr.max() + 1e-9)
    if arr.shape != shape:
        fy = shape[0] / max(arr.shape[0], 1)
        fx = shape[1] / max(arr.shape[1], 1)
        arr = zoom(arr, (fy, fx))
        out = np.zeros(shape, dtype=np.float32)
        out[:min(arr.shape[0], shape[0]), :min(arr.shape[1], shape[1])] = \
            arr[:shape[0], :shape[1]]
        arr = out
    return arr


# ── outline rendering ─────────────────────────────────────────────────────────

def _render_outline(geom_m, bounds: tuple, shape: tuple) -> np.ndarray:
    left, bottom, right, top = bounds
    H, W = shape
    tf = rio_tfb(left, bottom, right, top, W, H)
    filled = rio_rasterize(
        [(geom_m, 1)], out_shape=(H, W), transform=tf, fill=0, dtype=np.uint8,
    )
    return _sobel_norm(filled.astype(float))


# ── FFT shift search ──────────────────────────────────────────────────────────

def find_best_shift(
    edge: np.ndarray,
    outline: np.ndarray,
    res_x: float,
    res_y: float,
) -> tuple[float, float, float, float, float]:
    """
    Returns (dx_m, dy_m, best_score, baseline_score, patch_energy).
    patch_energy is used to normalise score_margin so small plots don't skew it.
    """
    eb = gaussian_filter(edge,    sigma=BLUR_SIGMA)
    ob = gaussian_filter(outline, sigma=BLUR_SIGMA)
    eb /= (eb.max() + 1e-9)
    ob /= (ob.max() + 1e-9)

    patch_energy = float(np.sum(eb))          # normaliser: total edge energy in patch
    baseline     = float(np.sum(eb * ob))     # score at zero shift (official position)

    corr = fftconvolve(eb, ob[::-1, ::-1], mode='same')
    cy0, cx0 = corr.shape[0] // 2, corr.shape[1] // 2
    wy = int(round(SEARCH_M / res_y))
    wx = int(round(SEARCH_M / res_x))
    y0, y1 = max(cy0 - wy, 0), min(cy0 + wy + 1, corr.shape[0])
    x0, x1 = max(cx0 - wx, 0), min(cx0 + wx + 1, corr.shape[1])
    roi     = corr[y0:y1, x0:x1]

    py, px = np.unravel_index(roi.argmax(), roi.shape)
    best   = float(roi.max())
    dy_px  = (py + y0) - cy0
    dx_px  = (px + x0) - cx0
    return dx_px * res_x, -dy_px * res_y, best, baseline, patch_energy


# ── confidence ────────────────────────────────────────────────────────────────

def compute_confidence(
    best: float,
    baseline: float,
    patch_energy: float,
    ratio: float,
    bnd: np.ndarray,
) -> tuple[float, dict]:
    """
    score_margin: normalised by patch_energy so small plots don't inflate it.
    ratio_fit:    proximity to 1.0, zero at the flag boundary.
    hint_density: mean boundary hint strength — confirms edges are real here.
    """
    raw_margin   = (best - baseline) / (patch_energy + 1e-9)
    score_margin = float(min(max(raw_margin * 40.0, 0.0), 1.0))  # scale: ~0.025 raw → 1.0
    ratio_fit    = float(max(0.0, 1.0 - abs(ratio - 1.0) / 0.55))
    hint_density = float(min(bnd.mean() * 4.0, 1.0))

    conf = 0.50 * score_margin + 0.30 * ratio_fit + 0.20 * hint_density
    signals = dict(
        score_margin=round(score_margin, 3),
        ratio_fit=round(ratio_fit, 3),
        hint_density=round(hint_density, 3),
    )
    return float(np.clip(conf, 0.0, 1.0)), signals


def _reproject(geom_m, src_crs):
    tf = Transformer.from_crs(str(src_crs), 'EPSG:4326', always_xy=True)
    return shp_transform(lambda xs, ys, z=None: tf.transform(xs, ys), geom_m)


# ── village-wide fallback shift ───────────────────────────────────────────────

def estimate_village_shift(village, img_src, bnd_src) -> tuple[float, float]:
    """Median shift from well-behaved plots. Used as fallback when per-plot signal is weak."""
    plots = village.plots.copy()
    plots = plots[plots['recorded_area_sqm'].notna() & (plots['recorded_area_sqm'] > 0)]
    plots['_r'] = plots['map_area_sqm'] / plots['recorded_area_sqm']
    good   = plots[(plots['_r'] > 0.85) & (plots['_r'] < 1.15)]
    sample = good.sample(min(N_VILLAGE_SAMPLE, len(good)), random_state=42)

    rx = abs(img_src.transform.a)
    ry = abs(img_src.transform.e)
    dxs, dys = [], []

    for _, row in sample.iterrows():
        try:
            patch = patch_for_plot(img_src, row['geometry'], pad_m=PAD_M)
        except Exception:
            continue
        H, W  = patch.image.shape[:2]
        bnd   = _read_bnd_patch(bnd_src, patch.bounds, (H, W)) if bnd_src else np.zeros((H, W), np.float32)
        sat_e = _sat_edges(patch.image)
        if sat_e.mean() < SAT_EDGE_MIN:
            continue
        edge  = 0.70 * sat_e + 0.30 * bnd
        geom_m  = geom_to_imagery_crs(img_src, row['geometry'])
        outline = _render_outline(geom_m, patch.bounds, (H, W))
        dx_m, dy_m, best, base, energy = find_best_shift(edge, outline, rx, ry)
        margin = (best - base) / (energy + 1e-9)
        if margin > 0.003 and abs(dx_m) < 45 and abs(dy_m) < 45:
            dxs.append(dx_m); dys.append(dy_m)

    if not dxs:
        return 0.0, 0.0
    print(f"  Village shift estimated from {len(dxs)} plots")
    return float(np.median(dxs)), float(np.median(dys))


# ── per-plot correction ────────────────────────────────────────────────────────

def correct_plot(row, img_src, bnd_src, vdx: float, vdy: float) -> dict:
    geom_4326 = row['geometry']
    map_area  = row.get('map_area_sqm')
    rec_area  = row.get('recorded_area_sqm')

    def _flag(note):
        return dict(status='flagged', confidence=None,
                    method_note=note, geometry=geom_4326)

    # Gate A: missing area
    if not rec_area or not map_area:
        return _flag('No recorded area — road, public land, or missing data.')

    ratio = map_area / rec_area

    # Gate B: area ratio
    if ratio < RATIO_LO or ratio > RATIO_HI:
        return _flag(
            f'Area ratio {ratio:.2f} outside [{RATIO_LO},{RATIO_HI}]: '
            f'map={map_area:.0f} m² vs recorded={rec_area:.0f} m². '
            'Shape or ownership-history error — shifting will not help.'
        )

    rx = abs(img_src.transform.a)
    ry = abs(img_src.transform.e)

    # Stage 1: edge signal
    try:
        patch = patch_for_plot(img_src, geom_4326, pad_m=PAD_M)
    except Exception as e:
        return _flag(f'patch_for_plot failed: {e}')

    H, W  = patch.image.shape[:2]
    bnd   = _read_bnd_patch(bnd_src, patch.bounds, (H, W)) if bnd_src else np.zeros((H, W), np.float32)
    sat_e = _sat_edges(patch.image)

    # Gate C: satellite edge strength (checked on sat alone, not the blend)
    sat_mean = float(sat_e.mean())
    if sat_mean < SAT_EDGE_MIN:
        return _flag(
            f'Satellite edge mean {sat_mean:.3f} < {SAT_EDGE_MIN}: '
            'patch has no clear field boundaries (trees, water, buildings, bare soil). '
            'Alignment would be unreliable here.'
        )

    edge    = 0.70 * sat_e + 0.30 * bnd
    geom_m  = geom_to_imagery_crs(img_src, geom_4326)
    outline = _render_outline(geom_m, patch.bounds, (H, W))

    # Stage 2: find best shift
    dx_m, dy_m, best, baseline, energy = find_best_shift(edge, outline, rx, ry)

    # Stage 3: confidence
    conf, signals = compute_confidence(best, baseline, energy, ratio, bnd)
    method = 'per_plot_xcorr'

    # Fallback: if per-plot confidence is low, try village-wide shift
    if conf < CONFIDENCE_THRESH:
        dx_m, dy_m = vdx, vdy
        method = 'village_median_fallback'
        # score the village shift by scoring at the fallback position
        eb = gaussian_filter(edge, sigma=BLUR_SIGMA);  eb /= (eb.max() + 1e-9)
        geom_fb    = translate(geom_m, dx_m, dy_m)
        outline_fb = _render_outline(geom_fb, patch.bounds, (H, W))
        ob = gaussian_filter(outline_fb, sigma=BLUR_SIGMA); ob /= (ob.max() + 1e-9)
        fb_score  = float(np.sum(eb * ob))
        fb_energy = float(np.sum(eb))
        conf, signals = compute_confidence(fb_score, baseline, fb_energy, ratio, bnd)
        conf = min(conf, 0.50)   # cap: fallback is never more trusted than per-plot

    if conf < CONFIDENCE_THRESH:
        return _flag(
            f'Confidence {conf:.3f} < {CONFIDENCE_THRESH} '
            f'[{method}]. Signals: {signals}. Keeping original geometry.'
        )

    geom_out = _reproject(translate(geom_m, dx_m, dy_m), img_src.crs)
    return dict(
        status='corrected',
        confidence=round(conf, 3),
        method_note=(
            f'{method} | dx={dx_m:.1f} m  dy={dy_m:.1f} m | '
            f'ratio={ratio:.2f} | sat_edge={sat_mean:.3f} | {signals}'
        ),
        geometry=geom_out,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def run(village_dir: str | Path) -> None:
    village_dir = Path(village_dir)
    village     = load(village_dir)
    n_truth     = 0 if village.example_truths is None else len(village.example_truths)
    print(f'Loaded {village.slug}')
    print(f'  {len(village.plots)} plots · {n_truth} example truths · '
          f'boundaries={"yes" if village.boundaries_path else "none"}')

    with rasterio.open(village.imagery_path) as img_src:
        bnd_ctx = rasterio.open(village.boundaries_path) if village.boundaries_path else None
        bnd_src = bnd_ctx.__enter__() if bnd_ctx else None
        try:
            print('\nEstimating village-wide fallback shift ...')
            vdx, vdy = estimate_village_shift(village, img_src, bnd_src)
            print(f'  dx={vdx:.2f} m   dy={vdy:.2f} m')

            print('\nCorrecting plots ...')
            records = []
            for i, (_, row) in enumerate(village.plots.iterrows()):
                if i % 100 == 0:
                    print(f'  {i} / {len(village.plots)}', end='\r')
                res = correct_plot(row, img_src, bnd_src, vdx, vdy)
                records.append({'plot_number': str(row['plot_number']), **res})
        finally:
            if bnd_ctx:
                bnd_ctx.__exit__(None, None, None)

    import geopandas as gpd
    preds  = gpd.GeoDataFrame(records, crs='EPSG:4326')
    n_corr = sum(r['status'] == 'corrected' for r in records)
    n_flag = sum(r['status'] == 'flagged'   for r in records)
    print(f'\n  {n_corr} corrected · {n_flag} flagged')

    out_path = village.dir / 'predictions.geojson'
    write_predictions(out_path, preds)
    print(f'  Written → {out_path}')

    if village.example_truths is not None:
        print()
        print(score(preds, village))


if __name__ == '__main__':
    village_path = sys.argv[1] if len(sys.argv) > 1 else \
        'data/34855_vadnerbhairav_chandavad_nashik'
    run(village_path)
