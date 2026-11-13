# -*- coding: utf-8 -*-
"""Check the star-doubling detector against real frames.

Four questions, in order of how much they matter:

1. Detections on real frames -- it should stay quiet, and whatever it
   does flag is worth opening the preview for rather than writing off
   as a false alarm.
2. Sensitivity -- fed a doubling of known offset, does it recover it,
   and down to what amplitude?
3. Asymmetry -- a brief jolt rarely splits the exposure evenly, so the
   second image is often fainter. How faint before we lose it?
4. Headroom -- with every threshold switched off, what fraction does the
   best vector of a frame gather? The bulk of that distribution is the
   noise floor, and ``doubling_min_fraction`` has to sit well above it.
   Read the median and p90, not the maximum: a genuinely doubled frame
   in the sample lands at the top and is not noise at all.

Usage::

    python calibrate_doubling.py "<Target>_sub" [frame_count]

A tuning aid; delete it once the thresholds have settled.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from dataclasses import replace

import numpy as np

import subframe_inspector as triage


def stars_in(frame: np.ndarray, settings: triage.Settings,
             thresholds: triage.Thresholds) -> list[dict]:
    """Stars of a full-resolution crop, through the pipeline's own code.

    Calls straight into the triage module rather than reimplementing the
    selection rules: an earlier copy here drifted out of step with them
    and quietly validated thresholds the pipeline no longer used.
    """
    return triage.measure_stars(frame, settings, thresholds).blobs


def centre_of(data: np.ndarray, settings: triage.Settings) -> np.ndarray:
    """The same central crop the pipeline measures at full resolution."""
    return triage.crop_centre(data, settings)


def detect(stars: list[dict], settings: triage.Settings,
           thresholds: triage.Thresholds, **overrides) -> dict | None:
    """Run the detector, optionally with some thresholds overridden."""
    if overrides:
        settings = replace(settings, **overrides)
    return triage.find_star_doubling(stars, settings, thresholds)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    folder = sys.argv[1]
    wanted = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    settings = triage.DEFAULT_SETTINGS

    paths = sorted(glob.glob(os.path.join(folder, "*.fit*")))
    if not paths:
        print(f"No FITS files in '{folder}'.")
        return 1

    # The thresholds are quoted in arcseconds and only become pixels once
    # the camera is known, so they come from the frames themselves.
    geometry = triage.Geometry.of(paths[0], settings)
    thresholds = triage.Thresholds.derive(settings, geometry)

    print(f"{len(paths)} frames in '{folder}', {wanted} analysed")
    print(f"{geometry.model} at {geometry.pixel_scale:.2f} arcsec/px")
    print(f"Thresholds: distance {thresholds.doubling_min_distance:.0f}-"
          f"{thresholds.doubling_max_distance:.0f}px (full resolution), "
          f"tolerance {thresholds.doubling_tolerance:.1f}px, "
          f"min fraction {settings.doubling_min_fraction:.0%}, "
          f"min {settings.doubling_min_pairs} pairs\n")

    # -- 1. Detections on real frames ------------------------------------------------
    print("--- 1. DETECTIONS on real frames (expect few; check each one) ---")
    baseline: np.ndarray | None = None
    positives = 0
    started = time.time()
    for path in paths[:wanted]:
        data = triage.load_frame(path)
        if data is None:
            continue
        centre = centre_of(data, settings)
        if baseline is None:
            baseline = centre
        found = detect(stars_in(centre, settings, thresholds), settings, thresholds)
        positives += found is not None
        verdict = "-" if found is None else (
            f"FOUND {found['distance']:.1f}px "
            f"fraction={found['fraction']:.0%} pairs={found['pairs']}")
        print(f"  {os.path.basename(path)[-19:]:21s}  {verdict}")
    print(f"  => {positives}/{wanted} flagged "
          f"({(time.time() - started) / max(wanted, 1):.2f}s per frame)")
    print("     Open the preview for each: from here a real doubling and a "
          "false alarm look identical.\n")

    if baseline is None:
        print("No usable frame; stopping here.")
        return 1

    # -- 2. Sensitivity ----------------------------------------------------
    print("--- 2. SENSITIVITY: synthetic doubling of known offset ---")
    for dx, dy in [(4, 2), (8, 6), (16, 0), (24, 18), (40, 30), (60, 50), (90, 60)]:
        doubled = np.maximum(baseline, np.roll(np.roll(baseline, dy, 0), dx, 1))
        found = detect(stars_in(doubled, settings, thresholds), settings, thresholds)
        expected = np.hypot(dx, dy)
        if found is None:
            print(f"  ({dx:3d},{dy:3d}) distance {expected:5.1f}px -> MISSED")
        else:
            error = np.hypot(found["dx"] - dx, found["dy"] - dy)
            print(f"  ({dx:3d},{dy:3d}) distance {expected:5.1f}px -> FOUND  "
                  f"dx={found['dx']:5.1f} dy={found['dy']:5.1f} err={error:4.1f}px "
                  f"fraction={found['fraction']:3.0%} pairs={found['pairs']:4d}")

    # -- 3. Asymmetry ------------------------------------------------------
    print("\n--- 3. ASYMMETRIC doubling (offset 24,18), second image fainter ---")
    reference = baseline.astype(np.float32)
    for weight in [1.0, 0.7, 0.5, 0.35, 0.25]:
        doubled = np.maximum(reference,
                             weight * np.roll(np.roll(reference, 18, 0), 24, 1))
        stars = stars_in(doubled, settings, thresholds)
        raw = detect(stars, settings, thresholds,
                     doubling_min_fraction=0.0, doubling_min_pairs=1)
        actual = detect(stars, settings, thresholds)
        fraction = "-" if raw is None else f"{raw['fraction']:3.0%}"
        print(f"  second image at {weight:4.2f} -> true fraction {fraction:>4s} | "
              f"{'FOUND' if actual else 'missed'}")

    # -- 4. Headroom ----------------------------------------------------
    print("\n--- 4. HEADROOM: best vector on clean frames, thresholds off ---")
    step = max(1, len(paths) // 25)
    fractions = []
    for path in paths[::step][:25]:
        data = triage.load_frame(path)
        if data is None:
            continue
        found = detect(stars_in(centre_of(data, settings), settings, thresholds),
                       settings, thresholds,
                       doubling_min_fraction=0.0, doubling_min_pairs=1)
        if found:
            fractions.append(found["fraction"])

    if fractions:
        values = np.array(fractions)
        floor = float(np.percentile(values, 90))
        print(f"  n={len(values)}  median={np.median(values):.0%}  "
              f"p90={floor:.0%}  max={values.max():.0%}")
        print(f"  => noise floor (p90) {floor:.0%} against a threshold of "
              f"{settings.doubling_min_fraction:.0%}: "
              f"{settings.doubling_min_fraction / max(floor, 0.01):.1f}x headroom")
        if values.max() > settings.doubling_min_fraction:
            print("     The maximum sits above the threshold, which almost "
                  "certainly means a real doubling in the sample, not noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
