"""Extract bead geometry (height, width, area) from Xiris RAW weld camera data.

Bead geometry is extracted from the spatial extent of the bright region:
  - Width: horizontal span of the bead profile (pixels -> mm)
  - Height: vertical span of the bead region (pixels -> mm)
  - Area: cross-sectional area via trapezoidal integration (mm^2)

Calibration: PIXEL_TO_MM = 0.05 mm/pixel (~50 um/pixel at typical working distance)
"""

import struct
import json
import os
import csv
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
from scipy import ndimage

# Xiris pixel-to-mm calibration
# 640x512 at typical WAAM working distance (~200mm)
PIXEL_TO_MM = 0.05  # ~50 um/pixel

# Time alignment coefficients (linear fit to align with aligned_data.csv)
# Formula: RelativeTime = slope * (MM*60 + SS.us) + intercept
# where MM:SS.us is extracted from the Unix timestamp in the filename
TIME_ALIGN = {
    "good": {"slope": 1.00000613, "intercept": -720.194556},
    "bad":  {"slope": 1.00008908, "intercept": -140.797584},
}


def filename_to_timestamps(filepath: str, bead_type: str = "good") -> dict:
    """Convert Xiris filename (Unix microsecond timestamp) to multiple time formats.

    Args:
        filepath: Path to .raw file (filename is Unix us timestamp).
        bead_type: 'good' or 'bad' to select the correct time alignment coefficients.

    Returns:
        Dict with unix_ts_us, unix_ts_s, datetime_str, rel_time_s.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    ts_us = int(basename)
    ts_s = ts_us / 1e6
    dt = datetime.fromtimestamp(ts_s)

    # Compute relative time aligned with aligned_data.csv
    mm_ss = dt.minute * 60 + dt.second + dt.microsecond / 1e6
    coeff = TIME_ALIGN[bead_type]
    rel_time = coeff["slope"] * mm_ss + coeff["intercept"]

    return {
        "unix_ts_us": ts_us,
        "unix_ts_s": ts_s,
        "datetime_str": dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "rel_time_s": rel_time,
    }


def parse_xiris_raw(filepath: str) -> Optional[dict]:
    """Parse a Xiris V4 raw file and return image array + metadata."""
    fsize = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        if fsize < 56:
            return None

        data = f.read()

    fields = struct.unpack("<iiiiiiii", data[0:32])
    _, header_len, aoi_left, aoi_top, aoi_right, aoi_bottom, img_width, bitdepth = fields
    pixel_fmt = struct.unpack("<i", data[32:36])[0]
    timestamp = struct.unpack("<d", data[40:48])[0]

    img_height = aoi_bottom - aoi_top
    pixel_data_size = img_height * img_width * bitdepth // 8
    pixel_start = header_len
    pixel_end = pixel_start + pixel_data_size

    if len(data) < pixel_end:
        return None

    pixels = np.frombuffer(data[pixel_start:pixel_end], dtype=np.uint16)
    pixels = pixels.reshape(img_height, img_width)
    img = pixels[:, aoi_left:aoi_right]

    footer = {}
    if pixel_end < len(data):
        try:
            footer = json.loads(data[pixel_end:].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    return {
        "image": img,
        "timestamp": footer.get("timeStamp", timestamp),
        "frame_number": footer.get("frameNumber", 0),
        "exposure_us": footer.get("Camera", {}).get("ExposureTime", np.nan),
        "source_temp_C": footer.get("sourceTemperature", np.nan),
    }


def extract_meltpool_features(img: np.ndarray) -> dict:
    """Extract melt pool geometry using adaptive thresholding.

    The melt pool appears as a bright region against a darker background.
    Uses percentile-based thresholding to handle varying exposure.
    """
    p50 = np.percentile(img, 50)
    p99 = np.percentile(img, 99)
    threshold = p50 + 0.5 * (p99 - p50)

    binary = img > threshold
    labeled, n_objects = ndimage.label(binary)

    if n_objects == 0:
        return {
            "meltpool_width_px": 0.0,
            "meltpool_height_px": 0.0,
            "meltpool_area_px": 0.0,
            "meltpool_centroid_x": np.nan,
            "meltpool_centroid_y": np.nan,
            "meltpool_circularity": 0.0,
        }

    # Find largest connected component (the melt pool)
    component_sizes = ndimage.sum(binary, labeled, range(1, n_objects + 1))
    largest_label = int(np.argmax(component_sizes)) + 1
    meltpool_mask = labeled == largest_label

    area_px = float(component_sizes[largest_label - 1])

    # Bounding box
    rows = np.any(meltpool_mask, axis=1)
    cols = np.any(meltpool_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    width_px = float(cmax - cmin + 1)
    height_px = float(rmax - rmin + 1)

    # Centroid
    cy, cx = ndimage.center_of_mass(meltpool_mask)

    # Circularity
    eroded = ndimage.binary_erosion(meltpool_mask)
    perimeter = max(float(np.sum(meltpool_mask) - np.sum(eroded)), 1.0)
    circularity = min(4.0 * np.pi * area_px / (perimeter ** 2), 1.0)

    return {
        "meltpool_width_px": width_px,
        "meltpool_height_px": height_px,
        "meltpool_area_px": area_px,
        "meltpool_centroid_x": float(cx),
        "meltpool_centroid_y": float(cy),
        "meltpool_circularity": circularity,
    }


def extract_bead_geometry(img: np.ndarray) -> dict:
    """Estimate bead cross-section geometry from the Xiris frame.

    Uses the column-mean intensity profile as a proxy for the bead
    cross-section. The bright melt pool and heated bead region create a
    characteristic profile that can be used to estimate width and height.

    The bead profile is extracted from the lower portion of the frame
    (where the bead/substrate is) using intensity thresholding.
    """
    h, w = img.shape

    # Use lower 60% of frame (bead region, not arc/wire above)
    bead_region = img[int(h * 0.4):, :]

    # Column-mean profile (average across rows -> 1D profile along width)
    col_profile = bead_region.mean(axis=0).astype(float)

    # Baseline subtraction: use edges as background
    edge_width = max(10, w // 20)
    baseline = np.mean(np.concatenate([
        col_profile[:edge_width], col_profile[-edge_width:]
    ]))
    profile = col_profile - baseline

    # Threshold to find bead region
    peak_val = np.max(profile)
    if peak_val < 50:  # No significant bead signal
        return {
            "bead_height_mm": 0.0,
            "bead_width_mm": 0.0,
            "bead_area_mm2": 0.0,
        }

    threshold = peak_val * 0.2  # 20% of peak
    above_thresh = profile > threshold

    if not np.any(above_thresh):
        return {"bead_height_mm": 0.0, "bead_width_mm": 0.0, "bead_area_mm2": 0.0}

    # Find bead edges
    positions = np.where(above_thresh)[0]
    left_edge = positions[0]
    right_edge = positions[-1]
    width_px = right_edge - left_edge + 1

    # Bead height proxy: use row-mean profile in the bead width region
    bead_cols = img[:, left_edge:right_edge + 1]
    row_profile = bead_cols.mean(axis=1).astype(float)

    # Find the bead extent (where intensity rises significantly from top)
    row_baseline = np.mean(row_profile[:max(5, h // 20)])
    row_peak = np.max(row_profile)
    row_thresh = row_baseline + 0.3 * (row_peak - row_baseline)
    above_row = row_profile > row_thresh

    if np.any(above_row):
        row_positions = np.where(above_row)[0]
        top_row = row_positions[0]
        bottom_row = row_positions[-1]
        height_px = bottom_row - top_row + 1
    else:
        height_px = 0

    # Convert to mm
    width_mm = width_px * PIXEL_TO_MM
    height_mm = height_px * PIXEL_TO_MM

    # Cross-section area (trapezoidal integration of profile)
    bead_profile_mm = profile[left_edge:right_edge + 1] / peak_val * height_mm
    area_mm2 = float(np.trapezoid(bead_profile_mm, dx=PIXEL_TO_MM))

    return {
        "bead_height_mm": round(height_mm, 3),
        "bead_width_mm": round(width_mm, 3),
        "bead_area_mm2": round(max(area_mm2, 0.0), 6),
    }


def process_folder(
    folder_path: str, output_csv: str, bead_type: str = "good"
) -> None:
    """Process all RAW files in a folder and output bead geometry CSV.

    Args:
        folder_path: Path to folder containing .raw files.
        output_csv: Path for output CSV file.
        bead_type: 'good' or 'bad' for correct time alignment coefficients.
    """
    raw_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".raw")]
    )
    print(f"Processing {len(raw_files)} files from {folder_path} (bead_type={bead_type})")

    rows = []
    for i, fname in enumerate(raw_files):
        fpath = os.path.join(folder_path, fname)
        data = parse_xiris_raw(fpath)
        if data is None:
            print(f"  Skipping {fname}: could not parse")
            continue

        img = data["image"]

        # Timestamps from filename (aligned with aligned_data.csv)
        ts = filename_to_timestamps(fpath, bead_type=bead_type)

        # Intensity features
        mean_intensity = float(img.mean())
        max_intensity = float(img.max())
        std_intensity = float(img.std())

        # Melt pool features
        mp = extract_meltpool_features(img)

        # Bead geometry
        bg = extract_bead_geometry(img)

        row = {
            "filename": fname,
            "frame_number": data["frame_number"],
            "rel_time": round(ts["rel_time_s"], 6),
            "unix_timestamp_us": ts["unix_ts_us"],
            "datetime": ts["datetime_str"],
            "camera_timestamp_s": round(data["timestamp"], 6),
            "mean_intensity": round(mean_intensity, 2),
            "max_intensity": round(max_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "exposure_us": data["exposure_us"],
            "source_temp_C": data["source_temp_C"],
            "meltpool_width_px": mp["meltpool_width_px"],
            "meltpool_height_px": mp["meltpool_height_px"],
            "meltpool_area_px": mp["meltpool_area_px"],
            "meltpool_centroid_x": round(mp["meltpool_centroid_x"], 2)
            if not np.isnan(mp["meltpool_centroid_x"]) else "",
            "meltpool_centroid_y": round(mp["meltpool_centroid_y"], 2)
            if not np.isnan(mp["meltpool_centroid_y"]) else "",
            "meltpool_circularity": round(mp["meltpool_circularity"], 4),
            "bead_height_mm": bg["bead_height_mm"],
            "bead_width_mm": bg["bead_width_mm"],
            "bead_area_mm2": bg["bead_area_mm2"],
        }
        rows.append(row)

        if (i + 1) % 200 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(raw_files)}] frame={data['frame_number']}, "
                f"rel_t={ts['rel_time_s']:.4f}s, "
                f"height={bg['bead_height_mm']:.3f}mm, "
                f"width={bg['bead_width_mm']:.3f}mm, "
                f"area={bg['bead_area_mm2']:.3f}mm2"
            )

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Print summary
        active = [r for r in rows if r["bead_height_mm"] > 0]
        print(f"\nWrote {len(rows)} rows to {output_csv}")
        print(f"Frames with bead detected: {len(active)} / {len(rows)}")
        if active:
            heights = [r["bead_height_mm"] for r in active]
            widths = [r["bead_width_mm"] for r in active]
            areas = [r["bead_area_mm2"] for r in active]
            print(f"  Height: min={min(heights):.3f}, max={max(heights):.3f}, "
                  f"mean={sum(heights)/len(heights):.3f} mm")
            print(f"  Width:  min={min(widths):.3f}, max={max(widths):.3f}, "
                  f"mean={sum(widths)/len(widths):.3f} mm")
            print(f"  Area:   min={min(areas):.3f}, max={max(areas):.3f}, "
                  f"mean={sum(areas)/len(areas):.3f} mm2")
    else:
        print("No valid frames found.")


def main() -> None:
    base_dir = Path(__file__).parent

    # Process good bead
    good_folder = base_dir / "xiris_good"
    good_csv = base_dir / "bead_geometry_good.csv"
    if good_folder.exists():
        process_folder(str(good_folder), str(good_csv), bead_type="good")
    else:
        print(f"Folder not found: {good_folder}")

    print("\n" + "=" * 60 + "\n")

    # Process bad bead
    bad_folder = base_dir / "xiris_bad"
    bad_csv = base_dir / "bead_geometry_bad.csv"
    if bad_folder.exists():
        process_folder(str(bad_folder), str(bad_csv), bead_type="bad")
    else:
        print(f"Folder not found: {bad_folder}")


if __name__ == "__main__":
    main()
