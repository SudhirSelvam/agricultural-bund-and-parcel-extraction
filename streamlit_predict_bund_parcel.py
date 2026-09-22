r"""Streamlit inference app for agricultural bund and parcel extraction.

Run:
    streamlit run D:\da\streamlit_predict_bund_parcel.py
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import streamlit as st
import torch
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import mapping, shape


APP_TITLE = "AgriParcel AI — Bund & Parcel Prediction"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "paired_boundary_training_run" / "best_checkpoint.pt"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, checkpoint_mtime: float) -> tuple[torch.nn.Module, torch.device, dict[str, Any]]:
    del checkpoint_mtime  # Included only to invalidate Streamlit's cache after retraining.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    metadata = {
        "epoch": int(checkpoint.get("epoch", 0)),
        "best_val_dice": float(checkpoint.get("best_val_dice", float("nan"))),
    }
    return model, device, metadata


def normalize_tile(tile: np.ndarray) -> torch.Tensor:
    """Convert a CHW RGB uint image to the ImageNet-normalized training format."""
    image = np.moveaxis(tile[:3], 0, 2).astype(np.float32)
    if image.max(initial=0) > 255:
        low, high = np.percentile(image, (1, 99))
        image = np.clip((image - low) * 255.0 / max(high - low, 1e-6), 0, 255)
    image = image / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.moveaxis(image, 2, 0)).unsqueeze(0).float()


def predict_boundary(
    src: rasterio.io.DatasetReader,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int,
    threshold: float,
    progress: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run non-overlapping tiled prediction without resizing the source pixels."""
    height, width = src.height, src.width
    probability = np.zeros((height, width), dtype=np.float32)
    total = math.ceil(height / tile_size) * math.ceil(width / tile_size)
    completed = 0
    amp_enabled = device.type == "cuda"

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            actual_h = min(tile_size, height - row)
            actual_w = min(tile_size, width - col)
            window = Window(col, row, actual_w, actual_h)
            tile = src.read(
                [1, 2, 3],
                window=window,
                boundless=True,
                fill_value=0,
                out_shape=(3, tile_size, tile_size),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            tensor = normalize_tile(tile).to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
            probability[row : row + actual_h, col : col + actual_w] = prediction[:actual_h, :actual_w]
            completed += 1
            progress.progress(completed / total, text=f"Predicting tile {completed:,} of {total:,}")

    return probability, (probability >= threshold).astype(np.uint8)


def create_instances(
    boundary: np.ndarray,
    close_pixels: int,
    thickness_pixels: int,
    min_parcel_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close boundary gaps and label enclosed non-boundary regions as parcels."""
    cleaned = boundary.copy()
    if close_pixels > 0:
        size = close_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    if thickness_pixels > 1:
        size = thickness_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    count, components, stats, _ = cv2.connectedComponentsWithStats((cleaned == 0).astype(np.uint8), 8)
    border_ids = np.unique(
        np.concatenate((components[0], components[-1], components[:, 0], components[:, -1]))
    )
    border_areas = {int(label): int(stats[label, cv2.CC_STAT_AREA]) for label in border_ids if label > 0}
    outside_id = max(border_areas, key=border_areas.get) if border_areas else 0

    instances = np.zeros_like(components, dtype=np.uint32)
    next_id = 1
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if label == outside_id or area < min_parcel_pixels:
            continue
        instances[components == label] = next_id
        next_id += 1
    return cleaned, (instances > 0).astype(np.uint8), instances


def vectorize_instances(
    instances: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS | None,
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for geometry, value in shapes(instances, mask=instances > 0, transform=transform):
        parcel_id = int(value)
        geom = shape(geometry)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "parcel_id": parcel_id,
                    "area_map_units2": round(float(geom.area), 3),
                    "perimeter_map_units": round(float(geom.length), 3),
                },
                "geometry": mapping(geom),
            }
        )
    result: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs:
        result["crs"] = {"type": "name", "properties": {"name": crs.to_string()}}
    return result


def render_preview(
    src: rasterio.io.DatasetReader,
    boundary: np.ndarray,
    parcels: np.ndarray,
    max_size: int = 1400,
) -> tuple[np.ndarray, np.ndarray]:
    scale = min(1.0, max_size / max(src.width, src.height))
    width = max(1, int(src.width * scale))
    height = max(1, int(src.height * scale))
    rgb = np.moveaxis(
        src.read([1, 2, 3], out_shape=(3, height, width), resampling=rasterio.enums.Resampling.bilinear),
        0,
        2,
    )
    if rgb.dtype != np.uint8:
        low, high = np.percentile(rgb, (1, 99))
        rgb = np.clip((rgb.astype(np.float32) - low) * 255 / max(high - low, 1e-6), 0, 255).astype(np.uint8)
    boundary_small = cv2.resize(boundary, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    parcel_small = cv2.resize(parcels, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    overlay = rgb.copy()
    overlay[parcel_small] = (0.72 * overlay[parcel_small] + 0.28 * np.array([34, 197, 94])).astype(np.uint8)
    overlay[boundary_small] = np.array([239, 68, 68], dtype=np.uint8)
    return rgb, overlay


def geotiff_bytes(array: np.ndarray, profile: dict[str, Any], dtype: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        output_profile = profile.copy()
        output_profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            nodata=0,
            compress="deflate",
            predictor=2,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(path, "w", **output_profile) as dst:
            dst.write(array.astype(dtype), 1)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def run_pipeline(
    image_path: Path,
    checkpoint_path: Path,
    tile_size: int,
    threshold: float,
    close_pixels: int,
    thickness_pixels: int,
    min_parcel_pixels: int,
) -> dict[str, Any]:
    model, device, checkpoint_info = load_model(str(checkpoint_path), checkpoint_path.stat().st_mtime)
    started = time.perf_counter()
    progress = st.progress(0.0, text="Opening raster")
    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise ValueError("The input must contain at least three RGB bands.")
        pixels = src.width * src.height
        if pixels > 150_000_000:
            raise ValueError(
                "This raster exceeds 150 million pixels. First run the model on generated 1024×1024 tiles, "
                "then mosaic the masks; whole-image connected-component parcel extraction would require too much RAM."
            )
        probability, raw_boundary = predict_boundary(src, model, device, tile_size, threshold, progress)
        progress.progress(0.93, text="Closing gaps and extracting parcel instances")
        boundary, parcel_mask, instances = create_instances(
            raw_boundary, close_pixels, thickness_pixels, min_parcel_pixels
        )
        geojson = vectorize_instances(instances, src.transform, src.crs)
        original_preview, overlay_preview = render_preview(src, boundary, parcel_mask)
        profile = src.profile.copy()
        boundary_tif = geotiff_bytes(boundary, profile, "uint8")
        parcel_tif = geotiff_bytes(parcel_mask, profile, "uint8")
        instance_dtype = "uint16" if int(instances.max(initial=0)) <= 65_535 else "uint32"
        instance_tif = geotiff_bytes(instances, profile, instance_dtype)
        metadata = {
            "source": image_path.name,
            "crs": src.crs.to_string() if src.crs else None,
            "width": src.width,
            "height": src.height,
            "transform": tuple(src.transform),
            "model_epoch": checkpoint_info["epoch"],
            "best_validation_dice": checkpoint_info["best_val_dice"],
            "threshold": threshold,
            "parcel_count": len(geojson["features"]),
            "boundary_pixels": int(boundary.sum()),
            "parcel_pixels": int(parcel_mask.sum()),
            "label_coverage_percent": round(100.0 * float(parcel_mask.mean()), 4),
            "processing_seconds": round(time.perf_counter() - started, 3),
            "device": str(device),
        }
    progress.progress(1.0, text="Complete")
    geojson_bytes = json.dumps(geojson, indent=2).encode("utf-8")
    metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("boundary_mask.tif", boundary_tif)
        archive.writestr("parcel_mask.tif", parcel_tif)
        archive.writestr("instance_mask.tif", instance_tif)
        archive.writestr("parcels.geojson", geojson_bytes)
        archive.writestr("prediction_metadata.json", metadata_bytes)
    return {
        "metadata": metadata,
        "original_preview": original_preview,
        "overlay_preview": overlay_preview,
        "boundary_tif": boundary_tif,
        "parcel_tif": parcel_tif,
        "instance_tif": instance_tif,
        "geojson": geojson_bytes,
        "metadata_json": metadata_bytes,
        "zip": bundle.getvalue(),
    }


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🌾", layout="wide")
    st.title(APP_TITLE)
    st.caption("ResNet34 U-Net boundary inference with georeferenced parcel extraction")

    with st.sidebar:
        st.header("Model")
        checkpoint_text = st.text_input("Checkpoint", str(DEFAULT_CHECKPOINT))
        checkpoint_path = Path(checkpoint_text.strip().strip('"'))
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.write(f"Device: **{device_name}**")

        st.header("Prediction")
        tile_size = st.select_slider("Tile size", options=[256, 512, 768, 1024], value=1024)
        threshold = st.slider("Boundary threshold", 0.05, 0.95, 0.50, 0.05)
        close_pixels = st.slider("Close boundary gaps (px)", 0, 20, 3)
        thickness_pixels = st.slider("Boundary expansion (px)", 1, 10, 2)
        min_parcel_pixels = st.number_input("Minimum parcel size (pixels)", 25, 10_000_000, 2_000, 25)

    source_mode = st.radio("Input source", ["Upload image", "Local file path"], horizontal=True)
    input_path: Path | None = None
    temporary_path: Path | None = None
    if source_mode == "Upload image":
        uploaded = st.file_uploader("Upload RGB GeoTIFF, TIFF, JPG, or PNG", type=["tif", "tiff", "jpg", "jpeg", "png"])
        if uploaded:
            suffix = Path(uploaded.name).suffix or ".tif"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(uploaded.getbuffer())
                temporary_path = Path(temp.name)
            input_path = temporary_path
    else:
        local_text = st.text_input("Local raster path", r"D:\da\paired_dataset\images\test\test_r022113_c069284.tif")
        if local_text.strip():
            input_path = Path(local_text.strip().strip('"'))

    can_run = input_path is not None and input_path.exists() and checkpoint_path.exists()
    if input_path is not None and not input_path.exists():
        st.error(f"Input image does not exist: {input_path}")
    if not checkpoint_path.exists():
        st.error(f"Checkpoint does not exist: {checkpoint_path}")

    if st.button("Run bund and parcel prediction", type="primary", disabled=not can_run, use_container_width=True):
        try:
            with st.spinner("Loading model and running tiled inference…"):
                st.session_state["prediction"] = run_pipeline(
                    input_path,
                    checkpoint_path,
                    int(tile_size),
                    float(threshold),
                    int(close_pixels),
                    int(thickness_pixels),
                    int(min_parcel_pixels),
                )
            st.success("Prediction completed.")
        except Exception as error:
            st.exception(error)
        finally:
            if temporary_path:
                temporary_path.unlink(missing_ok=True)

    result = st.session_state.get("prediction")
    if not result:
        st.info("Choose an image and run prediction. No result values are fabricated before inference.")
        return

    metadata = result["metadata"]
    dice = metadata["best_validation_dice"]
    if not math.isnan(dice) and dice < 0.5:
        st.warning(
            f"This checkpoint's best validation Dice is only {dice:.4f} at epoch {metadata['model_epoch']}. "
            "The application works, but the predicted boundaries may be unreliable until training improves."
        )

    columns = st.columns(5)
    columns[0].metric("Parcels", f"{metadata['parcel_count']:,}")
    columns[1].metric("Boundary pixels", f"{metadata['boundary_pixels']:,}")
    columns[2].metric("Parcel coverage", f"{metadata['label_coverage_percent']:.2f}%")
    columns[3].metric("Processing", f"{metadata['processing_seconds']:.1f} s")
    columns[4].metric("Model epoch", metadata["model_epoch"])

    original_tab, overlay_tab, metadata_tab = st.tabs(["Original", "Prediction overlay", "Metadata"])
    with original_tab:
        st.image(result["original_preview"], caption=metadata["source"], use_container_width=True)
    with overlay_tab:
        st.image(result["overlay_preview"], caption="Red: predicted bunds · Green: extracted parcels", use_container_width=True)
    with metadata_tab:
        st.json(metadata)

    st.subheader("Download GIS outputs")
    buttons = st.columns(3)
    stem = Path(metadata["source"]).stem
    buttons[0].download_button("Download all outputs (.zip)", result["zip"], f"{stem}_prediction.zip", "application/zip", use_container_width=True)
    buttons[1].download_button("Download parcels (.geojson)", result["geojson"], f"{stem}_parcels.geojson", "application/geo+json", use_container_width=True)
    buttons[2].download_button("Download metadata (.json)", result["metadata_json"], f"{stem}_metadata.json", "application/json", use_container_width=True)
    mask_buttons = st.columns(3)
    mask_buttons[0].download_button("Boundary GeoTIFF", result["boundary_tif"], f"{stem}_boundary.tif", "image/tiff", use_container_width=True)
    mask_buttons[1].download_button("Parcel GeoTIFF", result["parcel_tif"], f"{stem}_parcel.tif", "image/tiff", use_container_width=True)
    mask_buttons[2].download_button("Instance GeoTIFF", result["instance_tif"], f"{stem}_instances.tif", "image/tiff", use_container_width=True)


if __name__ == "__main__":
    main()
