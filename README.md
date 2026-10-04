# AgriParcel AI

Agricultural parcel and bund boundary extraction from drone imagery.
AgriParcel AI combines a trained computer-vision model, a Streamlit inference
application, and a React/Vite frontend for reviewing parcel maps, bund
detection results, GIS analysis, and exports.

## Features

- Upload RGB GeoTIFF, TIFF, JPG, or PNG imagery for inference.
- Detect agricultural bund and parcel boundaries with a ResNet34 U-Net model.
- Process large images as georeferenced tiles without changing source pixel
  alignment.
- Export boundary masks, parcel masks, instance masks, and GeoJSON.
- Preserve source CRS and affine transform for geospatial outputs.
- Review survey imagery, parcel statistics, GIS analysis, reports, and export
  workflows in the React frontend.

## Project layout

```text
.
├── streamlit_predict_bund_parcel.py       # Python inference application
├── paired_boundary_training_run/
│   └── best_checkpoint.pt                  # Git LFS model checkpoint
├── Aapakudal_Farm_Digitization/           # Training shapefile and metadata
├── build_paired_dataset.py                # Build aligned training tiles
├── generate_boundary_labels.py             # Generate parcel/boundary labels
├── split_drone_tif.py                     # Split source imagery into tiles
├── train_from_digitized_boundaries.py     # Train the segmentation model
├── requirements-streamlit-inference.txt
├── requirements-boundary-project.txt
└── Demystifying-Networking-main/
    └── Demystifying-Networking-main/      # React/Vite frontend
```

Large source rasters, generated datasets, caches, and temporary training
outputs are intentionally excluded from Git. The trained inference checkpoint
is tracked with Git LFS.

## Quick start: Streamlit inference

Use Python 3.10 or 3.11. A CUDA-enabled PyTorch installation is recommended
when GPU inference is available.

```powershell
cd D:\da
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-streamlit-inference.txt
streamlit run streamlit_predict_bund_parcel.py
```

After cloning, install and fetch Git LFS before starting the app:

```powershell
git lfs install
git lfs pull
```

The app automatically uses
`paired_boundary_training_run/best_checkpoint.pt`. It selects CUDA when
available and otherwise runs on the CPU.

## Frontend

The frontend is located in
`Demystifying-Networking-main/Demystifying-Networking-main`.

```powershell
cd D:\da\Demystifying-Networking-main\Demystifying-Networking-main
npm install
npm run dev
```

Useful validation commands:

```powershell
npm run lint
npm run typecheck
npm run build
```

To connect the frontend to a deployed inference service, copy `.env.example`
to `.env` and set `VITE_INFERENCE_API_URL`. The service accepts an image
multipart field named `image` and should return model results, mask URLs,
statistics, and optional GeoJSON.

## Training workflow

The training pipeline uses digitized parcel boundaries and aligned drone-image
tiles.

### Generate labels

```powershell
python generate_boundary_labels.py `
  --tiles "D:\da\dataset_tiles" `
  --shapefile "D:\da\Aapakudal_Farm_Digitization\Aapakudal_Farm_Digitization.shp" `
  --output "D:\da\ground_truth_dataset" `
  --boundary-thickness 5 `
  --workers 6
```

### Train the model

```powershell
python train_from_digitized_boundaries.py `
  --tiles "D:\da\dataset_tiles" `
  --labels "D:\da\ground_truth_dataset" `
  --output "D:\da\boundary_training_run" `
  --epochs 50 `
  --batch-size 4 `
  --crop-size 512 `
  --workers 4
```

For the complete training options and recovery instructions, see
[README_BOUNDARY_PROJECT.md](README_BOUNDARY_PROJECT.md).

## Model and outputs

The current model is a ResNet34 encoder with a U-Net segmentation decoder.
The inference application produces:

- boundary probability and binary masks;
- parcel masks;
- connected-component instance masks;
- GeoTIFF outputs with source georeferencing;
- GeoJSON parcel features when vector extraction succeeds.

The model checkpoint is approximately 280 MB and is stored through Git LFS.
Do not replace it with a regular Git blob.

## Data and privacy

The repository contains project code, a small digitized shapefile, frontend
assets, and the trained checkpoint. Full drone rasters and generated training
datasets are local-only and ignored by Git. Review imagery and vector data
before publishing this repository if the source data is subject to privacy,
landowner, or organizational restrictions.

## License

No license has been declared yet. Add a `LICENSE` file before distributing the
project or accepting external contributions.
