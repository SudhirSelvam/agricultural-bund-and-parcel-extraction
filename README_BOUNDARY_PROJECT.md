# Agricultural parcel boundary dataset and training pipeline

This project turns the digitized parcel shapefile into three exactly aligned
GeoTIFF label types and trains a ResNet34 U-Net on the boundary labels.

## 1. Install dependencies

Use a Python 3.10 or 3.11 virtual environment. Install a CUDA-enabled PyTorch
build suitable for the local NVIDIA driver first, then install the remaining
packages:

```powershell
cd D:\da
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# CUDA 12.4 example. Use the index recommended by pytorch.org for your driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-boundary-project.txt
```

Confirm that PyTorch sees the RTX 4050:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Generate georeferenced masks

PowerShell uses the backtick for multiline commands:

```powershell
python generate_boundary_labels.py `
  --tiles "D:\da\dataset_tiles" `
  --shapefile "D:\da\Aapakudal_Farm_Digitization\Aapakudal_Farm_Digitization.shp" `
  --output "D:\da\ground_truth_dataset" `
  --boundary-thickness 5 `
  --workers 6
```

For Command Prompt (`cmd.exe`), use the caret shown in the original request:

```bat
python generate_boundary_labels.py ^
  --tiles D:\da\dataset_tiles ^
  --shapefile D:\da\Aapakudal_Farm_Digitization\Aapakudal_Farm_Digitization.shp ^
  --output D:\da\ground_truth_dataset ^
  --boundary-thickness 5 ^
  --workers 6
```

The generator:

- reads the WGS 84 shapefile and reprojects it to each tile's CRS;
- uses the GeoPandas spatial index for each tile;
- clips parcel fill geometry while avoiding false boundaries at tile edges;
- writes binary parcel and boundary masks plus stable uint16 parcel IDs;
- preserves dimensions, CRS, affine transform and pixel alignment;
- writes red-boundary JPEG overlays for visual inspection;
- creates `dataset_statistics.json`.

The resulting layout is:

```text
ground_truth_dataset/
  parcel_labels/{train,val,test}/*.tif
  boundary_labels/{train,val,test}/*.tif
  instance_labels/{train,val,test}/*.tif
  visualizations/*_overlay.jpg
  dataset_statistics.json
```

QA images require substantial extra disk space. Add `--skip-qa` if they are not
needed, or `--qa-every 20` to create one representative overlay per 20 tiles.
Add `--overwrite` when intentionally regenerating existing outputs.

If a previous run stopped, simply run the same command again. Completed masks
are reused unless `--overwrite` is supplied. When recovering from a full disk,
remove or move the regenerable `visualizations` folder first and resume with
`--skip-qa`.

## 3. Train the boundary model

Defaults are suitable for a typical 6 GB RTX 4050: 512-pixel training crops,
batch size 4, full-resolution validation with batch size 1, and CUDA AMP.

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

If CUDA runs out of memory, reduce `--batch-size` to 2 or 1. The training run
produces `best_checkpoint.pt`, `last_checkpoint.pt`, `training_history.csv`,
`training_config.json`, and `test_metrics.json`.

ImageNet encoder weights are downloaded on the first run. Use
`--encoder-weights none` on an offline machine. Continue an interrupted run with:

```powershell
python train_from_digitized_boundaries.py `
  --tiles "D:\da\dataset_tiles" `
  --labels "D:\da\ground_truth_dataset" `
  --output "D:\da\boundary_training_run" `
  --epochs 50 `
  --resume "D:\da\boundary_training_run\last_checkpoint.pt"
```

## 4. Run the Streamlit inference app

The inference app accepts an RGB GeoTIFF, TIFF, JPG, or PNG upload and exports
georeferenced boundary masks, parcel masks, instance masks, and GeoJSON.

```powershell
pip install -r requirements-streamlit-inference.txt
streamlit run streamlit_predict_bund_parcel.py
```

The default model is `paired_boundary_training_run/best_checkpoint.pt`. This
large checkpoint is stored with Git LFS. After cloning, install Git LFS and
fetch the model before starting the app:

```powershell
git lfs install
git lfs pull
```

The large source raster, generated datasets, and training outputs are ignored
by Git because they are not required to run the upload-based inference app.
