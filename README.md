# Seg-UKAN: Retinal Vessel Segmentation with U-KAN + Segmentation-Guided Classification

This repository implements **U-KAN**, a U-Net-style architecture with Kolmogorov–Arnold Network (KAN) layers in the bottleneck, for retinal vessel segmentation on the **FIVES** fundus image dataset. It also includes an optional downstream pipeline that trains an EfficientNet-B0 classifier guided by the predicted segmentation masks.

## Contents

- `archs.py` — U-KAN model definition (KAN-based bottleneck blocks, optional SE/CBAM attention, encoder/decoder).
- `kan.py` — KAN linear layer implementation used by the bottleneck.
- `dataset.py` — `Dataset` (train/val, image + mask pairs) and `InferenceDataset` (image-only inference) classes.
- `losses.py` — Loss functions (e.g. `BCEDiceLoss`).
- `metrics.py` — Segmentation metrics: IoU, Dice, Hausdorff distance/HD95, recall, specificity, precision.
- `utils.py` — Small shared helpers (`AverageMeter`, `str2bool`).
- `train.py` — Train the U-KAN segmentation model.
- `val.py` — Evaluate a trained segmentation model on the val or test split.
- `infer.py` — Run inference with a trained segmentation model on an arbitrary image folder, saving binary masks and/or probability maps.
- `pipeline_preflight.py` — Sanity-checks the dataset layout, required Python packages, and CSV schemas before running the full pipeline.
- `generate_classifier_masks.py` — Generates split-safe segmentation masks (for train and test images) to be used as guidance input for the downstream classifier.
- `classifier_model.py` — Segmentation-guided EfficientNet-B0 classifier architectures (single-input mask-attention and dual-input fusion variants).
- `classifier_dataset.py` — Dataset class for the segmentation-guided classifier.
- `train_classifier.py` — Train the segmentation-guided classifier.
- `val_classifier.py` — Evaluate the classifier on val/test splits.
- `scripts.sh` — Example end-to-end command sequence for the full pipeline.
- `environment.yml` / `requirements.txt` — Dependency specifications.

## Installation

Python 3.8+ is recommended (the original `environment.yml` targets Python 3.6, but `requirements.txt` is compatible with newer versions).

```bash
pip install -r requirements.txt

# Install PyTorch separately according to your CUDA/CPU setup, e.g.:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Dataset

The scripts are written around the [FIVES](https://doi.org/10.6084/m9.figshare.19688169) fundus vessel segmentation dataset, expected in this layout:

```
<FIVES_ROOT>/
├── train/
│   ├── Original/        # training images (.png)
│   └── Ground truth/    # training masks (.png)
└── test/
    ├── Original/        # test images (.png)
    └── Ground truth/    # test masks (.png)
```

A generic `<dataset>/images` + `<dataset>/masks/<class_idx>` layout is also supported for datasets other than FIVES (see `dataset.py` for details).

## Usage

### 1. Train the segmentation model

```bash
python train.py --arch UKAN --dataset fives --input_w 256 --input_h 256 \
    --name fives_UKAN --data_dir [FIVES_ROOT] --val_ratio 0.2
```

### 2. Evaluate the segmentation model

```bash
python val.py --name fives_UKAN --split val --val_ratio 0.2
python val.py --name fives_UKAN --split test
```

### 3. Run inference on new images

```bash
python infer.py --name fives_UKAN --image_dir /path/to/images --save_dir vessel_predictions
```

### 4. (Optional) Segmentation-guided classification

After the segmentation model is trained:

```bash
# Ensure train/test label CSVs (img_id,label) exist in your dataset package.

# Generate split-safe segmentation masks for the classifier (soft probability maps by default).
python generate_classifier_masks.py --name fives_UKAN --output_dir outputs \
    --data_dir [FIVES_ROOT] --save_root classifier_masks --mask_type prob

# Train the classifier (train/val only).
python train_classifier.py --name fives_effb0_seg_guided --data_dir [FIVES_ROOT] \
    --train_labels_csv [TRAIN_CSV] --train_mask_dir classifier_masks/fives_UKAN/train/prob_maps

# Evaluate the classifier on val/test.
python val_classifier.py --name fives_effb0_seg_guided --split val
python val_classifier.py --name fives_effb0_seg_guided --split test \
    --test_labels_csv [TEST_CSV] --test_mask_dir classifier_masks/fives_UKAN/test/prob_maps
```

See `scripts.sh` for the full example sequence, and `pipeline_preflight.py` to validate your dataset/environment before running the pipeline end to end:

```bash
python pipeline_preflight.py --fives_root [FIVES_ROOT]
```

## Outputs

Training runs are saved under `outputs/<name>/` (segmentation) and `outputs_cls/<name>/` (classifier), each containing the run config (`config.yml`), logs (`log.csv`), TensorBoard event files, and model checkpoints (best/last/periodic).

## Streamlit Demo

This repository includes a Streamlit app in `streamlit_app.py` for the full inference pipeline:

- retinal vessel segmentation
- segmentation-guided disease classification

Run it locally with:

```bash
pip install -r requirements_streamlit.txt
python -m streamlit run streamlit_app.py
```

Before starting the app, place the trained artifacts in these paths:

- `app_artifacts/segmentation/config.yml`
- `app_artifacts/segmentation/model.pth`
- `app_artifacts/classification/classifier_config.yml`
- `app_artifacts/classification/best_classifier.pth`

Optional class labels file:

- `app_artifacts/classification/class_names.yml`
- `.streamlit/secrets.toml`

Example:

```yaml
class_names:
  - Normal
  - Diabetic Retinopathy
  - Glaucoma
  - Cataract
```

After the artifacts are in place, the app only needs a fundus image upload from the end user.

If the full `requirements.txt` fails on modern Windows/Python versions, use `requirements_streamlit.txt` for the demo app. It contains only the packages needed for inference and the Streamlit UI.

For Streamlit Community Cloud deployment, keep checkpoint files out of this repository and configure artifact access through Streamlit secrets.

If you intentionally want public artifact URLs, create a local `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example` and set:

```toml
[artifact_urls]
segmentation_config_url = "https://example.com/config.yml"
segmentation_model_url = "https://example.com/model.pth"
classification_config_url = "https://example.com/classifier_config.yml"
classification_model_url = "https://example.com/best_classifier.pth"
```

The app will download missing artifacts into `app_artifacts/` at startup.

Recommended setup for a public Streamlit app with private model files:

1. Create a separate private GitHub repository just for the four deployed artifacts.
2. Add only these files there:
   - `segmentation/config.yml`
   - `segmentation/model.pth`
   - `classification/classifier_config.yml`
   - `classification/best_classifier.pth`
3. Create a fine-grained GitHub token with read-only access to that private repository.
4. In Streamlit Community Cloud, open your app settings and paste the following into `Secrets`.

Use this secret format:

```toml
[github_artifacts]
owner = "your-github-username"
repo = "your-private-artifact-repo"
branch = "main"
token = "github_fine_grained_token_with_contents_read"
segmentation_config_path = "segmentation/config.yml"
segmentation_model_path = "segmentation/model.pth"
classification_config_path = "classification/classifier_config.yml"
classification_model_path = "classification/best_classifier.pth"
```

The app will use the GitHub Contents API with authenticated raw downloads for those files.

This lets the Streamlit app stay public while your model files remain private. The token is stored only in Streamlit secrets and is not committed to GitHub.

## License

No standalone license file is currently included with this project.
