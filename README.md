# Seg-UKAN: Retinal Vessel Segmentation with U-KAN and Segmentation-Guided Classification

This repository contains the improved Version 2 project for retinal fundus analysis. It combines:

- retinal vessel segmentation using a modified U-KAN pipeline
- disease classification using an EfficientNet-B0 model guided by the predicted vessel mask
- a Streamlit web app for end-to-end inference from a single uploaded fundus image

## Live Demo

- Streamlit app: https://eye-disease-predictor-aravindkumar.streamlit.app/

The deployed app is intended for retinal fundus photographs only. It includes input screening to reject obvious non-fundus images before prediction.

## Class Labels

The classifier uses these class IDs:

- `0` = `Normal`
- `1` = `AMD`
- `2` = `DR`
- `3` = `Glaucoma`

## Repository Contents

- `archs.py` - U-KAN segmentation architecture with optional attention modes
- `kan.py` - Kolmogorov-Arnold Network layers used in the segmentation model
- `dataset.py` - segmentation dataset loaders
- `losses.py` - segmentation loss functions
- `metrics.py` - segmentation evaluation metrics
- `train.py` - segmentation training
- `val.py` - segmentation validation and testing
- `infer.py` - segmentation inference on image folders
- `generate_classifier_masks.py` - generates segmentation-derived masks for classifier training and evaluation
- `classifier_model.py` - segmentation-guided EfficientNet-B0 classifier models
- `classifier_dataset.py` - classifier dataset loader
- `train_classifier.py` - classifier training
- `val_classifier.py` - classifier validation and testing
- `pipeline_preflight.py` - dataset and environment checks before running the full pipeline
- `streamlit_app.py` - deployed end-to-end web app
- `requirements.txt` - deployment-safe Python dependencies for Streamlit inference
- `requirements_streamlit.txt` - minimal local Streamlit app dependency set
- `environment.training.yml` - legacy training environment reference

## Dataset Layout

The main scripts were built around the FIVES dataset. Expected layout:

```text
<FIVES_ROOT>/
  train/
    Original/
    Ground truth/
  test/
    Original/
    Ground truth/
```

Other image and mask layouts may also work depending on the script and configuration.

## Training and Evaluation

### 1. Train segmentation

```bash
python train.py --arch UKAN --dataset fives --input_w 256 --input_h 256 --name fives_UKAN --data_dir [FIVES_ROOT] --val_ratio 0.2
```

### 2. Evaluate segmentation

```bash
python val.py --name fives_UKAN --split val --val_ratio 0.2
python val.py --name fives_UKAN --split test
```

### 3. Run segmentation inference

```bash
python infer.py --name fives_UKAN --image_dir /path/to/images --save_dir vessel_predictions
```

### 4. Train and evaluate classification

After segmentation training:

```bash
python generate_classifier_masks.py --name fives_UKAN --output_dir outputs --data_dir [FIVES_ROOT] --save_root classifier_masks --mask_type prob

python train_classifier.py --name fives_effb0_seg_guided --data_dir [FIVES_ROOT] --train_labels_csv [TRAIN_CSV] --train_mask_dir classifier_masks/fives_UKAN/train/prob_maps

python val_classifier.py --name fives_effb0_seg_guided --split val
python val_classifier.py --name fives_effb0_seg_guided --split test --test_labels_csv [TEST_CSV] --test_mask_dir classifier_masks/fives_UKAN/test/prob_maps
```

You can also run:

```bash
python pipeline_preflight.py --fives_root [FIVES_ROOT]
```

to verify the dataset and environment before running the full pipeline.

## Streamlit App

Run locally:

```bash
pip install -r requirements_streamlit.txt
python -m streamlit run streamlit_app.py
```

The app expects these artifact paths:

- `app_artifacts/segmentation/config.yml`
- `app_artifacts/segmentation/model.pth`
- `app_artifacts/classification/classifier_config.yml`
- `app_artifacts/classification/best_classifier.pth`

Optional:

- `app_artifacts/classification/class_names.yml`

Example `class_names.yml`:

```yaml
class_names:
  - Normal
  - AMD
  - DR
  - Glaucoma
```

## Deployment Setup

As of September 4, 2026, the recommended deployment pattern for this project is:

1. Keep the Streamlit app code in this public repository.
2. Keep the four model/config artifacts in a separate private GitHub repository.
3. Store the GitHub access token only in Streamlit secrets.
4. Deploy the app on Streamlit Community Cloud with Python `3.12`.

Required private artifact files:

- `segmentation/config.yml`
- `segmentation/model.pth`
- `classification/classifier_config.yml`
- `classification/best_classifier.pth`

Streamlit secrets format:

```toml
[github_artifacts]
owner = "quiethill07"
repo = "Eye-disease-predictor-artifacts-private-repo"
branch = "main"
token = "YOUR_FINE_GRAINED_READ_ONLY_TOKEN"
segmentation_config_path = "segmentation/config.yml"
segmentation_model_path = "segmentation/model.pth"
classification_config_path = "classification/classifier_config.yml"
classification_model_path = "classification/best_classifier.pth"
```

The app downloads missing artifacts from the private GitHub repository at startup.

## Notes

- Do not commit model weights, datasets, or secrets to this repository.
- `environment.training.yml` is kept only as a training reference and is not used for Streamlit deployment.
- The Streamlit app is meant for demonstration and inference workflow purposes, not as a substitute for clinical diagnosis.

## License

No standalone license file is included with this repository.
