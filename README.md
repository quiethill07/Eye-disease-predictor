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

## License

This project is released under the MIT License — see [LICENSE](LICENSE).
