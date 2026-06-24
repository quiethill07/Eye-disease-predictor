# Seg_UKAN_V3

This is a retinal image analysis system that does two connected jobs:

- segment blood vessels from fundus images
- classify eye disease using both the original retinal image and vessel-aware information

The current best variant uses `cbam_se` attention in the segmentation branch.

## Project Idea

The core idea is that vessel structure carries clinically useful signals, so instead of treating classification as a plain image problem, we first learn a vessel-focused representation and then use that to improve disease prediction. The project is essentially a hybrid multi-stage retinal diagnosis pipeline where segmentation supports classification, and attention-enhanced feature learning helps the model focus on medically relevant patterns.

## Main Files

- `train_segmentation.py`: train vessel segmentation
- `test_segmentation.py`: evaluate segmentation
- `train_classification.py`: train disease classification
- `test_classification.py`: evaluate classification
- `model_segmentation.py`: V3 compatibility wrapper around the V1 segmentation model
- `fusion_model.py`: V2-style classification fusion model
- `archs.py`: V1 segmentation architecture
- `kan.py`: KAN implementation used by the V1 segmentation branch

## Dataset Layout

Expected dataset layout:

```text
FIVES/
  FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/
    train/
      Original/
      Ground truth/
    test/
      Original/
      Ground truth/
```

## Example Commands

### Segmentation training

```powershell
python train_segmentation.py --fives_root "C:\path\to\FIVES\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation" --output_dir "C:\path\to\v3_seg_run" --epochs 20 --batch_size 2 --image_size 128 --num_workers 0 --val_size 0.2 --seed 42 --disable_amp --attention_mode cbam_se
```

### Segmentation test

```powershell
python test_segmentation.py --fives_root "C:\path\to\FIVES\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation" --checkpoint "C:\path\to\v3_seg_run\checkpoints\best_segmentation_model.pth" --output_dir "C:\path\to\v3_seg_test" --image_size 128 --batch_size 2 --num_workers 0 --val_size 0.2 --seed 42
```

### Classification training

```powershell
python train_classification.py --fives_root "C:\path\to\FIVES\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation" --output_dir "C:\path\to\v3_cls_run" --use_ground_truth_masks --epochs 20 --batch_size 4 --image_size 128 --num_workers 0 --val_size 0.2 --seed 42 --backbone_name efficientnet_b0 --no_pretrained --disable_amp --segmentation_attention_mode cbam_se
```

### Classification test

```powershell
python test_classification.py --fives_root "C:\path\to\FIVES\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation" --checkpoint "C:\path\to\v3_cls_run\checkpoints\best_classification_model.pth" --output_dir "C:\path\to\v3_cls_test" --use_ground_truth_masks --image_size 128 --batch_size 4 --num_workers 0 --val_size 0.2 --seed 42
```
