# Seg_UKAN_V3

`Seg_UKAN_V3` is a hybrid retinal image analysis project built by combining:

- the stronger segmentation direction from `Seg_UKAN` (V1)
- the cleaner classification and fusion pipeline from `Seg_UKAN_V2` (V2)

The current best variant uses `cbam_se` attention in the segmentation branch.

## Current Best Results

### Segmentation
- Test IoU: `0.3133`
- Test Dice: `0.4714`

### Classification
- Test Accuracy: `0.5600`
- Test Macro F1: `0.5588`

## Project Idea

This version keeps the newer V2 training and testing pipeline while replacing the segmentation backbone with the stronger V1-style UKAN/KAN-based segmentation model. It is intended as the main working version for future improvements.

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

## Notes

- Do not upload the dataset to GitHub.
- Do not upload training outputs, checkpoints, or test result folders unless you specifically want to publish them.
- Keep `Seg_UKAN_V3` as the main version for future work.
