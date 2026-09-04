dataset=fives
input_size=256
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [FIVES_ROOT] --val_ratio 0.2
python val.py --name ${dataset}_UKAN --split val --val_ratio 0.2
python val.py --name ${dataset}_UKAN --split test

# Segmentation-guided classification (after segmentation model is trained)

# 1) Ensure train/test label CSVs exist in your dataset package.
#
# 2) Generate split-safe segmentation masks for classifier (soft maps by default).
# python generate_classifier_masks.py --name ${dataset}_UKAN --output_dir outputs --data_dir [FIVES_ROOT] --save_root classifier_masks --mask_type prob
#
# 3) Train classifier (train/val only).
# python train_classifier.py --name ${dataset}_effb0_seg_guided --data_dir [FIVES_ROOT] --train_labels_csv [TRAIN_CSV] --train_mask_dir classifier_masks/${dataset}_UKAN/train/prob_maps
#
# 4) Evaluate classifier on val/test.
# python val_classifier.py --name ${dataset}_effb0_seg_guided --split val
# python val_classifier.py --name ${dataset}_effb0_seg_guided --split test --test_labels_csv [TEST_CSV] --test_mask_dir classifier_masks/${dataset}_UKAN/test/prob_maps






