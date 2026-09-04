import argparse
import importlib
import os
from pathlib import Path

import pandas as pd


REQUIRED_PY_PKGS = [
    'torch',
    'torchvision',
    'albumentations',
    'cv2',
    'numpy',
    'pandas',
    'yaml',
    'sklearn',
    'matplotlib',
]
OPTIONAL_PY_PKGS = [
    'medpy',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Preflight checks for segmentation + classification pipeline.')
    parser.add_argument('--fives_root', required=True, type=str, help='FIVES root directory')
    parser.add_argument('--train_csv', default='', type=str, help='train labels CSV path (optional)')
    parser.add_argument('--test_csv', default='', type=str, help='test labels CSV path (optional)')
    parser.add_argument('--train_mask_dir', default='', type=str, help='classifier train mask dir (optional)')
    parser.add_argument('--test_mask_dir', default='', type=str, help='classifier test mask dir (optional)')
    parser.add_argument('--seg_output_dir', default='outputs', type=str, help='segmentation output root')
    parser.add_argument('--seg_name', default='', type=str, help='segmentation experiment name (optional)')
    parser.add_argument('--strict_masks', default=False, action='store_true',
                        help='fail if train/test mask dirs are missing')
    return parser.parse_args()


def check_packages():
    missing = []
    for pkg in REQUIRED_PY_PKGS:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    return missing


def discover_csvs(root):
    train_csv = None
    test_csv = None
    for p in sorted(Path(root).rglob('*.csv')):
        name = p.name.lower()
        if train_csv is None and name == 'train_labels.csv':
            train_csv = str(p)
        if test_csv is None and name == 'test_labels.csv':
            test_csv = str(p)
    return train_csv, test_csv


def check_csv_schema(csv_path, split_name):
    if not csv_path or not os.path.isfile(csv_path):
        return [f'{split_name} CSV missing: {csv_path}']
    df = pd.read_csv(csv_path)
    needed = {'img_id', 'label'}
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        return [f'{split_name} CSV missing columns: {missing_cols}']
    if len(df) == 0:
        return [f'{split_name} CSV has no rows']
    return []


def count_files(dir_path, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    if not dir_path or not os.path.isdir(dir_path):
        return 0
    n = 0
    for name in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, name)) and os.path.splitext(name)[1].lower() in exts:
            n += 1
    return n


def main():
    args = parse_args()
    errors = []
    warnings = []

    fives_root = args.fives_root
    train_img_dir = os.path.join(fives_root, 'train', 'Original')
    train_gt_dir = os.path.join(fives_root, 'train', 'Ground truth')
    test_img_dir = os.path.join(fives_root, 'test', 'Original')
    test_gt_dir = os.path.join(fives_root, 'test', 'Ground truth')

    for d in [train_img_dir, train_gt_dir, test_img_dir, test_gt_dir]:
        if not os.path.isdir(d):
            errors.append(f'Missing required directory: {d}')

    missing_pkgs = check_packages()
    if missing_pkgs:
        errors.append(f'Missing Python packages: {missing_pkgs}')
    optional_missing = []
    for pkg in OPTIONAL_PY_PKGS:
        try:
            importlib.import_module(pkg)
        except Exception:
            optional_missing.append(pkg)
    if optional_missing:
        warnings.append(
            f'Optional packages not found: {optional_missing}. '
            f'Pipeline still runs; HD/HD95 metrics may be unavailable.'
        )

    auto_train_csv, auto_test_csv = discover_csvs(fives_root)
    train_csv = args.train_csv if args.train_csv else auto_train_csv
    test_csv = args.test_csv if args.test_csv else auto_test_csv

    errors.extend(check_csv_schema(train_csv, 'train'))
    errors.extend(check_csv_schema(test_csv, 'test'))

    if args.seg_name:
        seg_ckpt = os.path.join(args.seg_output_dir, args.seg_name, 'model.pth')
        seg_cfg = os.path.join(args.seg_output_dir, args.seg_name, 'config.yml')
        if not os.path.isfile(seg_ckpt):
            warnings.append(f'Segmentation best checkpoint not found yet: {seg_ckpt}')
        if not os.path.isfile(seg_cfg):
            warnings.append(f'Segmentation config not found yet: {seg_cfg}')

    train_mask_count = count_files(args.train_mask_dir, exts=('.png',))
    test_mask_count = count_files(args.test_mask_dir, exts=('.png',))
    if args.train_mask_dir and train_mask_count == 0:
        msg = f'No train masks found in: {args.train_mask_dir}'
        if args.strict_masks:
            errors.append(msg)
        else:
            warnings.append(msg)
    if args.test_mask_dir and test_mask_count == 0:
        msg = f'No test masks found in: {args.test_mask_dir}'
        if args.strict_masks:
            errors.append(msg)
        else:
            warnings.append(msg)

    print('--- Preflight Summary ---')
    print(f'FIVES root: {fives_root}')
    print(f'Train CSV: {train_csv}')
    print(f'Test CSV : {test_csv}')
    if args.train_mask_dir:
        print(f'Train mask files: {train_mask_count}')
    if args.test_mask_dir:
        print(f'Test mask files : {test_mask_count}')

    if warnings:
        print('\nWarnings:')
        for w in warnings:
            print(f'- {w}')

    if errors:
        print('\nErrors:')
        for e in errors:
            print(f'- {e}')
        raise SystemExit(1)

    print('\nAll required checks passed.')


if __name__ == '__main__':
    main()
