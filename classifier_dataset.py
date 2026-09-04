import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data


def _to_float_image(arr):
    orig_dtype = arr.dtype
    arr = arr.astype(np.float32)
    if orig_dtype == np.uint8:
        arr = arr / 255.0
    return arr


def _to_float_mask(arr):
    orig_dtype = arr.dtype
    arr = arr.astype(np.float32)
    if orig_dtype == np.uint8:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _find_existing_path(base_dir, stem, exts):
    for ext in exts:
        path = os.path.join(base_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def _resolve_mask_path(mask_dir, img_id, mask_ext, split_prefixes=None):
    direct = os.path.join(mask_dir, img_id + mask_ext)
    if os.path.isfile(direct):
        return direct

    if split_prefixes is None:
        split_prefixes = []

    for prefix in split_prefixes:
        path = os.path.join(mask_dir, f'{prefix}{img_id}{mask_ext}')
        if os.path.isfile(path):
            return path

    # fallback: try loose matching if naming was altered
    for name in os.listdir(mask_dir):
        if not name.lower().endswith(mask_ext.lower()):
            continue
        stem = os.path.splitext(name)[0]
        if stem == img_id or stem.endswith('_' + img_id):
            return os.path.join(mask_dir, name)
    return None


def load_label_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'img_id' not in df.columns:
        if 'image_id' in df.columns:
            df = df.rename(columns={'image_id': 'img_id'})
        else:
            raise ValueError(f'Label CSV must contain "img_id" or "image_id": {csv_path}')
    if 'label' not in df.columns:
        raise ValueError(f'Label CSV must contain "label" column: {csv_path}')
    keep_cols = ['img_id', 'label']
    for optional in ('image_relpath', 'mask_relpath'):
        if optional in df.columns:
            keep_cols.append(optional)
    df = df[keep_cols].copy()
    df['img_id'] = df['img_id'].astype(str)
    df['label'] = df['label'].astype(int)
    return df


def infer_num_classes(df, strict=True):
    """
    Derive the number of classes from a label dataframe.

    Labels are expected to be contiguous integers starting at 0. Using
    ``label.max() + 1`` silently produces a wrong head size whenever that
    assumption is violated (e.g. labels are 1-indexed, or a class is absent
    from the CSV), so validate it explicitly instead.
    """
    labels = sorted(set(int(v) for v in df['label'].tolist()))
    if len(labels) == 0:
        raise ValueError('Label CSV contains no rows; cannot infer num_classes.')

    expected = list(range(len(labels)))
    if labels != expected:
        message = (
            f'Labels are not contiguous 0-indexed integers. '
            f'Found {labels}, expected {expected}. '
            f'Re-map your labels, or pass --num_classes explicitly.'
        )
        if strict:
            raise ValueError(message)
        import warnings
        warnings.warn(message)
        return int(max(labels)) + 1

    return len(labels)


class SegGuidedClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, records, image_dir, mask_dir, img_ext='.png', mask_ext='.png', transform=None, split_prefixes=None,
                 image_root=None, mask_root=None):
        self.records = records.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.split_prefixes = split_prefixes if split_prefixes is not None else []
        self.image_root = image_root
        self.mask_root = mask_root
        self.image_ext_candidates = [img_ext, '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img_id = row['img_id']
        label = int(row['label'])

        image_path = None
        image_relpath = row.get('image_relpath', '')
        if isinstance(image_relpath, str) and image_relpath and self.image_root is not None:
            candidate = os.path.join(self.image_root, image_relpath)
            if os.path.isfile(candidate):
                image_path = candidate
        if image_path is None:
            image_path = _find_existing_path(self.image_dir, img_id, self.image_ext_candidates)
        if image_path is None:
            raise FileNotFoundError(f'Image not found for id "{img_id}" under {self.image_dir}')

        mask_path = None
        mask_relpath = row.get('mask_relpath', '')
        if isinstance(mask_relpath, str) and mask_relpath and self.mask_root is not None:
            candidate = os.path.join(self.mask_root, mask_relpath)
            if os.path.isfile(candidate):
                mask_path = candidate
        if mask_path is None:
            mask_path = _resolve_mask_path(
                self.mask_dir,
                img_id,
                self.mask_ext,
                split_prefixes=self.split_prefixes,
            )
        if mask_path is None:
            raise FileNotFoundError(f'Mask not found for id "{img_id}" under {self.mask_dir}')

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f'Failed reading image: {image_path}')
        if mask is None:
            raise RuntimeError(f'Failed reading mask: {mask_path}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = _to_float_image(image)
        mask = _to_float_mask(mask)

        if mask.ndim == 2:
            mask = mask[..., None]
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        return torch.from_numpy(image), torch.from_numpy(mask), torch.tensor(label, dtype=torch.long), img_id
