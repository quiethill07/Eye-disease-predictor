import os

import cv2
import numpy as np
import torch
import torch.utils.data


def to_unit_float(arr):
    orig_dtype = arr.dtype
    arr = arr.astype('float32')
    # Only scale raw image tensors; do not rescale already-normalized floats.
    if orig_dtype == np.uint8:
        arr = arr / 255.0
    return arr


def discover_image_ids(img_dir, exts):
    img_ids = []
    for entry in sorted(os.listdir(img_dir)):
        lower_entry = entry.lower()
        for ext in exts:
            if lower_entry.endswith(ext.lower()):
                img_ids.append(os.path.splitext(entry)[0])
                break
    return img_ids


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, mask_in_class_subdir=True, id_prefix=''):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.mask_in_class_subdir = mask_in_class_subdir
        self.id_prefix = id_prefix

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path)
        # cv2.imread returns None for a missing/corrupt file; without this check
        # the failure surfaces much later as an opaque NoneType error.
        if img is None:
            raise FileNotFoundError(f'Failed to read image: {img_path}')

        mask = []
        for i in range(self.num_classes):
            if self.mask_in_class_subdir:
                mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
            else:
                mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
            mask_i = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_i is None:
                raise FileNotFoundError(f'Failed to read mask: {mask_path}')
            mask.append(mask_i[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = to_unit_float(img)
        img = img.transpose(2, 0, 1)
        mask = to_unit_float(mask)
        mask = mask.transpose(2, 0, 1)

        if mask.max()<1:
            mask[mask>0] = 1.0

        sample_id = f'{self.id_prefix}_{img_id}' if self.id_prefix else img_id
        return img, mask, {'img_id': img_id, 'sample_id': sample_id}


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, img_ext, transform=None, id_to_ext=None, id_prefix='', id_to_path=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform
        self.id_to_ext = id_to_ext if id_to_ext is not None else {}
        self.id_prefix = id_prefix
        self.id_to_path = id_to_path if id_to_path is not None else {}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        if img_id in self.id_to_path:
            img_path = self.id_to_path[img_id]
        else:
            img_ext = self.id_to_ext.get(img_id, self.img_ext)
            img_path = os.path.join(self.img_dir, img_id + img_ext)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'Failed to read image: {img_path}')

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = to_unit_float(img)
        img = img.transpose(2, 0, 1)

        sample_id = f'{self.id_prefix}_{img_id}' if self.id_prefix else img_id
        return img, {'img_id': img_id, 'sample_id': sample_id, 'img_path': img_path}
