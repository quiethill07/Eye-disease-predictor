import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


LABEL_MAP = {"amd": 0, "dr": 1, "glaucoma": 2, "normal": 3}
INV_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
FIVES_SUFFIX_MAP = {"a": "amd", "d": "dr", "g": "glaucoma", "n": "normal"}
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _normalize_label(label: str) -> int:
    key = str(label).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unknown label '{label}'. Expected one of {list(LABEL_MAP.keys())}.")
    return LABEL_MAP[key]


def build_transforms(image_size: int = 512, train: bool = True) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.CLAHE(clip_limit=2.0, p=0.2),
                A.Affine(
                    translate_percent=(-0.03, 0.03),
                    scale=(0.95, 1.05),
                    rotate=(-8, 8),
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.3,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


@dataclass
class DatasetConfig:
    csv_path: str = ""
    image_root: str = ""
    mask_root: str = ""
    fives_root: str = ""
    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True
    val_size: float = 0.2
    seed: int = 42
    weighted_sampling: bool = False


class RetinalMultiTaskDataset(Dataset):
    """
    Expected CSV columns:
    image_path, mask_path, label, split
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: str = "",
        mask_root: str = "",
        transforms: Optional[A.Compose] = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_root = image_root
        self.mask_root = mask_root
        self.transforms = transforms

        required_cols = {"image_path", "mask_path", "label"}
        missing = required_cols - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        self.dataframe["label_idx"] = self.dataframe["label"].map(_normalize_label)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_path(self, path: str, root: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(root, path)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image_path = self._resolve_path(row["image_path"], self.image_root)
        mask_path = self._resolve_path(row["mask_path"], self.mask_root)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].float().unsqueeze(0)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(int(row["label_idx"]), dtype=torch.long),
            "image_path": image_path,
            "mask_path": mask_path,
        }


def load_splits(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataframe = pd.read_csv(csv_path)
    if "split" not in dataframe.columns:
        raise ValueError("CSV must contain a 'split' column with train/val/test values.")

    train_df = dataframe[dataframe["split"].str.lower() == "train"].reset_index(drop=True)
    val_df = dataframe[dataframe["split"].str.lower() == "val"].reset_index(drop=True)
    test_df = dataframe[dataframe["split"].str.lower() == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def infer_label_from_filename(file_path: str) -> str:
    stem = os.path.splitext(os.path.basename(file_path))[0]
    suffix = stem.split("_")[-1].strip().lower()
    if suffix not in FIVES_SUFFIX_MAP:
        raise ValueError(
            f"Could not infer label from filename '{file_path}'. "
            "Expected a FIVES-style suffix such as _A, _D, _G, or _N."
        )
    return FIVES_SUFFIX_MAP[suffix]


def build_fives_dataframe(fives_root: str, val_size: float = 0.2, seed: int = 42) -> pd.DataFrame:
    records = []
    for split_name in ["train", "test"]:
        image_dir = os.path.join(fives_root, split_name, "Original")
        mask_dir = os.path.join(fives_root, split_name, "Ground truth")
        if not os.path.isdir(mask_dir):
            mask_dir = os.path.join(fives_root, split_name, "Ground Truth")

        image_paths = sorted(glob(os.path.join(image_dir, "*")))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        for image_path in image_paths:
            extension = os.path.splitext(image_path)[1].lower()
            if extension not in VALID_IMAGE_EXTENSIONS:
                continue
            file_name = os.path.basename(image_path)
            mask_path = os.path.join(mask_dir, file_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask for {image_path}. Expected {mask_path}")

            records.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": infer_label_from_filename(file_name),
                    "source_split": split_name,
                }
            )

    dataframe = pd.DataFrame(records)
    train_source = dataframe[dataframe["source_split"] == "train"].reset_index(drop=True)
    test_df = dataframe[dataframe["source_split"] == "test"].copy()
    test_df["split"] = "test"

    train_idx, val_idx = train_test_split(
        train_source.index,
        test_size=val_size,
        stratify=train_source["label"],
        random_state=seed,
    )
    train_df = train_source.loc[train_idx].copy()
    val_df = train_source.loc[val_idx].copy()
    train_df["split"] = "train"
    val_df["split"] = "val"

    combined = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    return combined[["image_path", "mask_path", "label", "split"]]


def create_dataloaders(config: DatasetConfig) -> Dict[str, DataLoader]:
    if config.csv_path:
        train_df, val_df, test_df = load_splits(config.csv_path)
    elif config.fives_root:
        dataframe = build_fives_dataframe(config.fives_root, val_size=config.val_size, seed=config.seed)
        train_df = dataframe[dataframe["split"] == "train"].reset_index(drop=True)
        val_df = dataframe[dataframe["split"] == "val"].reset_index(drop=True)
        test_df = dataframe[dataframe["split"] == "test"].reset_index(drop=True)
    else:
        raise ValueError("Provide either csv_path or fives_root in DatasetConfig.")

    train_dataset = RetinalMultiTaskDataset(
        dataframe=train_df,
        image_root=config.image_root,
        mask_root=config.mask_root,
        transforms=build_transforms(config.image_size, train=True),
    )
    val_dataset = RetinalMultiTaskDataset(
        dataframe=val_df,
        image_root=config.image_root,
        mask_root=config.mask_root,
        transforms=build_transforms(config.image_size, train=False),
    )
    test_dataset = RetinalMultiTaskDataset(
        dataframe=test_df,
        image_root=config.image_root,
        mask_root=config.mask_root,
        transforms=build_transforms(config.image_size, train=False),
    )

    train_loader_kwargs = {
        "batch_size": config.batch_size,
        "drop_last": True,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.weighted_sampling:
        label_counts = train_df["label"].str.lower().value_counts().to_dict()
        sample_weights = train_df["label"].str.lower().map(lambda label: 1.0 / max(label_counts[label], 1)).to_numpy()
        train_loader_kwargs["sampler"] = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader_kwargs["shuffle"] = False
    else:
        train_loader_kwargs["shuffle"] = True

    return {
        "train": DataLoader(
            train_dataset,
            **train_loader_kwargs,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
    }
