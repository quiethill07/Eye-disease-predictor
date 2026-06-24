"""
Notebook-friendly example for the Kaggle FIVES layout shown in the screenshot.

Paste pieces of this into Kaggle cells, or upload the project files and run:

!python train.py --fives_root "/kaggle/input/<dataset-folder>/<fives-root-folder>" --output_dir /kaggle/working/outputs
!python test.py --fives_root "/kaggle/input/<dataset-folder>/<fives-root-folder>" --checkpoint /kaggle/working/outputs/checkpoints/best_model.pth --output_dir /kaggle/working/test_outputs
"""

import os
from glob import glob


def find_fives_root(base_input: str = "/kaggle/input") -> str:
    candidates = []
    for dataset_dir in glob(os.path.join(base_input, "*")):
        for inner in glob(os.path.join(dataset_dir, "*")):
            train_original = os.path.join(inner, "train", "Original")
            test_original = os.path.join(inner, "test", "Original")
            if os.path.isdir(train_original) and os.path.isdir(test_original):
                candidates.append(inner)

    if not candidates:
        raise FileNotFoundError("Could not auto-detect a FIVES root under /kaggle/input.")

    print("Detected FIVES root:", candidates[0])
    return candidates[0]


if __name__ == "__main__":
    root = find_fives_root()
    print(root)
