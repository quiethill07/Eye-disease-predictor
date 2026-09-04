from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

import albumentations as A
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import yaml

import archs
from classifier_model import DualInputSegGuidedEfficientNet, SegGuidedEfficientNetB0


ARTIFACT_ROOT = Path("app_artifacts")
SEG_CONFIG_PATH = ARTIFACT_ROOT / "segmentation" / "config.yml"
SEG_MODEL_PATH = ARTIFACT_ROOT / "segmentation" / "model.pth"
CLS_CONFIG_PATH = ARTIFACT_ROOT / "classification" / "classifier_config.yml"
CLS_MODEL_PATH = ARTIFACT_ROOT / "classification" / "best_classifier.pth"
CLASS_NAMES_PATH = ARTIFACT_ROOT / "classification" / "class_names.yml"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
DEFAULT_CLASS_NAMES = ["Normal", "AMD", "DR", "Glaucoma"]
ARTIFACT_FILES = {
    "segmentation_config_url": SEG_CONFIG_PATH,
    "segmentation_model_url": SEG_MODEL_PATH,
    "classification_config_url": CLS_CONFIG_PATH,
    "classification_model_url": CLS_MODEL_PATH,
}


def load_yaml_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping in {path}")
    return data


def torch_load_compat(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format. Expected a state dict or checkpoint dict.")

    cleaned = {}
    for key, value in checkpoint.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def read_class_names(num_classes: int) -> List[str]:
    if CLASS_NAMES_PATH.is_file():
        data = load_yaml_file(CLASS_NAMES_PATH)
        names = data.get("class_names", [])
        if isinstance(names, list) and len(names) == num_classes:
            return [str(name) for name in names]
    if num_classes == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES
    return [f"Class {idx}" for idx in range(num_classes)]


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image


def build_segmentation_model(config: dict, checkpoint_path: Path):
    model = archs.__dict__[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config["deep_supervision"],
        embed_dims=config["input_list"],
        no_kan=config.get("no_kan", False),
        attention_mode=config.get("attention_mode", "none"),
    )
    state_dict = extract_state_dict(torch_load_compat(checkpoint_path, map_location="cpu"))
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Segmentation checkpoint does not match the config.\n"
            f"Missing keys: {list(incompatible.missing_keys)}\n"
            f"Unexpected keys: {list(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model


def build_classifier_model(config: dict, checkpoint_path: Path):
    num_classes = int(config["num_classes"])
    model_type = config.get("model_type", "single")
    dropout = float(config.get("dropout", 0.2))

    if model_type == "dual_input":
        model = DualInputSegGuidedEfficientNet(
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
        )
    else:
        model = SegGuidedEfficientNetB0(
            num_classes=num_classes,
            pretrained=False,
        )

    state_dict = extract_state_dict(torch_load_compat(checkpoint_path, map_location="cpu"))
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Classifier checkpoint does not match the classifier config.\n"
            f"Missing keys: {list(incompatible.missing_keys)}\n"
            f"Unexpected keys: {list(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model


def get_artifact_urls() -> Dict[str, str]:
    urls: Dict[str, str] = {}
    if hasattr(st, "secrets") and "artifact_urls" in st.secrets:
        secret_urls = st.secrets["artifact_urls"]
        for key in ARTIFACT_FILES:
            value = secret_urls.get(key, "")
            if value:
                urls[key] = str(value)
    return urls


def ensure_remote_artifacts() -> List[str]:
    missing_local = [path for path in ARTIFACT_FILES.values() if not path.is_file()]
    if not missing_local:
        return []

    urls = get_artifact_urls()
    still_missing = []
    for key, dest_path in ARTIFACT_FILES.items():
        if dest_path.is_file():
            continue
        source_url = urls.get(key, "").strip()
        if not source_url:
            still_missing.append(str(dest_path))
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(source_url) as response:
            data = response.read()
        dest_path.write_bytes(data)

    return [str(path) for path in ARTIFACT_FILES.values() if not path.is_file()]


@st.cache_resource(show_spinner=False)
def load_pipeline():
    missing = ensure_remote_artifacts()
    if missing:
        raise FileNotFoundError(
            "Missing required model artifact files:\n- " + "\n- ".join(missing)
        )

    seg_config = load_yaml_file(SEG_CONFIG_PATH)
    cls_config = load_yaml_file(CLS_CONFIG_PATH)
    seg_model = build_segmentation_model(seg_config, SEG_MODEL_PATH)
    cls_model = build_classifier_model(cls_config, CLS_MODEL_PATH)
    class_names = read_class_names(int(cls_config["num_classes"]))
    return {
        "seg_config": seg_config,
        "cls_config": cls_config,
        "seg_model": seg_model,
        "cls_model": cls_model,
        "class_names": class_names,
    }


def run_segmentation(model, config: dict, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    transform = A.Compose(
        [
            A.Resize(int(config["input_h"]), int(config["input_w"])),
            A.Normalize(),
        ]
    )
    augmented = transform(image=image_bgr)
    image = augmented["image"].astype("float32").transpose(2, 0, 1)
    tensor = torch.from_numpy(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

    height, width = image_bgr.shape[:2]
    prob_map = cv2.resize(probs, (width, height), interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_map >= 0.5).astype(np.uint8)
    return prob_map, binary_mask


def run_classification(model, config: dict, image_bgr: np.ndarray, seg_prob_map: np.ndarray) -> np.ndarray:
    transform = A.Compose(
        [
            A.Resize(int(config.get("input_h", 512)), int(config.get("input_w", 512))),
            A.Normalize(),
        ]
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_u8 = np.clip(seg_prob_map * 255.0, 0, 255).astype(np.uint8)
    augmented = transform(image=image_rgb, mask=mask_u8)

    image = augmented["image"].astype("float32").transpose(2, 0, 1)
    mask = augmented["mask"].astype("float32")
    if mask.max() > 1.0:
        mask = mask / 255.0
    if mask.ndim == 2:
        mask = mask[..., None]
    mask = np.clip(mask, 0.0, 1.0).transpose(2, 0, 1)

    image_tensor = torch.from_numpy(image).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor, mask_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def render_setup_help():
    st.error("Model artifacts are missing, so the app cannot produce predictions yet.")
    st.markdown(
        """
Create these files inside the repository before deployment:

- `app_artifacts/segmentation/config.yml`
- `app_artifacts/segmentation/model.pth`
- `app_artifacts/classification/classifier_config.yml`
- `app_artifacts/classification/best_classifier.pth`

Optional:

- `app_artifacts/classification/class_names.yml`
- Streamlit secrets with artifact download URLs

`class_names.yml` format:

```yaml
class_names:
  - Normal
  - Diabetic Retinopathy
  - Glaucoma
  - Cataract
```

Do not commit large model files to Git unless you intentionally want them public. A better deployment pattern is to place the weights on a public file host and download them into these paths during app startup, or add them manually on the deployment machine.
        """
    )


def main():
    st.set_page_config(page_title="Eye Disease Predictor", layout="wide")
    st.markdown(
        """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 136, 0, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 194, 255, 0.10), transparent 24%),
            linear-gradient(180deg, #09111b 0%, #05080e 55%, #020409 100%);
    }
    .hero-card {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(18, 25, 38, 0.92), rgba(7, 12, 21, 0.92));
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
        margin-bottom: 1.1rem;
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        color: #7dd3fc;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.45rem;
        line-height: 1.1;
    }
    .hero-copy {
        color: #cbd5e1;
        font-size: 1rem;
        max-width: 52rem;
    }
    .prediction-card {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(12, 18, 30, 0.95), rgba(18, 44, 54, 0.92));
        border: 1px solid rgba(125, 211, 252, 0.16);
        margin-bottom: 0.8rem;
    }
    .prediction-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #94a3b8;
    }
    .prediction-name {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0.2rem 0;
    }
    .prediction-confidence {
        color: #7dd3fc;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="hero-card">
    <div class="hero-kicker">Retinal AI Workflow</div>
    <div class="hero-title">Eye Disease Predictor</div>
    <div class="hero-copy">
        Upload one fundus image to run the full pipeline: vessel segmentation first,
        then disease classification guided by the predicted vessel map.
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    try:
        pipeline = load_pipeline()
    except Exception:
        render_setup_help()
        return

    image_file = st.file_uploader("Upload a fundus image", type=[ext.lstrip(".") for ext in IMAGE_EXTS])
    if not image_file:
        st.info("Upload a fundus image to begin.")
        return

    image_bgr = decode_uploaded_image(image_file)
    with st.spinner("Running segmentation and classification..."):
        seg_prob_map, seg_binary_mask = run_segmentation(
            pipeline["seg_model"],
            pipeline["seg_config"],
            image_bgr,
        )
        class_probs = run_classification(
            pipeline["cls_model"],
            pipeline["cls_config"],
            image_bgr,
            seg_prob_map,
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prob_uint8 = np.clip(seg_prob_map * 255.0, 0, 255).astype(np.uint8)
    binary_uint8 = (seg_binary_mask * 255).astype(np.uint8)

    pred_idx = int(np.argmax(class_probs))
    pred_label = pipeline["class_names"][pred_idx]
    pred_conf = float(class_probs[pred_idx])

    top_two = np.argsort(class_probs)[::-1][:2]
    runner_up = pipeline["class_names"][int(top_two[1])] if len(top_two) > 1 else "N/A"
    runner_up_conf = float(class_probs[int(top_two[1])]) if len(top_two) > 1 else 0.0

    pred_col, summary_col = st.columns([1.25, 1.75])
    with pred_col:
        st.markdown(
            f"""
<div class="prediction-card">
    <div class="prediction-label">Primary Prediction</div>
    <div class="prediction-name">{pred_label}</div>
    <div class="prediction-confidence">Confidence: {pred_conf:.2%}</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with summary_col:
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Confidence", f"{pred_conf:.2%}")
        stat2.metric("Runner-up", runner_up)
        stat3.metric("Runner-up score", f"{runner_up_conf:.2%}")

    col1, col2, col3 = st.columns(3)
    col1.image(image_rgb, caption="Original fundus image", use_container_width=True)
    col2.image(prob_uint8, caption="Segmentation probability map", use_container_width=True, clamp=True)
    col3.image(binary_uint8, caption="Binary vessel mask", use_container_width=True, clamp=True)

    results_table: List[Dict[str, str]] = []
    for index, probability in enumerate(class_probs):
        results_table.append(
            {
                "class": pipeline["class_names"][index],
                "probability": round(float(probability), 4),
                "confidence": f"{float(probability):.2%}",
            }
        )
    results_table.sort(key=lambda row: row["probability"], reverse=True)

    st.subheader("Class probabilities")
    st.dataframe(results_table, use_container_width=True, hide_index=True)

    with st.expander("Loaded artifact summary"):
        st.json(
            {
                "segmentation_config": str(SEG_CONFIG_PATH),
                "segmentation_checkpoint": str(SEG_MODEL_PATH),
                "classification_config": str(CLS_CONFIG_PATH),
                "classification_checkpoint": str(CLS_MODEL_PATH),
                "classes": pipeline["class_names"],
                "remote_urls_configured": sorted(list(get_artifact_urls().keys())),
            }
        )


if __name__ == "__main__":
    main()
