import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import random
import json


# =========================== Evolving code starts here ======================================================
### <CODE-BLOCK>
# =========================== Evolving code ends here ========================================================

def evaluate_attack(y_true: np.ndarray, y_signal: np.ndarray) -> Dict[str, float]:
    """
    Evaluate MIA attack performance given binary labels and a scalar signal.
    """
    auc = roc_auc_score(y_true, y_signal)
    if auc < 0.5:
        auc = 1 - auc
        y_signal = -y_signal

    fpr, tpr, _ = roc_curve(y_true, y_signal)
    tpr_1_score = tpr[np.where(fpr < 0.01)[0][-1]] if np.any(fpr < 0.01) else 0.0
    tpr_5_score = tpr[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else 0.0
    combined_score = (tpr_1_score + tpr_5_score) / 2

    logging.info(f"AUC score: {auc:.3f}")
    logging.info(f"TPR@1% FPR: {tpr_1_score:.3f}")
    logging.info(f"TPR@5% FPR: {tpr_5_score:.3f}")
    logging.info(f"Combined score: {combined_score:.3f}")
    logging.info(f"to latex: {auc:.3f} & {tpr_1_score:.3f} & {tpr_5_score:.3f}")

    return {
        "auc_score": float(round(auc, 3)),
        "tpr_1_score": float(round(tpr_1_score, 3)),
        "tpr_5_score": float(round(tpr_5_score, 3)),
        "combined_score": float(round(combined_score, 3)),
    }


def load_inference_tensors(tensors_dir: Path, split: str) -> List[Dict[str, torch.Tensor]]:
    """
    Load logits, input_ids and labels from safetensors files.

    Expects files produced by openlvlm-mia/main.py:
        - img_logits.safetensors
        - labels.safetensors

    Returns a dict mapping sample_id -> {logits, label}.
    """
    logits_path = tensors_dir / "img_logits.safetensors"
    labels_path = tensors_dir / "labels.safetensors"

    if not logits_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Expected logits/labels safetensors in {tensors_dir}, "
            "but one or more files are missing."
        )

    logits_tensors = load_safetensors(str(logits_path))
    labels_tensors = load_safetensors(str(labels_path))

    # Use keys from logits as canonical set, e.g. "logits_00000"
    sample_dict: Dict[str, Dict[str, torch.Tensor]] = {}
    for key in sorted(logits_tensors.keys()):
        if not key.startswith("img_logits_"):
            continue
        sample_id = key.split("_", 2)[2]
        logits = logits_tensors[key]
        label_key = f"label_{sample_id}"
        if label_key not in labels_tensors:
            logging.warning(f"Missing label for sample {sample_id}, skipping.")
            continue
        sample_dict[sample_id] = {
            "logits": logits.clone().detach().to(dtype=torch.float32, device="cpu"),
            "label": labels_tensors[label_key].clone().detach().to(dtype=torch.int64, device="cpu"),
        }
    
    # get 50% of the samples (randomly with seed 42)
    random.seed(42)
    random.shuffle(list(sample_dict.values()))
    if split == "train":
        return list(sample_dict.values())[:len(sample_dict) // 2]
    elif split == "test":
        return list(sample_dict.values())[len(sample_dict) // 2:]
    else:
        raise ValueError(f"Invalid split: {split}")    

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run membership inference attacks from saved safetensor data "
            "(logits, input_ids, labels) produced by openlvlm-mia."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="None",
        help="Output directory for results (default: same as input file directory)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split to use for inference (default: train, choices: train, test)"
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = output_dir / "mia-from-data.log"
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    tensors_dir = Path("/lustre/orion/lrn087/proj-shared/toan/vr-mia/src/mia_agents_dual/examples/vlm/openlvlm-mia/outputs/llava_dalle/inference_tensors")
    logging.info(f"Loading inference tensors from {tensors_dir}")
    samples = load_inference_tensors(tensors_dir, split=args.split)
    logging.info(f"Loaded {len(samples)} usable samples")

    # Build labels and signals
    labels: List[int] = []
    signals: List[float] = []

    for data in tqdm(samples, desc="Computing MIA signals"):
        label_tensor = data["label"]
        label = int(label_tensor.item()) if label_tensor.numel() == 1 else int(label_tensor.view(-1)[0].item())
        logits = data["logits"]

        signal = compute_mia_signal(logits)

        labels.append(label)
        signals.append(signal)

    y_true = np.asarray(labels, dtype=int)
    y_signal = np.asarray(signals, dtype=float)

    num_members = int(np.sum(y_true == 1))
    num_nonmembers = int(np.sum(y_true == 0))
    logging.info(
        f"Samples: {len(y_true)} total "
        f"({num_members} members, {num_nonmembers} non-members)"
    )

    logging.info(f"Evaluating MIA")
    results = evaluate_attack(y_true, y_signal)

    results_path = output_dir / "mia-results.json"
    with open(results_path, "w") as f:
        json.dump(results, f)

    logging.info(f"***Saved output directory: {output_dir}")

if __name__ == "__main__":
    main()

