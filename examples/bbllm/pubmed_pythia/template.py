import pickle as pkl
import numpy as np
import argparse
from tqdm import tqdm
import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.metrics import roc_auc_score, roc_curve
import nltk
from unidecode import unidecode
import random
from multiprocessing import Pool
import math

# =========================== Evolving code starts here ======================================================
### <CODE-BLOCK>
# =========================== Evolving code ends here ========================================================

def _process_sample(sample: Dict[str, Any]) -> float:
    _sample = {
        "original_text": sample["original_text"],
        "prefix": sample["context"],
        "ground_truth_suffix": sample["ground_truth_suffix"],
        "suffix_generations": sample["suffix_generations"],
    }
    return compute_mia_signal(_sample)

def compute_mia_signals(generated_samples: List[Dict[str, Any]]) -> List[float]:
    num_workers = 16
    with Pool(processes=num_workers) as pool:
        mia_signals = list(tqdm(
            pool.imap(_process_sample, generated_samples),
            total=len(generated_samples),
            desc="Computing MIA signals",
        ))
    return mia_signals


def evaluate_attack(
    y_true: np.ndarray,
    y_signal: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate MIA attack performance.

    Args:
        y_true: Ground truth labels (1=member, 0=non-member)
        y_signal: MIA signals

    Returns:
        Dictionary with metrics
    """
    auc = roc_auc_score(y_true, y_signal)
    if auc < 0.5:
        auc = 1 - auc
        y_signal = -y_signal

    fpr, tpr, thresholds = roc_curve(y_true, y_signal)
    tpr_1_score = tpr[np.where(fpr < 0.01)[0][-1]] if np.any(fpr < 0.01) else 0.0
    tpr_5_score = tpr[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else 0.0
    combined_score = auc

    print(f"AUC score: {auc:.3f}")
    print(f"TPR@1% FPR: {tpr_1_score:.3f}")
    print(f"TPR@5% FPR: {tpr_5_score:.3f}")
    print(f"Combined score: {combined_score:.3f}")

    return {
        "auc_score": round(auc, 3),
        "tpr_1_score": round(tpr_1_score, 3),
        "tpr_5_score": round(tpr_5_score, 3),
        "combined_score": round(combined_score, 3),
    }

def select_subset(saved_data: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    all_data = saved_data.copy()
    random.seed(42)
    random.shuffle(all_data)

    num_samples = len(all_data)
    idx_split = num_samples // 2

    if split == "train":
        return all_data[:idx_split]
    elif split == "test":
        return all_data[idx_split:]
    else:
        raise ValueError(f"Invalid split: {split}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MIA signals from pre-generated completion data."
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
        help="Split to use (default: train)"
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    all_results = []
    # for length in [32, 64, 128]:
    for subset in ["pubmed_central"]:
        # Setup logging
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

        input_path = Path(f"tmp/generated_data/mimir_{subset}_pythia14.pkl")
        logging.info(f"Loading data from {input_path}")
        with open(input_path, "rb") as f:
            saved_data = pkl.load(f)
        
        saved_data = select_subset(saved_data, args.split)
        logging.info(f"Loaded {len(saved_data)} samples of {args.split} set")

        # Extract labels before computing signals
        labels = [sample_data["label"] for sample_data in saved_data]

        # Compute MIA signals
        logging.info("Computing MIA signals...")
        mia_signals = compute_mia_signals(saved_data)

        y_true = np.array(labels, dtype=int)
        y_signal = np.array(mia_signals, dtype=float)

        num_members = np.sum(y_true == 1)
        num_nonmembers = np.sum(y_true == 0)
        logging.info(f"Samples: {len(y_true)} total ({num_members} members, {num_nonmembers} non-members)")

        # Evaluate attack
        logging.info(f"=== Results for {subset} ===")
        results = evaluate_attack(y_true, y_signal)
        all_results.append(results)

    avg_results = {
        "auc_score": round(np.mean([result["auc_score"] for result in all_results]), 3),
        "tpr_1_score": round(np.mean([result["tpr_1_score"] for result in all_results]), 3),
        "tpr_5_score": round(np.mean([result["tpr_5_score"] for result in all_results]), 3),
        "combined_score": round(np.mean([result["combined_score"] for result in all_results]), 3)
    }
    logger.info(f"Average results: {avg_results}")
    # Save results
    results_path = output_dir / "mia-results.json"
    with open(results_path, "w") as f:
        json.dump(avg_results, f)

    print(f"***Saved output directory: {output_dir}")
