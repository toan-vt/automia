import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MIA evaluation for evolved code blocks from successful runtime runs "
            "and aggregate results into a CSV."
        )
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Path to the eval template script (e.g. pubmed/mia-eval.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Top-level output directory containing the 'runtime' subdirectory.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to CSV file for aggregated results (default: <output-dir>/eval_results.csv).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="test",
        help="Comma-separated list of splits to evaluate (default: test).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k results to evaluate (default: 5).",
    )
    return parser.parse_args()


def find_successful_runs(output_dir: Path, top_k: int = 0) -> List[Path]:
    runtime_root = output_dir / "runtime"
    if not runtime_root.is_dir():
        LOGGER.warning("Runtime directory does not exist: %s", runtime_root)
        return []

    run_dirs: List[Path] = []

    if top_k == 0:
        for child in sorted(runtime_root.iterdir()):
            if not child.is_dir():
                continue
            mia_run = child / "mia_run.py"
            mia_results = child / "mia-results.json"
            if mia_run.is_file() and mia_results.is_file():
                run_dirs.append(child)

        LOGGER.info("Found %d successful runtime runs under %s", len(run_dirs), runtime_root)
        return run_dirs

    # get top-k results by auc_score
    auc_scores = []
    for child in sorted(runtime_root.iterdir(), key=lambda x: x.name):
        if not child.is_dir():
            continue
        mia_results = child / "mia-results.json"
        if mia_results.is_file():
            with mia_results.open("r", encoding="utf-8") as f:
                auc_scores.append(json.load(f)["auc_score"])
    auc_scores.sort(reverse=True)

    # get top k auc scores
    top_k_auc_scores = auc_scores[:top_k]
    for child in sorted(runtime_root.iterdir(), key=lambda x: x.name):
        if not child.is_dir():
            continue
        mia_results = child / "mia-results.json"
        if mia_results.is_file():
            with mia_results.open("r", encoding="utf-8") as f:
                auc_score = json.load(f)["auc_score"]
                if auc_score not in top_k_auc_scores:
                    continue
                run_dirs.append(child)

    return run_dirs


def extract_evolving_block(mia_run_path: Path) -> str:
    start_token = "Evolving code starts here"
    end_token = "Evolving code ends here"

    in_block = False
    block_lines: List[str] = []

    with mia_run_path.open("r", encoding="utf-8") as f:
        for line in f:
            if start_token in line:
                in_block = True
                continue
            if end_token in line:
                if in_block:
                    break
            if in_block:
                block_lines.append(line.rstrip("\n"))

    code_block = "\n".join(block_lines).rstrip("\n")
    if code_block:
        code_block += "\n"
    return code_block


def render_eval_script(template_path: Path, code_block: str) -> str:
    start_token = "Evolving code starts here"
    end_token = "Evolving code ends here"

    with template_path.open("r", encoding="utf-8") as f:
        template_lines = f.readlines()

    out_lines: List[str] = []
    in_block = False
    for line in template_lines:
        if start_token in line:
            out_lines.append(line.rstrip("\n") + "\n")
            if code_block:
                out_lines.append(code_block if code_block.endswith("\n") else code_block + "\n")
            in_block = True
            continue
        if end_token in line:
            in_block = False
            out_lines.append(line)
            continue
        if not in_block:
            out_lines.append(line)

    return "".join(out_lines)


def append_result_row(csv_path: Path, row: Dict[str, Any]) -> None:
    fieldnames = ["run_id", "split", "auc_score", "tpr_1_score", "tpr_5_score", "combined_score"]
    csv_exists = csv_path.is_file()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def load_completed_pairs(csv_path: Path) -> Set[Tuple[str, str]]:
    """
    Load already-completed (run_id, split) pairs from an existing CSV, if any.
    """
    completed: Set[Tuple[str, str]] = set()
    if not csv_path.is_file():
        return completed

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id")
            split = row.get("split")
            if run_id and split:
                completed.add((str(run_id).strip(), str(split).strip()))
    return completed


def run_eval_for_run(
    run_dir: Path,
    template_path: Path,
    splits: List[str],
    csv_path: Path,
    completed_pairs: Set[Tuple[str, str]],
) -> None:
    run_id = run_dir.name
    mia_run_path = run_dir / "mia_run.py"

    pending_splits = [split for split in splits if (run_id, split) not in completed_pairs]
    if not pending_splits:
        LOGGER.info("Skipping run %s (all requested splits already in CSV).", run_id)
        return

    LOGGER.info("Processing run %s", run_id)
    code_block = extract_evolving_block(mia_run_path)
    if not code_block:
        LOGGER.warning("No evolving code block found in %s, skipping.", mia_run_path)
        return

    rendered_source = render_eval_script(template_path, code_block)
    eval_script_path = run_dir / "mia_eval_tmp.py"
    eval_script_path.write_text(rendered_source, encoding="utf-8")

    for split in pending_splits:
        eval_output_dir = run_dir / "eval" / split
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(eval_script_path),
            "--output-dir",
            str(eval_output_dir),
            "--split",
            split,
        ]
        LOGGER.info("Running eval for run=%s split=%s", run_id, split)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            LOGGER.error("Eval failed for run=%s split=%s: %s", run_id, split, e)
            continue

        results_path = eval_output_dir / "mia-results.json"
        if not results_path.is_file():
            LOGGER.error(
                "mia-results.json not found for run=%s split=%s at %s",
                run_id,
                split,
                results_path,
            )
            continue

        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)

        row: Dict[str, Any] = {
            "run_id": run_id,
            "split": split,
            "auc_score": results.get("auc_score"),
            "tpr_1_score": results.get("tpr_1_score"),
            "tpr_5_score": results.get("tpr_5_score"),
            "combined_score": results.get("combined_score"),
        }
        append_result_row(csv_path, row)
        completed_pairs.add((run_id, split))


def main() -> None:
    setup_logging()
    args = parse_args()

    template_path = Path(args.template)
    output_dir = Path(args.output_dir)

    if not template_path.is_file():
        raise FileNotFoundError(f"Template script not found: {template_path}")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    csv_path = Path(args.csv_path) if args.csv_path is not None else output_dir / "eval_results.csv"
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No splits specified.")

    completed_pairs = load_completed_pairs(csv_path)
    if completed_pairs:
        LOGGER.info("Loaded %d completed run/split pairs from %s", len(completed_pairs), csv_path)
    run_dirs = find_successful_runs(output_dir, top_k=args.top_k)
    if not run_dirs:
        LOGGER.info("No successful runtime runs found under %s", output_dir)
        return

    for run_dir in run_dirs:
        run_eval_for_run(run_dir, template_path, splits, csv_path, completed_pairs)


if __name__ == "__main__":
    main()
