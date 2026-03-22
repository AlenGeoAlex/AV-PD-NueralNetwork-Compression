"""
Usage:
    python run_eval.py \
        --model    runs/train/yolo11s_kd/weights/best.pt \
        --test-set /path/to/Dataset/merged_dataset \
        --output   results/yolo11s_kd \
        --name     "YOLO11s + KD"

    # Minimal (auto-name from model filename)
    python run_eval.py -m yolo11s.pt -t /path/to/dataset -o results/yolo11s

    # With all options
    python run_eval.py \
        -m  yolo11x.pt \
        -t  /path/to/Dataset/merged_dataset \
        -o  results/yolo11x \
        -n  "YOLO11x Teacher" \
        --conf 0.001 \
        --iou  0.6 \
        --img-size 640 \
        --device cuda \
        --latency-runs 200 \
        --latency-warmup 50
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root or this directory
sys.path.insert(0, str(Path(__file__).parent))

from .evaluator import run_evaluation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a YOLO model on the pedestrian detection test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-m", "--model",     required=True, help="Path to .pt weights file")
    p.add_argument("-t", "--test-set",  required=True, help="Root of merged dataset")
    p.add_argument("-o", "--output",    required=True, help="Output directory")
    p.add_argument("-n", "--name",      default=None,  help="Display name for this model")
    p.add_argument("--conf",            type=float, default=0.001,
                   help="Confidence threshold for inference (default: 0.001 for full PR curve)")
    p.add_argument("--iou",             type=float, default=0.6,  help="NMS IoU threshold")
    p.add_argument("--img-size",        type=int,   default=640,  help="Inference image size")
    p.add_argument("--device",          default=None, help="'cpu' or 'cuda' (auto-detect if omitted)")
    p.add_argument("--latency-runs",    type=int,   default=200,  help="Latency measurement runs")
    p.add_argument("--latency-warmup",  type=int,   default=50,   help="Latency warmup runs")
    p.add_argument("-q", "--quiet",     action="store_true", help="Suppress progress output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results = run_evaluation(
        model_path     = args.model,
        test_set_path  = args.test_set,
        output_dir     = args.output,
        model_name     = args.name,
        conf_threshold = args.conf,
        iou_nms        = args.iou,
        img_size       = args.img_size,
        device         = args.device,
        latency_warmup = args.latency_warmup,
        latency_runs   = args.latency_runs,
        verbose        = not args.quiet,
    )

    summary = results["summary"]
    print("\n── Results Summary ────────────────────────────────────")
    print(f"  mAP50:     {summary['map50']*100:.2f}%")
    print(f"  mAP50-95:  {summary['map50_95']*100:.2f}%")
    print(f"  Precision: {summary['precision']*100:.2f}%")
    print(f"  Recall:    {summary['recall']*100:.2f}%")
    print(f"  F1:        {summary['f1']*100:.2f}%")
    print(f"  FPS:       {results['latency']['fps']}")
    print(f"  Size (MB): {results['model_info']['size_mb']}")
    print(f"\n  comparison.json → {args.output}/comparison.json")
    print("───────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()