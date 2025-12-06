"""
Evaluate FER-based interview analyzer against the FER2013 dataset.

This script measures:
1. Face detection success (when bypassing detection with full-frame boxes).
2. Emotion classification accuracy versus FER2013 labels.
3. Interview confidence score statistics from InterviewEmotionAnalyzer.

Usage:
    python tests/evaluate_fer2013.py --data-dir ./data --split test --max-samples 1000
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from fer import FER

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interview_analyzer import InterviewEmotionAnalyzer


LABEL_TO_NAME = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
NAME_TO_LABEL = {name: idx for idx, name in LABEL_TO_NAME.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FER model on FER2013")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where FER2013 will be downloaded/cached",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "private", "public"],
        help="Dataset split/usage to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of samples to evaluate (0 = entire split)",
    )
    parser.add_argument(
        "--upsample-size",
        type=int,
        default=96,
        help="Resize each FER2013 frame to this square size before inference",
    )
    parser.add_argument(
        "--mtcnn",
        action="store_true",
        help="Use MTCNN for detection (slower, but occasionally more accurate)",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=20,
        help="Minimum face size for Haar detector (ignored when using manual boxes)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print running metrics every N samples",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Path to fer2013.csv (defaults to <data-dir>/fer2013.csv)",
    )
    return parser.parse_args()


SPLIT_TO_USAGE = {
    "train": "Training",
    "test": "PublicTest",
    "public": "PublicTest",
    "private": "PrivateTest",
}


class FER2013Dataset:
    """Minimal dataset wrapper around fer2013.csv with Usage column."""

    def __init__(self, csv_path: Path, split: str):
        usage = SPLIT_TO_USAGE.get(split)
        if usage is None:
            raise ValueError(f"Unknown split '{split}'")

        df = pd.read_csv(csv_path)
        if "Usage" not in df.columns:
            raise ValueError(f"'Usage' column not found in {csv_path}")

        subset = df[df["Usage"].str.lower() == usage.lower()]
        if subset.empty:
            raise ValueError(f"No rows with Usage='{usage}' in {csv_path}")

        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []

        for _, row in subset.iterrows():
            pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
            image = torch.from_numpy(pixels.reshape(48, 48)).unsqueeze(0).float() / 255.0
            self.images.append(image)
            self.labels.append(int(row["emotion"]))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], self.labels[idx]


def load_dataset(data_dir: str, split: str, csv_file: str = None):
    csv_path = Path(csv_file) if csv_file else Path(data_dir) / "fer2013.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"FER2013 CSV not found at {csv_path}")
    return FER2013Dataset(csv_path, split)


def to_numpy_image(tensor_image, target_size: int) -> np.ndarray:
    """Convert Torch tensor (1 x 48 x 48) to BGR uint8 image resized to target_size."""
    image = tensor_image.squeeze().numpy()  # (48, 48)
    image = (image * 255.0).astype("uint8")
    image = cv2.merge([image, image, image])  # grayscale -> 3 channels
    if target_size and target_size != image.shape[0]:
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return image


def evaluate(
    args: argparse.Namespace, dataset, fer_model: FER, analyzer: InterviewEmotionAnalyzer
) -> Dict[str, float]:
    total = 0
    evaluated = 0
    correct = 0
    detection_failures = 0
    per_class_total = Counter()
    per_class_correct = Counter()
    confusion = defaultdict(Counter)

    for idx in range(len(dataset)):
        if args.max_samples and evaluated >= args.max_samples:
            break

        tensor_image, label = dataset[idx]
        img = to_numpy_image(tensor_image, args.upsample_size)
        h, w, _ = img.shape

        try:
            results = fer_model.detect_emotions(
                img, face_rectangles=[(0, 0, w, h)]
            )
        except Exception as exc:
            detection_failures += 1
            continue

        if not results:
            detection_failures += 1
            continue

        emotions = results[0]["emotions"]
        predicted_name = max(emotions, key=emotions.get)
        predicted_label = NAME_TO_LABEL.get(predicted_name, -1)

        analyzer.add_emotion_data(emotions)

        total += 1
        per_class_total[label] += 1
        confusion[label][predicted_label] += 1

        if predicted_label == label:
            correct += 1
            per_class_correct[label] += 1

        evaluated += 1

        if evaluated % args.log_every == 0:
            interim_acc = correct / total if total else 0.0
            print(
                f"[{evaluated} samples] accuracy={interim_acc:.3f}, "
                f"detection_failures={detection_failures}"
            )

    accuracy = correct / total if total else 0.0
    per_class_accuracy = {
        LABEL_TO_NAME[k]: (per_class_correct[k] / per_class_total[k])
        if per_class_total[k]
        else 0.0
        for k in sorted(LABEL_TO_NAME.keys())
    }

    summary = {
        "samples_total": evaluated,
        "samples_evaluated": total,
        "detection_failures": detection_failures,
        "overall_accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion": {LABEL_TO_NAME[k]: dict(confusion[k]) for k in confusion},
        "avg_confidence": float(np.mean(analyzer.confidence_history))
        if analyzer.confidence_history
        else 0.0,
        "confidence_std": float(np.std(analyzer.confidence_history))
        if analyzer.confidence_history
        else 0.0,
    }

    return summary


def pretty_print(summary: Dict[str, float]):
    print("\n=== FER2013 Evaluation Summary ===")
    print(f"Samples requested: {summary['samples_total']}")
    print(f"Samples evaluated (with detections): {summary['samples_evaluated']}")
    print(f"Detection failures: {summary['detection_failures']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.3f}")
    print("\nPer-class accuracy:")
    for emotion, acc in summary["per_class_accuracy"].items():
        print(f"  {emotion:>8}: {acc:.3f}")
    print(
        f"\nInterview analyzer confidence mean={summary['avg_confidence']:.3f}, "
        f"std={summary['confidence_std']:.3f}"
    )


def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    dataset = load_dataset(args.data_dir, args.split, args.csv_file)
    fer_model = FER(
        mtcnn=args.mtcnn,
        min_face_size=args.min_face_size,
    )
    analyzer = InterviewEmotionAnalyzer(window_size=30)

    summary = evaluate(args, dataset, fer_model, analyzer)
    pretty_print(summary)


if __name__ == "__main__":
    main()

