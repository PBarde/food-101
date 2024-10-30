from pathlib import Path
import json
import numpy as np

import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description="Script to plot training results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--save_path", type=str, default='./run/',
                        help="Path to save the run")

    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    save_path = Path(args.save_path)
    with open(save_path / "train_metrics.json") as f:
        train_metrics = json.load(f)

    with open(save_path / "valid_metrics.json") as f:
        valid_metrics = json.load(f)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(train_metrics["loss"], label="train")
    ax[0].plot(valid_metrics["loss"], label="valid")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(100.*np.array(train_metrics["accuracy"]), label="train")
    ax[1].plot(100.*np.array(valid_metrics["accuracy"]), label="valid")
    ax[1].set_title("Accuracy (%)")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    plt.savefig(save_path / "train_plot.png")
    plt.show()

    print(f"Saved plot to {save_path / 'train_plot.png'}")