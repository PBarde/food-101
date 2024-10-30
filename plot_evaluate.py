from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd

def get_argparser():
    parser = argparse.ArgumentParser(description="Script to plot evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--save_path", type=str, default='./run/',
                        help="Path to save the run")

    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()

    save_path = Path(args.save_path)
    with open(save_path / "class_metrics.json", 'r', encoding='utf-8') as f:
        class_metrics = json.load(f)

    print(class_metrics)

    # we do a barplot of the class-wise metrics
    class_accuracies = class_metrics["accuracy"]
    class_precision = class_metrics["precision"]
    class_recall = class_metrics["recall"]
    class_f1 = class_metrics["f1_score"]
    clas = class_metrics["class_names"]


    fig, ax = plt.subplots()
    bar_width = 0.2
    x = np.arange(len(class_accuracies))
    ax.bar(x, class_accuracies, bar_width, label='Class Accuracy')
    ax.bar(x + bar_width, class_precision, bar_width, label='Class Precision')
    ax.bar(x + 2*bar_width, class_recall, bar_width, label='Class Recall')
    ax.bar(x + 3*bar_width, class_f1, bar_width, label='Class F1')
    ax.set_xticks(x + 2*bar_width)
    ax.set_xticklabels(clas, rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path / "class_metrics.png")
    print(f"Saved plot to {save_path / 'class_metrics.png'}")

    # we do a boxplot of the best and worst classes
    sorted_acc_idx = np.argsort(class_accuracies)
    sorted_acc = np.array(class_accuracies)[sorted_acc_idx]
    sorted_clas = np.array(clas)[sorted_acc_idx]
    sorted_precision = np.array(class_precision)[sorted_acc_idx]
    sorted_recall = np.array(class_recall)[sorted_acc_idx]
    sorted_f1 = np.array(class_f1)[sorted_acc_idx]

    n_to_plot = 10
    idxes = np.arange(n_to_plot)
    idxes = np.flip(idxes)
    
    # plot the n_to_plot worst classes
    fig, ax = plt.subplots()
    bar_width = 0.2
    x = np.arange(n_to_plot)
    ax.bar(x, sorted_acc[idxes], bar_width, label='Class Accuracy')
    ax.bar(x + bar_width, sorted_precision[idxes], bar_width, label='Class Precision')
    ax.bar(x + 2*bar_width, sorted_recall[idxes], bar_width, label='Class Recall')
    ax.bar(x + 3*bar_width, sorted_f1[idxes], bar_width, label='Class F1')
    ax.set_xticks(x + 2*bar_width)
    ax.set_xticklabels(sorted_clas[idxes], rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path / "worst_classes.png")
    print(f"Saved plot to {save_path / 'worst_classes.png'}")

    # plot the n_to_plot best classes
    idxes = np.arange(len(class_accuracies) - n_to_plot, len(class_accuracies))
    idxes = np.flip(idxes)

    fig, ax = plt.subplots()
    bar_width = 0.2
    x = np.arange(n_to_plot)
    ax.bar(x, sorted_acc[idxes], bar_width, label='Class Accuracy')
    ax.bar(x + bar_width, sorted_precision[idxes], bar_width, label='Class Precision')
    ax.bar(x + 2*bar_width, sorted_recall[idxes], bar_width, label='Class Recall')
    ax.bar(x + 3*bar_width, sorted_f1[idxes], bar_width, label='Class F1')
    ax.set_xticks(x + 2*bar_width)
    ax.set_xticklabels(sorted_clas[idxes], rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path / "best_classes.png")
    print(f"Saved plot to {save_path / 'best_classes.png'}")


    # load confusion matrix
    with open(save_path / "confusion_matrix.json") as f:
        confusion_matrix = json.load(f)

    tot_matrix = np.sum(confusion_matrix)

    # we plot the confusion matrix
    df_cm = pd.DataFrame(confusion_matrix/tot_matrix)
    plt.figure(figsize = (20,14))
    sns.heatmap(df_cm, annot=False, cmap='Blues', fmt='g')
    # add tick names
    plt.xticks(np.arange(len(clas)), clas, rotation=90)
    plt.yticks(np.arange(len(clas)), clas, rotation=0)

    # add labels true and predicted
    plt.xlabel("True")
    plt.ylabel("Predicted")
    
    plt.savefig(save_path / "confusion_matrix.png")
    print(f"Saved plot to {save_path / 'confusion_matrix.png'}")

    print("Done with the plotting!")
    


