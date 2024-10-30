from pathlib import Path
import json
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from utils import set_torch_seed, get_mean_std_from_json
from model import get_model_from_name

def get_argparser():
    parser = argparse.ArgumentParser(description="Script to plot training results.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--save_path", type=str, default='./run/',
                        help="Path to save the run")
    parser.add_argument("--model_file_name", type=str, default='best_model.pth', help="Name of the model file")
    parser.add_argument("--model_name", type=str, default="small", choices=["small", "simple", "inception_v3", "inception_v3_ft"], help="Name of the model to use")
    parser.add_argument("--eval_data_path", type=str, default='/tmp/food-101/test', help="Path to the evaluation dataset")
    parser.add_argument("--random_seed", type=int, default=1994, help="Random seed for reproducibility")

    return parser

if __name__ == "__main__":
    args = get_argparser().parse_args()

    # we set the random seed
    set_torch_seed(args.random_seed)
    
    save_path = Path(args.save_path)

    # we load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_from_name(args.model_name)
    model.load_state_dict(torch.load(save_path / args.model_file_name, weights_only=True))
    model.to(device)
    model.eval()


    # we load the evaluation dataset
    mean, std = get_mean_std_from_json(save_path / "mean_std.json")

    if args.model_name in ["inception_v3", "inception_v3_ft"]:
        eval_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
    else:
        eval_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    eval_dataset = ImageFolder(args.eval_data_path, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)



    # evaluate the model
    # class-wise accuracy, recall, precision, f1-score
    class_correct = np.zeros(101)
    class_total = np.zeros(101)
    class_true_positive = np.zeros(101)
    class_true_negative = np.zeros(101)
    class_false_positive = np.zeros(101)
    class_false_negative = np.zeros(101)
    confusion_matrix = np.zeros((101, 101))

    with torch.no_grad():
        i = 0
        pbar = tqdm(eval_loader, desc="Evaluating the model")
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()
            # for each element of the batch
            for i, label in enumerate(labels):
                prediction = predicted[i]
                
                # we increment the confusion matrix
                confusion_matrix[label, prediction] += 1

                # we increment the total count for this label
                class_total[label] += 1

                # if the prediction is correct
                if prediction == label:
                    class_correct[label] += 1
                    class_true_positive[label] += 1
                    # we increment the true negative count for all other classes
                    class_true_negative += 1
                    class_true_negative[label] -= 1

                # if the prediction is incorrect
                else:
                    # it is a false positive for the predicted class
                    class_false_positive[prediction] += 1
                    # it is a false negative for the actual class
                    class_false_negative[label] += 1
                    # we increment the true negative count for all other classes
                    class_true_negative += 1
                    class_true_negative[label] -= 1
            
            i += 1
            pbar.update(1)
            # if i >= 2:
            #     break
        pbar.close()

    # we compute the metrics
    class_metrics = {}
    accuracy = class_correct / class_total
    recall = class_true_positive / (class_true_positive + class_false_negative + 1e-6)
    precision = class_true_positive / (class_true_positive + class_false_positive + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    class_metrics = {
        "accuracy": accuracy.tolist(),
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "f1_score": f1_score.tolist(),
        "class_names": eval_dataset.classes
    }

    # we save the metrics
    with open(save_path / "class_metrics.json", "w", encoding='utf-8') as f:
        json.dump(class_metrics, f, indent=4)
        print(f"Saved class-wise metrics to {save_path / 'class_metrics.json'}")

    # we compute the global metrics
    total_accuracy = sum(class_correct) / sum(class_total)
    mean_recall = np.mean(recall)
    mean_precision = np.mean(precision)
    mean_f1_score = np.mean(f1_score)

    global_metrics = {}
    global_metrics["total_accuracy"] = total_accuracy.item()
    global_metrics["mean_recall"] = mean_recall.item()
    global_metrics["mean_precision"] = mean_precision.item()
    global_metrics["mean_f1_score"] = mean_f1_score.item()

    # we save the global metrics
    with open(save_path / "global_metrics.json", "w", encoding='utf-8') as f:
        json.dump(global_metrics, f, indent=4)
        print(global_metrics)
        print(f"Saved global metrics to {save_path / 'global_metrics.json'}")
        
    # we save the confusion matrix
    with open(save_path / "confusion_matrix.json", "w", encoding='utf-8') as f:
        json.dump(confusion_matrix.tolist(), f, indent=4)
        print(f"Saved confusion matrix to {save_path / 'confusion_matrix.json'}")

    print("Evaluation completed!")