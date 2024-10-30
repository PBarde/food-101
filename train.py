import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary

import argparse
from tqdm import tqdm
from pathlib import Path
import json

from model import get_model_from_name
from utils import set_torch_seed, get_mean_std_from_json

RANDOM_SEED = 1994
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True


def get_argparser():
    parser = argparse.ArgumentParser(description="Script to train on food-101 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_imgs_path", type=str, default='/tmp/food-101/train',
                        help="Path to the train images folder")
    parser.add_argument("--valid_imgs_path", type=str, default='/tmp/food-101/valid',
                        help="Path to the valid images folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the dataloader")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--random_seed", type=int, default=1994, help="Random seed for reproducibility")

    parser.add_argument("--model_name", type=str, default="simple", choices=["small", "simple", "improved", "inception_v3", "inception_v3_ft"], help="Name of the model to use")
    parser.add_argument("--save_path", type=str, default='./run/',
                        help="Path to save the run")

    return parser

def get_mean_std_from_datapath(data_path):
    # define a simple transformation that only converts the images to tensors (without normalization)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor() 
    ])

    # load the dataset 
    dataset = ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)


    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    # loop over the dataset to calculate mean and std
    pbar = tqdm(desc=f"Calculating mean and std for {data_path}", total=len(dataloader))
    for images, _ in dataloader:

        n_samples += images.size(0)
        
        mean += images.mean([0, 2, 3]) * images.size(0) 
        std += images.std([0, 2, 3]) * images.size(0)  
        pbar.update(1)
    pbar.close()

    mean /= n_samples
    std /= n_samples

    return mean, std


if __name__ == '__main__':
    args = get_argparser().parse_args()

    # set the random seed
    set_torch_seed(args.random_seed)

    # create the save path
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # we save the args to a json file
    with open(str(Path(args.save_path) / "args.json"), "w", encoding='utf-8') as f:
        json.dump(vars(args), f)
        print(f"Args saved to {Path(args.save_path) / 'args.json'}")

    save_mean_std_path = Path(args.save_path) / "mean_std.json"

    if not save_mean_std_path.exists():
        print(f"Mean and std not found at {save_mean_std_path}. Calculating...")
        # calculate mean and std of the dataset
        mean, std = get_mean_std_from_datapath(args.train_imgs_path)
        print(f"Train datasaset Mean: {mean}, Std: {std}")

        with open(str(save_mean_std_path), "w", encoding='utf-8') as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
            print(f"Mean and std saved to {save_mean_std_path}")
    
    # load mean and std from json
    mean, std = get_mean_std_from_json(save_mean_std_path)

    if args.model_name in ["inception_v3", "inception_v3_ft"]:
        # define the transformation for the train images
        train_preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])

        # define the transformation for the validation images
        valid_preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
    else:
        # define the transformation for the train images
        train_preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # define the transformation for the validation images
        valid_preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # create the datasets
    train_dataset = ImageFolder(root=args.train_imgs_path, transform=train_preprocess)
    valid_dataset = ImageFolder(root=args.valid_imgs_path, transform=valid_preprocess)

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    # define the model
    model = get_model_from_name(args.model_name).to(device)
    summary(model, (3, 512, 512), device=device.type)

    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize the best validation accuracy
    best_valid_accuracy = 0

    # start the training loop

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(args.n_epochs):
        # initialize the train and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        train_accuracy = 0.0
        valid_accuracy = 0.0

        # set the model to train mode
        model.train()
        
        # loop over the training set
        pb = tqdm(desc=f"Training epoch {epoch}", total=len(train_loader))
        i = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # clear the gradients
            optimizer.zero_grad()
            
            # forward pass
            output = model(data)
            
            # calculate the loss
            loss = criterion(output, target)
            
            # backward pass
            loss.backward()
            
            # update the weights
            optimizer.step()
            
            # update the train loss
            train_loss += loss.item() * data.size(0)

            # compute the accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_accuracy += accuracy.item() * data.size(0)

            if i % 100 == 0:
                print(f"Train Loss: {loss.item():.4f}, Train Accuracy: {accuracy.item():.4f}"),
            i += 1
            pb.update(1)

        pb.close()
        
        # set the model to evaluation mode
        model.eval()
        
        # loop over the training set
        pb = tqdm(desc=f"Eval epoch {epoch}", total=len(valid_loader))

        # turn off gradients for validation
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                
                # forward pass
                output = model(data)

                # calculate the loss
                loss = criterion(output, target)

                # update the validation loss
                valid_loss += loss.item() * data.size(0)

                # compute the accuracy
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                valid_accuracy += accuracy.item() * data.size(0)
                pb.update(1)

        pb.close()

        # calculate the average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        train_accuracy = train_accuracy / len(train_loader.dataset)
        valid_accuracy = valid_accuracy / len(valid_loader.dataset)

        # pppend the losses and accuracies

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        
        print(f"Epoch: {epoch+1}/{args.n_epochs}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            torch.save(model.state_dict(), str(Path(args.save_path) / "best_model.pth"))
            print(f"Best model saved with accuracy: {best_valid_accuracy:.4f}")

        # save the model at the end of each epoch
        torch.save(model.state_dict(), str(Path(args.save_path) / "model.pth"))
        print(f"Model saved at the end of epoch {epoch}")

        # save the losses and accuracies
        with open(str(Path(args.save_path) / "train_metrics.json"), "w", encoding='utf-8') as f:
            json.dump({"loss": train_losses, "accuracy": train_accuracies}, f)
        
        with open(str(Path(args.save_path) / "valid_metrics.json"), "w", encoding='utf-8') as f:
            json.dump({"loss": valid_losses, "accuracy": valid_accuracies}, f)

        print(f"Losses and accuracies saved at the end of epoch {epoch}")
    
    print("Training completed!")
    


    



