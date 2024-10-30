import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

from utils import set_torch_seed

IMG_SUFFIX = ".jpg"

def get_argparser():
    parser = argparse.ArgumentParser(description="Script to process dataset. Split and create folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default='/tmp/food-101',
                        help="Path to the dataset folder")
    parser.add_argument("--n_valid_per_class", type=int, default=105, help="Number of validation images per class")
    parser.add_argument("--random_seed", type=int, default=1994, help="Random seed for reproducibility")

    return parser

def split_train_valid(source_json, n_valid_per_class, np_random):
    train_json = {}
    valid_json = {}
    for key, value in source_json.items():
        np_random.shuffle(value)
        train_json[key] = value[n_valid_per_class:]
        valid_json[key] = value[:n_valid_per_class]
    return train_json, valid_json

def make_image_folder(source_folder, target_folder, class_dict):
    target_folder.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(class_dict.items(), desc=f"Creating class folders for {target_folder}")
    for class_name, class_images in class_dict.items():
        class_folder = target_folder / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        for image_name in class_images:
            source_image = (source_folder / image_name).with_suffix(IMG_SUFFIX)
            target_image = (target_folder / image_name).with_suffix(IMG_SUFFIX)

            try:
                target_image.symlink_to(source_image)
            except FileExistsError:
                print(f"File {target_image} already exists. Skipping.")
                
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    args = get_argparser().parse_args()

    data_path = Path(args.data_path)

    # set the random seed
    set_torch_seed(args.random_seed)

    np_random = np.random.RandomState(args.random_seed)

    with open(data_path / 'meta/train.json', 'r', encoding='utf-8') as f:
        train_json = json.load(f)
    with open(data_path / 'meta/test.json', 'r', encoding='utf-8') as f:
        test_json = json.load(f)

    # we split the train images into train and valid
    train_split_json, valid_split_json = split_train_valid(train_json, args.n_valid_per_class, np_random)

    # we save the split json files
    with open(data_path / 'meta/train_split.json', 'w', encoding='utf-8') as f:
        json.dump(train_split_json, f)
    with open(data_path / 'meta/valid_split.json', 'w', encoding='utf-8') as f:
        json.dump(valid_split_json, f)

    # we create the corresponding image folders
    train_folder = data_path / 'train'
    valid_folder = data_path / 'valid'
    test_folder = data_path / 'test'

    make_image_folder(data_path / 'images', train_folder, train_split_json)
    make_image_folder(data_path / 'images', valid_folder, valid_split_json)
    make_image_folder(data_path / 'images', test_folder, test_json)

    print("\n Done spliting the dataset and creating the folders. Enjoy training!")


