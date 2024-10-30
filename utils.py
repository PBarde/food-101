import json
import torch

def set_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_mean_std_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Loaded mean and std from {json_path}: {data}")
    return torch.tensor(data["mean"]), torch.tensor(data["std"])