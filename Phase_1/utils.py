from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import math
import os
from torch import nn, Tensor
from collections import OrderedDict
from typing import Tuple
from PIL import Image
from torchvision import transforms, models
from typing import Optional
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt

def save_checkpoint(state: OrderedDict, filename: str = 'checkpoint.pth', cpu: bool = False) -> None:
    if cpu:
        new_state = OrderedDict()
        for k in state.keys():
            newk = k.replace('module.', '')  # remove module. if model was trained using DataParallel
            new_state[newk] = state[k].cpu()
        state = new_state
    torch.save(state, filename)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class NormalizedModel(nn.Module):
    """
    Wrapper for a model to account for the mean and std of a dataset.
    mean and std do not require grad as they should not be learned, but determined beforehand.
    mean and std should be broadcastable (see pytorch doc on broadcasting) with the data.
    Args:
        model (nn.Module): model to use to predict
        mean (torch.Tensor): sequence of means for each channel
        std (torch.Tensor): sequence of standard deviations for each channel
    """

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(NormalizedModel, self).__init__()

        self.model = model
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model(normalized_input)


def requires_grad_(model:nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()

def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]

class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std
    
def normalize_model(model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)



def proj_lp(v, radius, p):
    
    # Project on the lp ball centered at 0 and of radius xi
    # Supports only p = 2 and p = Inf for now

    if p == 2.0:
        v_norm = torch.norm(v.flatten(1))
        print(f"v_norm is : {v_norm}")
        v = v * torch.min(torch.tensor(1.), radius/v_norm) if v_norm > 0 else v
    elif p == float('inf') or p == 'Inf':
        v = torch.sign(v) * torch.min(torch.abs(v), radius)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    
    return v

def calculate_epsilon(X, epsilon_factor=0.05):
    # Calculate the mean L2-norm of the input X
    X_norms = torch.norm(X.reshape(X.shape[0], -1), p=2, dim=1)
    X_mean_norm = torch.mean(X_norms)

    # Calculate epsilon
    epsilon = epsilon_factor * X_mean_norm.item()
    print(f"epsilon is : {epsilon}")
    
    return epsilon

def renorm(adv_images, images, max_norm, p=2):
    adv = torch.renorm(adv_images - images, p=2, dim=0, maxnorm=max_norm) + images
    
    return adv

def imagenet_model_factory(name: str, pretrained: bool = True, state_dict_path: Optional[str] = None, **kwargs):
    if 'inception' in name:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = [transforms.Resize(299, interpolation=Image.LANCZOS)]
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = [transforms.Resize(256, interpolation=Image.LANCZOS), transforms.CenterCrop(224)]

    model = models.__dict__[name](pretrained=pretrained, **kwargs)

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
            state_dict = {k[len('module.model.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    normalized_model = normalize_model(model=model, mean=mean, std=std)
    normalized_model.eval()
    requires_grad_(normalized_model, False)

    return normalized_model, transform

class ForwardCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_samples_called = 0

    def __call__(self, module, input) -> None:
        self.num_samples_called += len(input[0])


class BackwardCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_samples_called = 0

    def __call__(self, module, grad_input, grad_output) -> None:
        self.num_samples_called += len(grad_output[0])
        

def load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor

def load_mnist(
    n_examples: Optional[int] = None,
    data_dir: str = 'data/MNIST',
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.MNIST(root='data/MNIST',
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True)
    return load_dataset(dataset, n_examples)

def plot_perturbation(perturbation):
    # Convert tensor to numpy array and transpose to (32, 32, 3)
    perturbation_np = perturbation.squeeze().numpy().transpose(1, 2, 0)
    perturbation_np = perturbation_np * 1000
    # Scale values to range [0, 1]
    min_val = np.min(perturbation_np)
    max_val = np.max(perturbation_np)
    perturbation_np = (perturbation_np - min_val) / (max_val - min_val)

    # Plot the image
    plt.imshow(perturbation_np)
    plt.axis('off')
    plt.show()
    
def check_device_type(obj):
    if torch.is_tensor(obj):
        print("Tensor")
    elif torch.cuda.is_available() and isinstance(obj, torch.nn.Module):
        print("CUDA Module")
    elif isinstance(obj, torch.nn.Module):
        print("CPU Module")
    else:
        print("Unknown device type")

