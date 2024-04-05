import os
import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset  # Ensure Dataset is imported here
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_tensor

torch.manual_seed(1)
batch_size = 512
use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        # super().__init__(root=root, train=train, download=download, transform=transform)
class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]
        image = image.float() 

        return image, label
    
transform = A.Compose([
    A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
    A.RandomCrop(width=32, height=32),
    A.Flip(p=0.5),
    A.CoarseDropout(max_holes=1, max_height=8, max_width=8, fill_value=0, p=0.5),
    ToTensorV2(),
])
#torch.utils.data.DataLoader
train_dataset = Cifar10SearchDataset('./data', train=True, download=True, transform=transform)
test_dataset = Cifar10SearchDataset('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)