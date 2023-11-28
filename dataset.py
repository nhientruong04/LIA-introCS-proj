from torchvision import datasets
from pathlib import Path
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# Only supports 3 datasets for manual training: CIFAR-10, CIFAR-100, MNIST

configs = {
    'cifar10': {
        'name': 'CIFAR10',
        'num_class': 10,
        'data_folder': './cifar10'
    },
    'cifar100': {
        'name': 'CIFAR100',
        'num_class': 100,
        'data_folder': './cifar100',
    }
}

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

class Dataset():
    def __init__(self, dataset_name, shuffle, split=[0.9, 0.1], seed=42):
        self.dataset_name = dataset_name.lower()
        self.dataset_config = configs[self.dataset_name]
        self.transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            normalize,
                        ])
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        
    def prepare_dataset(self):
        dataset_name = getattr(datasets, self.dataset_config['name'])
        train_dataset = dataset_name(
            root=self.dataset_config['data_folder'], train=True,
            download=True, transform=self.transform
        )

        num_train = len(train_dataset)
        valid_size = self.split[1]
        split = int(np.floor(valid_size * num_train))
        indices = list(range(num_train))

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        training_settings = {
            'train': train_sampler,
            'valid': valid_sampler,
            'num_classes': self.dataset_config['num_class']
        }

        return training_settings, train_dataset