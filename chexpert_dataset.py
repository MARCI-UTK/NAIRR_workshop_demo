import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2 

class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            root: str,
            split: str, 
            transform: (v2.Compose | None), 
            uncertainty: str = 'zeros'
        ) -> None:

        self.root = root

        csv_path = os.path.join(root, f'{split}.csv')
        self.df = pd.read_csv(csv_path)

        self.transform = transform
        self.uncertainty = uncertainty

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: 
        sample = self.df.iloc[idx]

        sample_path = str(sample['Path'])
        sample_path = sample_path.replace('CheXpert-v1.0/', '')
        sample_path = os.path.join(self.root, sample_path)
    
        image = Image.open(sample_path).convert('L')

        if self.transform:
            image = self.transform(image)

        labels = sample[5:].values.astype(np.float32)

        # Handle NaN and uncertainty (-1.0 labels)
        if self.uncertainty == 'zeros':
            labels = np.nan_to_num(labels, nan=0.0)
            labels = np.where(labels == -1.0, 0.0, labels)

        elif self.uncertainty == 'ones':
            labels = np.nan_to_num(labels, nan=0.0)
            labels = np.where(labels == -1.0, 1.0, labels)

        elif self.uncertainty == 'ignore':
            labels = np.nan_to_num(labels, nan=0.0)

        labels = torch.tensor(labels)

        return image, labels