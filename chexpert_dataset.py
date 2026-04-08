import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import v2 

"""
Implementation of a Dataset class to load the CheXpert dataset
"""
class CheXpertDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            root: str,
            split: str, 
            transform: (v2.Compose | None), 
            uncertainty: str = 'zeros'
        ) -> None:

        """
        Initialize a CheXpert dataset object. 

        Parameters
        ----------
        root : str
            The root directory of the CheXpert dataset
        split : str
            Whether this object is loading the training or validation split of CheXpert 
            Accepts 'train' or 'valid
        transform: v2.Compose or None 
            The set of augmentations applied to each sample 
        uncertainty : str
            Specifies how to handle the 'uncertainty' labels in CheXpert 

        Returns
        -------
        None
            No return value.
        """

        self.root = root

        # Loads the CSV file that contains image paths along 
        # with their corresponding annotation
        csv_path = os.path.join(root, f'{split}.csv')
        self.df = pd.read_csv(csv_path)

        # Specify augmentations 
        self.transform = transform

        # Specify how to handle 'uncertain' annotations 
        # Default is to make these values 0.0 which is 'not present in scan'
        self.uncertainty = uncertainty

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: 
        sample = self.df.iloc[idx]

        # Load image path from CSV row and load image using PIL 
        sample_path = str(sample['Path'])
        sample_path = sample_path.replace('CheXpert-v1.0/', '')
        sample_path = os.path.join(self.root, sample_path)
    
        image = Image.open(sample_path).convert('L')

        # Augment image, if applicable
        if self.transform:
            image = self.transform(image)

        # Extract labels from CSV row 
        labels = sample[5:].values.astype(np.float32)

        # Handle NaN values and uncertainty
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