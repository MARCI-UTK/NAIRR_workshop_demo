import os 
import tqdm
import timm 
import yaml
import torch
import argparse
from torchvision.transforms import v2 
from torch.utils.tensorboard import SummaryWriter 
from torch.nn.parallel import DistributedDataParallel as DDP

import chexpert_dataset

"""
Training run for a single GPU process 
"""
def train(local_rank: int, global_rank: int, world_size: int, 
          is_logger: bool, params: dict) -> None: 
    
    """
    Model training routine.

    Parameters
    ----------
    local_rank : int
        GPU ID on the local node
    global_rank : int
        GPU ID in the overall training process 
    world_size: int 
        Total number of GPUs being used by the training process 
    is_logger : bool
        When True, enables Tensorboard logs and progress bar.
    params : dict
        Configuration of training session.

    Returns
    -------
    None
        No return value.
    """

    # Specify GPU 
    device = f'cuda:{local_rank}'

    # Load parameters specified in config file 
    model_name = params['model_name']
    log_dir = params['log_dir']
    data_root = params['data']['root']
    num_workers = params['data']['num_workers']
    batch_size = int(params['data']['batch_size'])
    num_epochs = int(params['optimization']['num_epochs'])
    learning_rate = float(params['optimization']['learning_rate'])

    # Setup Tensorboard logging 
    if is_logger:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    """
    Initialize and setup model 
    """
    # Load from timm
    model = timm.create_model(model_name, pretrained=False)

    # Change the first conv. layer to 1 channel for greyscale input 
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Change the final FC layer to output 14 classes to fit CheXpert 
    model.fc = torch.nn.Linear(model.fc.in_features, 14)

    # Put model on GPU and wrap in DDP for distributed training
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    """
    Build dataset
    Setup the following for train and val. splits:
        - Augmentations 
        - Dataset
        - DataLoader 
    """
    train_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5029], std=[0.289])
    ])

    val_transform = v2.Compose([
        v2.Resize((288, 288)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5029], std=[0.289])
    ])

    train_dataset = chexpert_dataset.CheXpertDataset(
        root=data_root,
        split='train',
        transform=train_transform
    )

    val_dataset = chexpert_dataset.CheXpertDataset(
        root=data_root,
        split='valid',
        transform=val_transform
    )

    # This is required for distributed training
    # It replaces 'shuffle' in the DataLoader 
    sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )  

    # Initialize optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Initialize loss function  
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Start training loop 
    for e in range(num_epochs): 
        
        # This is for the sampler used by the DataLoader to randomize the 
        # order of the data in every epoch 
        sampler.set_epoch(e)

        # Training epoch 
        model.train()
        pbar = tqdm.tqdm(train_dataloader) if is_logger else train_dataloader
        for idx, (img, lbl) in enumerate(pbar): 

            # Clear optimizer for current step 
            optimizer.zero_grad()

            # Move data to GPU 
            img = img.to(device)
            lbl = lbl.to(device)

            # Forward pass 
            out = model(img)
            
            # Backprop
            loss = loss_fn(out, lbl)
            loss.backward()
            optimizer.step()

            # Update progress bar and add loss to Tensorboard 
            if is_logger: 
                pbar.set_postfix(loss=loss.item())

                itr = e * len(train_dataloader) + idx
                writer.add_scalar('Loss/train', loss.item(), itr)

        # Validation epoch
        # Same as training epoch just no backprop
        model.eval()
        pbar = tqdm.tqdm(val_dataloader) if is_logger else val_dataloader

        with torch.no_grad(): 

            for idx, (img, lbl) in enumerate(pbar): 
                img = img.to(device)
                lbl = lbl.to(device)

                out = model(img)

                loss = loss_fn(out, lbl)

                if is_logger: 
                    pbar.set_postfix(loss=loss.item())

                    itr = e * len(val_dataloader) + idx
                    writer.add_scalar('Loss/validation', loss.item(), itr)
