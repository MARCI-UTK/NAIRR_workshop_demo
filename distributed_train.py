import os 
import tqdm
import timm 
import yaml
import torch
import argparse
from torch.utils import tensorboard
from torchvision.transforms import v2 
from torch.nn.parallel import DistributedDataParallel as DDP

import chexpert_dataset

def train(local_rank: int, global_rank: int, world_size: int, 
          is_logger: bool, params: dict) -> None: 

    device = f'cuda:{local_rank}'

    model_name = params['model_name']
    log_dir = params['log_dir']
    data_root = params['data']['root']
    num_epochs = params['optimization']['num_epochs']

    if is_logger:
        writer = tensorboard.SummaryWriter(log_dir=log_dir)

    model = timm.create_model(model_name, pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 14)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    train_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.53305], std=[0.03491])
    ])

    val_transform = v2.Compose([
        v2.Resize((288, 288)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.53305], std=[0.03491])
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

    sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=2,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=64,
        num_workers=2,
        shuffle=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    for e in range(num_epochs): 
        sampler.set_epoch(e)

        model.train()
        pbar = tqdm.tqdm(train_dataloader) if is_logger else train_dataloader
        for idx, (img, lbl) in enumerate(pbar): 
            optimizer.zero_grad()

            img = img.to(device)
            lbl = lbl.to(device)

            out = model(img)

            loss = loss_fn(out, lbl)
            loss.backward()
            optimizer.step()

            if is_logger: 
                pbar.update(1)

                itr = e * len(train_dataloader) + idx
                writer.add_scalar('Loss/train', loss.item(), itr)

            return 

        model.eval()
        pbar = tqdm.tqdm(val_dataloader) if is_logger else val_dataloader
        for idx, (img, lbl) in enumerate(pbar): 
            img = img.to(device)
            lbl = lbl.to(device)

            out = model(img)

            loss = loss_fn(out, lbl)

            if is_logger: 
                pbar.update(1)

                itr = e * len(train_dataloader) + idx
                writer.add_scalar('Loss/validation', loss.item(), itr)
