import os
import yaml
import torch
import argparse

import distributed_train 

def setup_ddp() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    return local_rank

def cleanup_ddp() -> None:
    torch.distributed.destroy_process_group()

    return 

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True)
    args = parser.parse_args()

    with open(args.fname, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    local_rank = setup_ddp()
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    is_logger = (global_rank == 0)
    if is_logger:
        print(f"Training with {world_size} processes")

    try:
        train.train(local_rank, global_rank, world_size, is_logger, params)
    finally:
        cleanup_ddp()

    return 

if __name__ == "__main__":
    main()