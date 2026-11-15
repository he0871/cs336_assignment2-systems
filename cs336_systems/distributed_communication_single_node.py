import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

WARMUP_ITERATIONS = 5

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, tensor_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (tensor_size,), device=f"cuda")
    for _ in range(WARMUP_ITERATIONS):
        dist.all_reduce(data)

    torch.cuda.synchronize()
    start = time.perf_counter()
    dist.all_reduce(data)
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = end - start
    print(f"Rank {rank}, tensor size {tensor_size}, world size {world_size}, time taken: {elapsed} seconds")
    dist.destroy_process_group()

if __name__ == "__main__":
    times = []
    for world_size in [2, 4, 6]:
        for tensor_size in [1000000, 10000000, 100000000]:
            mp.spawn(distributed_demo, args=(world_size, tensor_size), nprocs=world_size, join=True)
