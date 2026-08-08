import torch
from cs336_basics.model import scaled_dot_product_attention
import time

start = time.perf_counter()
compiled_attention = torch.compile(scaled_dot_product_attention)


# num_layers for this model is 32
D = [16, 32, 64, 128]
T = [256, 1024, 4096, 8192, 16384]

# warmup
Q = torch.randn((8, 256, 16), requires_grad=True, device="cuda")
K = torch.randn((8, 256, 16), requires_grad=True, device="cuda")
V = torch.randn((8, 256, 16), requires_grad=True, device="cuda")
mask = None
for _ in range(10):
    y = compiled_attention(Q, K, V, mask)
    loss = y.sum()
    loss.backward()

curr = start

for d in D:
    for t in T:
        print(f"Running for d={d}, t={t}")
        Q = torch.randn((8, t, d), requires_grad=True, device="cuda")
        K = torch.randn((8, t, d), requires_grad=True, device="cuda")
        V = torch.randn((8, t, d), requires_grad=True, device="cuda")
        mask = None
        for _ in range(100):
            y = compiled_attention(Q, K, V, mask)
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
        print(f"ForwardTime taken: {forward_end - curr} seconds")
        print(f"Memory allocated: {torch.cuda.memory_allocated()}")
        
        for _ in range(100):
            loss = y.sum()
            loss.backward()
            torch.cuda.synchronize()
            backward_end = time.perf_counter()
        print(f"Backward Time taken: {backward_end - forward_end} seconds")
        curr = backward_end


