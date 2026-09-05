So the kernel shape is:

Grid: one program per Q tile (and usually another grid axis for batch/head).
Inside the kernel: a for over K/V tiles, same as your inner PyTorch loop.
That is FlashAttention-2’s usual Triton layout: Q stays on-chip, K and V stream through.

tl.loads are the “bring this tile on-chip from HBM”


So Strides is D, and  tile sizes is Br or Bc which is depends on how I chunk the matrix


For tile q_tile, you want to start at query row q_tile * Br, feature 0:

offsets = (q_tile * Br, 0)
Triton then computes q_tile * Br * D for you, because q_stride = D



Q is contiguous (B, N, D) = (4, 128, 64).

Br = 16 → cdiv(N, Br) = 8 Q-tiles per sequence
Grid: 8 by 4 → 32 programs
program_id(0) = Q-tile t in {0,…,7}
program_id(1) = batch b in {0,…,3}
Each program owns one 16 × 64 Q slab from one sequence. s_j stays (16, 16) (if Bc=16); batch is which program you are.