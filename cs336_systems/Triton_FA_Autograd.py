import torch
import triton
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import  Float
from Triton_FlashAttention import flash_attention_kernel

class FlashAttention2Func(torch.autograd.Function):
       @staticmethod
       def forward(ctx,
        Q: Float[Tensor, "... queries d"], 
        K: Float[Tensor, "... keys d"], 
        V: Float[Tensor, "... keys d"], 
        is_causal=False):
        D = Q.shape[-1]
        NUM_QUERIES = Q.shape[-2]
        NUM_KEYS = K.shape[-2]
        ROW_TILE_SIZE = max(16, NUM_QUERIES // (4 * D))
        COLUMN_TILE_SIZE = max(16, NUM_KEYS // (4 * D))
        BATCH_SIZE = Q.shape[0]
        v_stride = max(16, NUM_KEYS // (4 * D))
        q_stride = D
        q_stride_dim = 1
        k_stride = D
        k_stride_dim = 1
        v_stride = D
        v_stride_dim = 1
        o_stride = D
        o_stride_dim = 1
        l_stride = 1
        l_stride_dim = 1
     

        output = torch.empty((NUM_QUERIES, D))
        tl = torch.zeros((NUM_QUERIES, 1))

        #assert len(Q.shape) == 2, "Q must be a 2D tensor"

        flash_attention_kernel[(triton.cdiv(NUM_QUERIES,ROW_TILE_SIZE), BATCH_SIZE)](
            Q.contiguous().view(-1, D),
            K.contiguous().view(-1, D),
            V.contiguous().view(-1, D),
            output.contiguous().view(-1, D),
            tl.contiguous().view(-1, 1),
            q_stride, q_stride_dim,
            k_stride, k_stride_dim,
            v_stride, v_stride_dim,
            o_stride, o_stride_dim,
            l_stride, l_stride_dim,
            NUM_QUERIES, NUM_KEYS, D,
            ROW_TILE_SIZE, COLUMN_TILE_SIZE,
        )

        ctx.save_for_backward(Q, K, V,tl)
        return output