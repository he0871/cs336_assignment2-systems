import torch
import triton
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import  Float
from cs336_systems.Triton_FlashAttention import flash_attention_kernel, flash_attention_backward_kernel

class TritonFlashAttention2Func(torch.autograd.Function):
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
     

        output = torch.empty((BATCH_SIZE, NUM_QUERIES, D), device="cuda")
        tl = torch.zeros((BATCH_SIZE, NUM_QUERIES, 1), device="cuda")

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
            BATCH_SIZE,
            is_causal,
        )
        tl = tl.squeeze(-1)
        ctx.save_for_backward(Q, K, V,tl,output)
        return output

    @staticmethod
    def backward(ctx,
        grad_output: Float[Tensor, "... queries d"],
        is_causal=False):

        Q = ctx.saved_tensors[0]
        K = ctx.saved_tensors[1]
        V = ctx.saved_tensors[2]
        tl = ctx.saved_tensors[3]
        output = ctx.saved_tensors[4]
        
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
        grad_output_stride = D
        grad_output_stride_dim = 1
        grad_q_stride = D
        grad_q_stride_dim = 1
        grad_k_stride = D
        grad_k_stride_dim = 1
        grad_v_stride = D
        grad_v_stride_dim = 1
    

        grad_q = torch.zeros((BATCH_SIZE, NUM_QUERIES, D), device="cuda")
        grad_k = torch.zeros((BATCH_SIZE, NUM_KEYS, D), device="cuda")
        grad_v = torch.zeros((BATCH_SIZE, NUM_KEYS, D), device="cuda")

        flash_attention_backward_kernel[(triton.cdiv(NUM_QUERIES,ROW_TILE_SIZE), BATCH_SIZE)](
            Q.contiguous().view(-1, D),
            K.contiguous().view(-1, D),
            V.contiguous().view(-1, D),
            output.contiguous().view(-1, D),
            tl.contiguous().view(-1, 1),
            grad_output.contiguous().view(-1, D),
            grad_q.contiguous().view(-1, D),
            grad_k.contiguous().view(-1, D),
            grad_v.contiguous().view(-1, D),
            q_stride, q_stride_dim,
            k_stride, k_stride_dim,
            v_stride, v_stride_dim,
            l_stride, l_stride_dim,
            o_stride, o_stride_dim,
            grad_output_stride, grad_output_stride_dim,
            grad_q_stride, grad_q_stride_dim,
            grad_k_stride, grad_k_stride_dim,
            grad_v_stride, grad_v_stride_dim,
            NUM_QUERIES, NUM_KEYS, D,
            ROW_TILE_SIZE, COLUMN_TILE_SIZE,
            BATCH_SIZE,
            is_causal,
        )
        
        return grad_q, grad_k, grad_v, None