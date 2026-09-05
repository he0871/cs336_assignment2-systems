import triton
import triton.language as tl
from einops import rearrange, einsum
from jaxtyping import  Float, Int
from torch import Tensor

@triton.jit
def flash_attention_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
        q_stride, q_stride_dim,
        k_stride, k_stride_dim,
        v_stride, v_stride_dim,
        o_stride, o_stride_dim,
        l_stride, l_stride_dim,
        NUM_QUERIES: Int,
        NUM_KEYS: Int,
        D: Int,
        ROW_TILE_SIZE: tl.constexpr, #Br
        COLUMN_TILE_SIZE: tl.constexpr, #Bc
        BATCH_SIZE: tl.constexpr,
):
    q_tile = tl.program_id(0)
    batch = tl.program_id(1)
    q_block_ptr = tl.make_block_ptr(
        q_ptr,
        shape=(BATCH_SIZE, NUM_QUERIES, D),
        strides=(NUM_QUERIES * q_stride,q_stride, q_stride_dim), # Strides are distances between elements in the block
        offsets=(batch, q_tile * ROW_TILE_SIZE, 0), 
        block_shape=(1, ROW_TILE_SIZE, D),
        order=(2,1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        k_ptr,
        shape=(BATCH_SIZE, NUM_KEYS, D),
        strides=(NUM_KEYS * k_stride,k_stride, k_stride_dim),
        offsets=(batch,0, 0),
        block_shape=(1, COLUMN_TILE_SIZE, D),
        order=(2, 1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        v_ptr,
        shape=(BATCH_SIZE, NUM_KEYS, D),
        strides=(NUM_KEYS * v_stride,v_stride, v_stride_dim),
        offsets=(batch, 0, 0),
        block_shape=(1, COLUMN_TILE_SIZE, D),
        order=(2, 1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        o_ptr,
        shape=(BATCH_SIZE, NUM_QUERIES, D),
        strides=(NUM_QUERIES * o_stride, o_stride, o_stride_dim),
        offsets=(batch, q_tile * ROW_TILE_SIZE, 0),
        block_shape=(1, ROW_TILE_SIZE, D),
        order=(2, 1, 0),
    )   
    l_block_ptr = tl.make_block_ptr(
        l_ptr,
        shape=(BATCH_SIZE, NUM_QUERIES, 1),
        strides=(NUM_QUERIES * l_stride, l_stride, l_stride_dim),
        offsets=(batch, q_tile * ROW_TILE_SIZE, 0),
        block_shape=(1, ROW_TILE_SIZE, 1),
        order=(2, 1, 0),
    )
    q_block = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m_j = tl.full((ROW_TILE_SIZE, 1), float("-inf"))
    prev_l = tl.zeros((ROW_TILE_SIZE, 1))
    prev_o = tl.zeros((ROW_TILE_SIZE, D))
    o_i = tl.zeros((ROW_TILE_SIZE, D))
    l_j = tl.zeros((ROW_TILE_SIZE, 1))
    o_j = tl.zeros((ROW_TILE_SIZE, D))
    for i in range(tl.cdiv(NUM_KEYS, COLUMN_TILE_SIZE)):
        
        k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k_transposed = tl.transpose(k_block, (1, 0))
        s_j = tl.matmul(q_block, k_transposed)
        s_j = s_j / (D ** 0.5)
        prev_m = m_j
        m_j = tl.max(prev_m, tl.max(s_j, dim=1, keepdim=True).values)

        p_j = tl.exp(s_j - m_j)

        adjust = tl.exp(prev_m - m_j)

        l_j = adjust * prev_l + tl.sum(p_j, dim=1, keepdim=True)

        prev_l = l_j

        o_j = adjust * prev_o + p_j @ v_block
        prev_o = o_j

        k_block_ptr = k_block_ptr.advance((0,COLUMN_TILE_SIZE, 0))
        v_block_ptr = v_block_ptr.advance((0,COLUMN_TILE_SIZE, 0))

    o_i = o_j / l_j
    l_i = tl.log(l_j) + m_j
    tl.store(o_block_ptr, o_i, boundary_check=(0, ))
    tl.store(l_block_ptr, l_i, boundary_check=(0, ))

 