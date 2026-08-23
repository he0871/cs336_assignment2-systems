import torch
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Bool, Float, Int

class FlashAttention2Func(torch.autograd.Function):
       
    @staticmethod
    def forward(ctx,
        Q: Float[Tensor, "... queries d"], 
        K: Float[Tensor, "... keys d"], 
        V: Float[Tensor, "... keys d"], 
        is_causal=False):

        d = Q.shape[-1]
        seq_len = Q.shape[-2]
        num_keys = K.shape[-2] 
        batch_size = Q.shape[0]
        br = max(16, seq_len // (4 * d))
        bc = 16
        q_chunk_pt = 0
        print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
        Q_blocks = rearrange(Q, "... (n_q Br) d ->   n_q ... Br d", Br=br)
        K_blocks = rearrange(K, "... (n_k Bc) d ->   n_k ... Bc d", Bc=bc)
        V_blocks = rearrange(V, "... (n_v Bc) d ->   n_v ... Bc d", Bc=bc)
        print(f"br: {br}, bc: {bc}")
        print(f"Q_blocks: {Q_blocks.shape}, K_blocks: {K_blocks.shape}, V_blocks: {V_blocks.shape}")


        O_blocks = []
        L_blocks = []

        # m, l, o = None, None, None
        
        l_j = torch.zeros((br, 1))
        o_j = torch.zeros((br, d))
        m_j = torch.full((br, 1), float("-inf"))
        
        prev_o = torch.zeros((br, d))
        for q_i in Q_blocks:
            #o_i = torch.zeros((br, d)) # [..., br, d]

            m_j = torch.full((br, 1), float("-inf"))
            prev_l = torch.zeros((br, 1))

            for i in range(len(K_blocks)):
                k_j = K_blocks[i] # [... bc, d]
                v_j = V_blocks[i] # [... bc, d]
                K_transposed = rearrange(k_j, " ... Bc d_k -> ... d_k Bc", Bc=bc)
                # print(f"K_transposed: {K_transposed.shape}, q_i: {q_i.shape}")
                s_j = q_i @ K_transposed # [..., br, bc]
                s_j = s_j / (d ** 0.5)

                prev_m = m_j
                m_j = torch.max(prev_m, s_j.max(dim=2, keepdim=True).values)
                
                #print(f"prev_m: {prev_m.shape}, m_j: {m_j.shape}, s_j: {s_j.shape}")
                p_j = torch.exp(s_j - m_j) # [..., br, bc]
                adjust = torch.exp(prev_m - m_j)  # [..., br, 1]
                #print(f"adjust: {adjust.shape}, prev_m: {prev_m.shape}, m_j: {m_j.shape}")
                #adjust = rearrange(adjust, "... br 1 -> ... br", br=br)
                #print(f"adjust: {adjust.shape}, prev_l: {prev_l.shape}, p_j: {p_j.shape}")
                l_j = adjust * prev_l + torch.sum(p_j, dim=2, keepdim=True) # [..., br, 1]
                #print(f"prev_l: {prev_l.shape}, l_j: {l_j.shape}")
                
                prev_l = l_j # [..., br, 1]
                #print(f"adjust: {adjust.shape}, prev_o: {prev_o.shape}, p_j: {p_j.shape}, v_j: {v_j.shape}")
                o_j = adjust * prev_o  + p_j @ v_j # [..., br, d]
                prev_o = o_j


            o_i = o_j / l_j # [..., br, d]
            O_blocks.append(o_i)
      
            l_tc = torch.log(l_j)
            l_i = m_j + l_tc
            print(f"l_i: {l_i.shape}")
            L_blocks.append(l_i)
            #print(f"O_blocks length: {len(O_blocks)}")
        ob_tensor = torch.cat(O_blocks, dim=1)
        print(f"final ob_tensor shape: {ob_tensor.shape}")
        ll_tensor = torch.cat(L_blocks, dim=1)
        ll_tensor = rearrange(ll_tensor, "... seq_len 1 -> ... seq_len", seq_len=seq_len)
        print(f"final l shape: {ll_tensor.shape}")
        ctx.save_for_backward(Q, K, V, ob_tensor, ll_tensor)
        return ob_tensor


                


                


                




