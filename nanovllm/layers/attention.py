import torch
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_heads == self.num_kv_heads:
            return x
        repeat = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def torch_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: int) -> torch.Tensor:
        k = self.expand_kv_heads(k)
        v = self.expand_kv_heads(v)
        scores = torch.einsum("qhd,khd->hqk", q, k).float() * self.scale
        q_pos = torch.arange(query_start, query_start + q.size(0), device=q.device)
        k_pos = torch.arange(k.size(0), device=q.device)
        causal_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        return torch.einsum("hqk,khd->qhd", probs, v)

    def gather_paged_kv(self, cache: torch.Tensor, block_table: torch.Tensor, seq_len: int) -> torch.Tensor:
        block_size = cache.size(1)
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = block_table[:num_blocks].long()
        return cache.index_select(0, block_ids).reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]

    def torch_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        outputs = []
        for i in range(context.cu_seqlens_q.numel() - 1):
            q_start = int(context.cu_seqlens_q[i].item())
            q_end = int(context.cu_seqlens_q[i + 1].item())
            k_end = int(context.cu_seqlens_k[i + 1].item())
            q_len = q_end - q_start
            k_len = k_end - int(context.cu_seqlens_k[i].item())
            if context.block_tables is None:
                k_seq = k[q_start:q_start + k_len]
                v_seq = v[q_start:q_start + k_len]
            else:
                k_seq = self.gather_paged_kv(self.k_cache, context.block_tables[i], k_len)
                v_seq = self.gather_paged_kv(self.v_cache, context.block_tables[i], k_len)
            outputs.append(self.torch_attention(q[q_start:q_end], k_seq, v_seq, k_len - q_len))
        return torch.cat(outputs, dim=0)

    def torch_decode(self, q: torch.Tensor) -> torch.Tensor:
        context = get_context()
        outputs = []
        for i in range(q.size(0)):
            seq_len = int(context.context_lens[i].item())
            k_seq = self.gather_paged_kv(self.k_cache, context.block_tables[i], seq_len)
            v_seq = self.gather_paged_kv(self.v_cache, context.block_tables[i], seq_len)
            outputs.append(self.torch_attention(q[i:i + 1], k_seq, v_seq, seq_len - 1))
        return torch.cat(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if flash_attn_varlen_func is None:
                return self.torch_prefill(q, k, v)
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if flash_attn_with_kvcache is None:
                return self.torch_decode(q)
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
