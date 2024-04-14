import torch
from flash_attn import flash_attn_with_kvcache

batch = 2
nheads = 6
seqlen_q = 3
seqlen_k = 2048
dim_qk = 256
dim_v = 256

dtype = torch.float16

q = torch.randn(batch, seqlen_q, nheads, dim_qk, device='cuda', dtype=dtype)
k, v = None, None
kcache = torch.randn(batch, seqlen_k, nheads, dim_qk, device='cuda', dtype=dtype)
vcache = torch.randn(batch, seqlen_k, nheads, dim_v, device='cuda', dtype=dtype)

cache_seqlens = torch.randint(0, seqlen_k + 1, (batch,), device='cuda', dtype=torch.int32)

out = flash_attn_with_kvcache(
    q,
    kcache,
    vcache,
    k,
    v,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=cache_seqlens,
    cache_batch_idx=None,
    block_table=None,
    causal=False,
    window_size=(-1,-1),
    rotary_interleaved=False,
    alibi_slopes=None,
    num_splits=1, # flash_fwd_kernel
)

out = flash_attn_with_kvcache(
    q,
    kcache,
    vcache,
    k,
    v,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=cache_seqlens,
    cache_batch_idx=None,
    block_table=None,
    causal=False,
    window_size=(-1,-1),
    rotary_interleaved=False,
    alibi_slopes=None,
    num_splits=0, #  flash_fwd_splitkv_kernel, flash_fwd_splitkv_combine_kernel
)
