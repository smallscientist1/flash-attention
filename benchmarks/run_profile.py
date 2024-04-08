from flash_attn import flash_attn_func
import torch

batch_size, seq_len = 4, 2048
Kdim = 512
D = 512
nheads = 8
causal = False
device = 'cuda'
dtype = torch.float16
dropout_p = 0.0

q = torch.randn(batch_size,seq_len,nheads,Kdim,device=device, dtype=dtype,requires_grad=True)
k = torch.randn(batch_size,seq_len,nheads,Kdim,device=device,dtype=dtype,requires_grad=True)
v = torch.randn(batch_size,seq_len,nheads,D,device=device,dtype=dtype,requires_grad=True)
o = flash_attn_func(q,k,v,dropout_p=dropout_p,causal=causal)
grad = torch.randn(batch_size,seq_len,nheads,D,device=device,dtype=dtype)
o.backward(grad)
