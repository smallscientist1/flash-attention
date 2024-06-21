from flash_attn import flash_attn_func, flash_attn_varlen_func
import torch
import math
import torch.nn.functional as F

from copy import deepcopy

batch_size, seq_len = 4, 2048
Kdim = 256
D = 256
'''
D must = Kdim
'''
nheads = 8
causal = False
device = 'cuda'
dtype = torch.float16
dropout_p = 0.0

q = torch.randn(batch_size,seq_len,nheads,Kdim,device=device, dtype=dtype,requires_grad=True)
k = torch.randn(batch_size,seq_len,nheads,Kdim,device=device,dtype=dtype,requires_grad=True)
v = torch.randn(batch_size,seq_len,nheads,D,device=device,dtype=dtype,requires_grad=True)
o = flash_attn_func(q,k,v,dropout_p=dropout_p,causal=causal)

grad = torch.randn(batch_size,seq_len,nheads,Kdim,device=device, dtype=dtype)
# q.retain_grad()
# k.retain_grad()
# v.retain_grad()
try:
    o.backward(grad)
except:
    print("backward fail!!!")
qgrad = deepcopy(q.grad)
kgrad = deepcopy(k.grad)
vgrad = deepcopy(v.grad)


softmax_scale = 1.0/math.sqrt(Kdim)
q_ref = q.detach().clone().requires_grad_(True)
k_ref = k.detach().clone().requires_grad_(True)
v_ref = v.detach().clone().requires_grad_(True)
q2 = q_ref.transpose(1,2)
k2 = k_ref.transpose(1,2)
v2 = v_ref.transpose(1,2)
qk = q2 @ k2.transpose(-1,-2)
s = F.softmax(qk*softmax_scale,dim=-1)
out = s @ v2
out = out.transpose(1,2)

iscorrect = torch.allclose(out,o)
print(iscorrect)
if not iscorrect:
    diffo = (o - out).abs()
    print("fwd", iscorrect, diffo.max(), diffo.min(), diffo.mean())

q_ref.grad = None
k_ref.grad = None
v_ref.grad = None
# q.retain_grad()
# k.retain_grad()
# v.retain_grad()
out.backward(grad)
iscorrect = torch.allclose(q_ref.grad,qgrad)
print(iscorrect)
if not iscorrect:
    diffo = (q_ref.grad-qgrad).abs()
    print("dq", iscorrect, diffo.max(), diffo.min(), diffo.mean())
iscorrect = torch.allclose(k_ref.grad,kgrad)
print(iscorrect)
if not iscorrect:
    diffo = (k_ref.grad-kgrad).abs()
    print("dk", iscorrect, diffo.max(), diffo.min(), diffo.mean())
print(iscorrect)
if not iscorrect:
    diffo = (v_ref.grad-vgrad).abs()
    print("dv", iscorrect, diffo.max(), diffo.min(), diffo.mean())
