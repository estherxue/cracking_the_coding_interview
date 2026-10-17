import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.scale = self.head_dim ** -0.5
        self.theta = theta
        rope_theta = torch.outer(torch.arange(max_seq_len).float(),
            self.theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("cos", torch.cos(rope_theta))
        self.register_buffer("sin", torch.sin(rope_theta))

    def forward(self, x, attention_mask=None, padding_mask=None):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        n_seq = q.shape[1]
        assert n_seq <= self.cos.shape[0]
        q = q.view(q.shape[0], n_seq, self.num_heads, -1)
        k = k.view(k.shape[0], n_seq, self.num_heads, -1)
        v = v.view(v.shape[0], n_seq, self.num_heads, -1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k1, k2 = k[..., 0::2], k[..., 1::2]
        q1, q2 = q[..., 0::2], q[..., 1::2]
        cos = self.cos[:n_seq]
        sin = self.sin[:n_seq]
        k = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], -1).flatten(-2)
        q = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], -1).flatten(-2)
        q_T_k = torch.einsum('b h n d, b h m d -> b h n m', q, k)
        attn_weights = q_T_k * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[None, None, :, :]
        if padding_mask is not None:
            attn_weights = attn_weights + padding_mask[:, None, None, :]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('b h n m, b h m d -> b h n d', attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(attn_output.shape[0], -1, self.d_model)
        attn_output = self.W_o(attn_output)
        return attn_output