import torch


# class RotaryPositionalEmbedding(torch.nn.Module):
#     def __init__(self, dim, base=10000, precision=torch.half):
#         """Rotary positional embedding
#         Reference : https://blog.eleuther.ai/rotary-embeddings/
#         Paper: https://arxiv.org/pdf/2104.09864.pdf
#         Args:
#             dim: Dimension of embedding
#             base: Base value for exponential
#             precision: precision to use for numerical values
#         """
#         super().__init__()
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer("inv_freq", inv_freq)
#         self.seq_len_cached = 0
#         self.cos_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
#         self.sin_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
#         self.precision = precision

#     def forward(self, x, seq_len: int = 0):
#         """
#         Args:
#             x: Input x with T X B X C
#             seq_len: Sequence length of input x
#         Returns:
#             cos_cached: [seq_len, 1, 1, dim]
#             sin_cached: [seq_len, 1, 1, dim]
#         """
#         if seq_len > self.seq_len_cached:
#             self.seq_len_cached = seq_len
#             # 生成位置索引 [0, 1, ..., seq_len-1]
#             t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
#             # 计算位置编码的频率
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#              # 复制一份得到完整维度
#             emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
#             # 转换为所需形状并缓存
#             self.cos_cached = emb.cos().view(emb.size(0), 1, 1, emb.size(1))
#             self.sin_cached = emb.sin().view(emb.size(0), 1, 1, emb.size(1))
            
            

#         return self.cos_cached, self.sin_cached

# # rotary pos emb helpers:
# def rotate_half(x):
#     x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
#     return torch.cat(
#         (-x2, x1), dim=x1.ndim - 1
#     )  # dim=-1 triggers a bug in earlier torch versions


# def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
#     # 这里的切片操作导致了空张量
#     cos, sin = (
#         cos[offset : q.shape[0] + offset, ...],
#         sin[offset : q.shape[0] + offset, ...],
#     )
#     # print(offset)
#     return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# class RotaryPositionalEmbedding(torch.nn.Module):
#     def __init__(self, dim, base=10000, precision=torch.half):
#         """Rotary positional embedding
#         Reference : https://blog.eleuther.ai/rotary-embeddings/
#         Paper: https://arxiv.org/pdf/2104.09864.pdf
#         Args:
#             dim: Dimension of embedding
#             base: Base value for exponential
#             precision: precision to use for numerical values
#         """
#         super().__init__()
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim).float() / dim))
#         self.register_buffer("inv_freq", inv_freq)
#         self.seq_len_cached = 0
#         self.cos_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
#         self.sin_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
#         self.precision = precision

#     def forward(self, x, seq_len: int = 0):
#         """
#         Args:
#             x: Input x with T X B X C
#             seq_len: Sequence length of input x
#         """
#         if seq_len > self.seq_len_cached:
#             self.seq_len_cached = seq_len
#             t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#             emb = freqs.to(x.device)
#             self.cos_cached = emb.cos().view(emb.size(0), 1, 1, emb.size(1))
#             self.sin_cached = emb.sin().view(emb.size(0), 1, 1, emb.size(1))
#         return self.cos_cached, self.sin_cached

# # rotary pos emb helpers:
# def roll_left(x):
#     # 将x第一个元素移动到最后一个位置
#     x = torch.roll(x,shifts=-1, dims=-1) 
#     return x
# def roll_right(x):
#     # 将x最后一个元素移动到第一个位置
#     x = torch.roll(x,shifts=1, dims=-1) 
#     return x
# def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
#     cos, sin = (
#         cos[offset : q.shape[0] + offset, ...],
#         sin[offset : q.shape[0] + offset, ...],
#     )
#     return (q * cos) - (roll_left(q) * sin)+(roll_right(q) * roll_right(sin)), (k * cos)- (roll_left(k) * sin) + (roll_right(k) * roll_right(sin))

# class RotaryPositionalEmbedding(torch.nn.Module):
#     def __init__(self, dim, base=10000, precision=torch.half):
#         """Rotary positional embedding
#         Reference : https://blog.eleuther.ai/rotary-embeddings/
#         Paper: https://arxiv.org/pdf/2104.09864.pdf
#         Args:
#             dim: Dimension of embedding
#             base: Base value for exponential
#             precision: precision to use for numerical values
#         """
#         super().__init__()
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim).float() / dim))
#         self.register_buffer("inv_freq", inv_freq)
#         self.seq_len_cached = 0
#         self.cos_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
#         self.sin_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
#         self.precision = precision

#     def forward(self, x, seq_len: int = 0):
#         """
#         Args:
#             x: Input x with T X B X C
#             seq_len: Sequence length of input x
#         """
#         if seq_len > self.seq_len_cached:
#             self.seq_len_cached = seq_len
#             t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#             emb = freqs.to(x.device)
#             self.cos_cached = emb.cos().view(emb.size(0), 1, 1, emb.size(1))
#             self.sin_cached = emb.sin().view(emb.size(0), 1, 1, emb.size(1))
#         return self.cos_cached, self.sin_cached

# # rotary pos emb helpers:
# def roll_left(x):
#     # 将x第一个元素移动到最后一个位置
#     x = torch.roll(x,shifts=-1, dims=-1) 
#     return x
# def roll_right(x):
#     # 将x最后一个元素移动到第一个位置
#     x = torch.roll(x,shifts=1, dims=-1) 
#     # 令 x 的第一个元素为零
#     x = torch.cat((torch.zeros_like(x[..., :1]), x[..., 1:]), dim=-1)
#     return x
# def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
#     cos, sin = (
#         cos[offset : q.shape[0] + offset, ...],
#         sin[offset : q.shape[0] + offset, ...],
#     )
#     #令sin的最后一维最后一个数字为零
#     sin0 = torch.cat((sin[..., :-1], torch.zeros_like(sin[..., -1:])), dim=-1)
#     return (q * cos) - (roll_left(q) * sin0)+(roll_right(q) * roll_right(sin)), (k * cos)- (roll_left(k) * sin0) + (roll_right(k) * roll_right(sin))

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        """Rotary positional embedding
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
            precision: precision to use for numerical values
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.sin_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.precision = precision

    def forward(self, x, seq_len: int = 0):
        """
        Args:
            x: Input x with T X B X C
            seq_len: Sequence length of input x
        """
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = freqs.to(x.device)
            self.cos_cached = emb.cos().view(emb.size(0), 1, 1, emb.size(1))
            self.sin_cached = emb.sin().view(emb.size(0), 1, 1, emb.size(1))
        return self.cos_cached, self.sin_cached

# rotary pos emb helpers:
def roll_left(x):
    # 将x第一个元素移动到最后一个位置
    x = torch.roll(x,shifts=-1, dims=-1) 
    return x
def roll_right(x):
    # 将x最后一个元素移动到第一个位置
    x = torch.roll(x,shifts=1, dims=-1) 
    # 令 x 的第一个元素为零
    x = torch.cat((torch.zeros_like(x[..., :1]), x[..., 1:]), dim=-1)
    return x
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    #令sin的最后一维最后一个数字为零
    sin0 = torch.cat((sin[..., :-1], torch.zeros_like(sin[..., -1:])), dim=-1)
    cos = roll_right(cos)
    cos = torch.cat((cos[..., 1:2], cos[...,1:]), dim=-1)
    return (q * cos) - (roll_left(q) * sin0)+(roll_right(q) * roll_right(sin)), (k * cos)- (roll_left(k) * sin0) + (roll_right(k) * roll_right(sin))

