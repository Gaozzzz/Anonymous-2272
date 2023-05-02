import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(60)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # attention_weights = self.norm(attention_scores)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attended_values)


class Memory_Unit_transformer(nn.Module):
    def __init__(self, nums, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.multihead_attention = MultiHeadAttention(dim, num_heads)
        self.reset_parameters()
        self.norm = nn.LayerNorm(dim)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)
        if self.memory_block is not None:
            self.memory_block.data.uniform_(-stdv, stdv)

    def forward(self, data):  # data size---> B,T,D
        expanded_memory_block = self.memory_block.unsqueeze(0).expand(data.size(0), -1, -1)
        attended_memory = self.multihead_attention(data, expanded_memory_block, expanded_memory_block)

        attention_scores = torch.einsum('btd,kd->btk', data, attended_memory.mean(dim=0)) / (self.dim ** 0.5)

        augment = torch.einsum('btk,btd->btd', attention_scores, attended_memory)
        augment = self.norm(augment)

        return augment


if __name__ == "__main__":
    torch.cuda.set_device(1)  # set your gpu device
    # mu = Memory_Unit(60,512).cuda()
    data = torch.randn((64, 200, 1024)).cuda()
    mu_t = Memory_Unit_transformer(60, 1024).cuda()
    mu_t(data)
    print(111)
