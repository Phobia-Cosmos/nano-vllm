import torch
from torch import nn


class Sampler(nn.Module):
    # TODO:这个是什么意思？为什么要compile以及这个Sample的作用是什么？logits是什么？这个函数的作用是什么？计算公式是什么？
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
