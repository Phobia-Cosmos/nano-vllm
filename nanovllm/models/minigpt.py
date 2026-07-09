import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer


class MiniGPTCharTokenizer:
    def __init__(self, stoi: dict[str, int], itos: list[str], unk_token: str = "<unk>"):
        self.stoi = stoi
        self.itos = itos
        self.unk_token = unk_token
        self.unk_token_id = self.stoi[self.unk_token]
        self.eos_token_id = -1

    @classmethod
    def from_pretrained(cls, path: str):
        tokenizer_path = Path(path) / "tokenizer.json"
        payload = json.loads(tokenizer_path.read_text())
        if "stoi" not in payload:
            return MiniGPTHFTokenizer(tokenizer_path)
        return cls(
            stoi={str(k): int(v) for k, v in payload["stoi"].items()},
            itos=[str(x) for x in payload["itos"]],
            unk_token=str(payload.get("unk_token", "<unk>")),
        )

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_token_id) for ch in text]

    def decode(self, ids: list[int], **_: object) -> str:
        pieces = []
        for idx in ids:
            token = self.itos[int(idx)]
            pieces.append("?" if token == self.unk_token else token)
        return "".join(pieces)


class MiniGPTHFTokenizer:
    def __init__(self, tokenizer_path: str | Path):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        eos_token_id = self.tokenizer.token_to_id("<eos>")
        self.eos_token_id = -1 if eos_token_id is None else eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False, **_: object) -> str:
        return self.tokenizer.decode([int(idx) for idx in ids], skip_special_tokens=skip_special_tokens)


class MiniGPTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size, bias=config.bias)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.max_position_embeddings, config.max_position_embeddings), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MiniGPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MiniGPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.attn = MiniGPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.mlp = MiniGPTMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPTForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([MiniGPTBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, bias=config.bias)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.token_embedding.weight

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"sequence length {seq_len} exceeds max_position_embeddings")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return self.forward_hidden(input_ids).reshape(-1, self.config.hidden_size)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.inference_mode()
    def logits_for_sequences(self, token_ids: list[list[int]]) -> torch.Tensor:
        device = self.lm_head.weight.device
        last_logits = []
        max_len = self.config.max_position_embeddings
        for ids in token_ids:
            window = ids[-max_len:]
            input_ids = torch.tensor([window], dtype=torch.long, device=device)
            logits = self.compute_logits(self.forward_hidden(input_ids))
            last_logits.append(logits[0, -1])
        return torch.stack(last_logits, dim=0)
