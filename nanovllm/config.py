import os
import json
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    # TODO:一个batch处理的tokens？和max_model_len区别是什么？
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    # TODO:目标GPU利用率是吗？
    gpu_memory_utilization: float = 0.9
    # TODO:可以随意设置的吗？
    tensor_parallel_size: int = 1
    # TODO:这个是什么意思？
    enforce_eager: bool = False
    # TODO：这个是什么？AutoConfig又是什么？
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        config_path = os.path.join(self.model, "config.json")
        with open(config_path, encoding="utf-8") as f:
            raw_config = json.load(f)
        if raw_config.get("model_type") == "minigpt":
            dtype_name = raw_config.get("torch_dtype", "float32")
            raw_config["dtype"] = getattr(torch, dtype_name)
            raw_config.setdefault("num_key_value_heads", raw_config["num_attention_heads"])
            raw_config.setdefault("head_dim", raw_config["hidden_size"] // raw_config["num_attention_heads"])
            raw_config.setdefault("bias", True)
            raw_config.setdefault("dropout", 0.0)
            self.hf_config = SimpleNamespace(**raw_config)
            self.enforce_eager = True
            self.tensor_parallel_size = 1
        else:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
