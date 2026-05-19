"""
1. 位置编码：none / simple absolute / learned / sinusoidal
2. Q,K,V 必要性：separate / share_qk / share_kv / single_shared
3. ResNet/残差结构：standard / no_residual / no_layernorm / no_addnorm
4. CNN 对照：引入位置编码后，用卷积结构替代 self-attention 进行比较

输出内容：
- 每个实验的逐 epoch 训练记录 history.csv
- 每个实验的 test_id / test_ood 丰富指标 metrics.json
- 汇总表 summary.csv
- 曲线图：loss、teacher token accuracy、greedy token accuracy、sequence accuracy
- 按长度分组的 sequence accuracy: length_bucket_accuracy.csv

运行示例：
python transformer_ablation_rich_metrics.py --mode pe --epochs 8 --device cuda
python transformer_ablation_rich_metrics.py --mode all --epochs 12 --train-size 12000

说明：
本代码默认使用 synthetic reverse task，输入随机整数序列，目标为其反转序列。
该任务对顺序信息敏感，适合验证位置编码、QKV 分工和残差连接的必要性。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


PAD = 0
BOS = 1
EOS = 2


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReverseDataset(Dataset):
    def __init__(self, n_samples: int, min_len: int, max_len: int, vocab_size: int, seed: int):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        rng = random.Random(seed)
        for _ in range(n_samples):
            length = rng.randint(min_len, max_len)
            src = [rng.randint(3, vocab_size - 1) for _ in range(length)]
            tgt = list(reversed(src)) + [EOS]
            self.samples.append((torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def pad_1d(seqs: List[torch.Tensor], pad_value: int = PAD) -> torch.Tensor:
    max_len = max(len(x) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, x in enumerate(seqs):
        out[i, : len(x)] = x
    return out


def collate_reverse(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    srcs, tgts = zip(*batch)
    src = pad_1d(list(srcs), PAD)
    tgt = pad_1d(list(tgts), PAD)
    # decoder 输入为 BOS + target 去掉最后一位
    bos_col = torch.full((tgt.size(0), 1), BOS, dtype=torch.long)
    tgt_in = torch.cat([bos_col, tgt[:, :-1]], dim=1)
    return {"src": src, "tgt_in": tgt_in, "tgt_out": tgt}


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale


class NoPositionalEncoding(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SimpleAbsolutePositionalEncoding(nn.Module):
    """简单绝对位置编码：把 pos / max_len 加到各维度上。表达力弱，但可以提供粗略顺序。"""
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1) / max_len
        pe = pos.repeat(1, d_model).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        return x + self.pos_embedding(pos)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def build_pe(name: str, d_model: int, max_len: int) -> nn.Module:
    if name == "none":
        return NoPositionalEncoding()
    if name == "simple":
        return SimpleAbsolutePositionalEncoding(d_model, max_len)
    if name == "learned":
        return LearnedPositionalEncoding(d_model, max_len)
    if name == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len)
    raise ValueError(f"Unknown positional encoding: {name}")


class MultiHeadAttention(nn.Module):
    """
    qkv_mode 用于做 Q/K/V 必要性的消融：
    - separate: 标准做法，Q、K、V 三套独立线性映射
    - share_qk: Q 和 K 共用映射，检验 query/key 角色区分是否必要
    - share_kv: K 和 V 共用映射，检验“检索地址”和“被读取内容”是否需要分离
    - single_shared: Q/K/V 均共用一个映射，最强约束版本
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float, qkv_mode: str = "separate"):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv_mode = qkv_mode

        if qkv_mode == "separate":
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
        elif qkv_mode == "share_qk":
            self.qk_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
        elif qkv_mode == "share_kv":
            self.q_proj = nn.Linear(d_model, d_model)
            self.kv_proj = nn.Linear(d_model, d_model)
        elif qkv_mode == "single_shared":
            self.shared_proj = nn.Linear(d_model, d_model)
        else:
            raise ValueError(f"Unknown qkv_mode: {qkv_mode}")

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn: Optional[torch.Tensor] = None

    def _project(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.qkv_mode == "separate":
            return self.q_proj(q), self.k_proj(k), self.v_proj(v)
        if self.qkv_mode == "share_qk":
            return self.qk_proj(q), self.qk_proj(k), self.v_proj(v)
        if self.qkv_mode == "share_kv":
            kv = self.kv_proj(k)
            return self.q_proj(q), kv, kv
        if self.qkv_mode == "single_shared":
            return self.shared_proj(q), self.shared_proj(k), self.shared_proj(v)
        raise RuntimeError("unreachable")

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        q, k, v = self._project(q, k, v)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if attn_mask is not None:
            # attn_mask: [t_q, t_k], True 表示禁止关注
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: [batch, t_k], True 表示 PAD
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        self.last_attn = attn.detach().cpu() if need_weights else None
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(out.size(0), -1, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SublayerConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float, residual: bool = True, layernorm: bool = True):
        super().__init__()
        self.residual = residual
        self.layernorm = layernorm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model) if layernorm else nn.Identity()

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        y = self.dropout(sublayer(x))
        if self.residual:
            y = x + y
        if self.layernorm:
            y = self.norm(y)
        return y


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, qkv_mode: str, residual: bool, layernorm: bool):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, qkv_mode)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout, residual, layernorm)
        self.sublayer2 = SublayerConnection(d_model, dropout, residual, layernorm)

    def forward(self, x: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer1(x, lambda z: self.self_attn(z, z, z, key_padding_mask=src_pad_mask))
        x = self.sublayer2(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, qkv_mode: str, residual: bool, layernorm: bool):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, qkv_mode)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, qkv_mode)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout, residual, layernorm)
        self.sublayer2 = SublayerConnection(d_model, dropout, residual, layernorm)
        self.sublayer3 = SublayerConnection(d_model, dropout, residual, layernorm)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_causal_mask: torch.Tensor,
        need_cross_weights: bool = False,
    ) -> torch.Tensor:
        x = self.sublayer1(x, lambda z: self.self_attn(z, z, z, attn_mask=tgt_causal_mask))
        x = self.sublayer2(
            x,
            lambda z: self.cross_attn(z, memory, memory, key_padding_mask=src_pad_mask, need_weights=need_cross_weights),
        )
        x = self.sublayer3(x, self.ffn)
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        pe_type: str,
        qkv_mode: str,
        residual: bool,
        layernorm: bool,
    ):
        super().__init__()
        self.src_embed = TokenEmbedding(vocab_size, d_model)
        self.tgt_embed = TokenEmbedding(vocab_size, d_model)
        self.pos = build_pe(pe_type, d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, qkv_mode, residual, layernorm)
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, qkv_mode, residual, layernorm)
            for _ in range(num_layers)
        ])
        self.generator = nn.Linear(d_model, vocab_size)

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_pad_mask = src.eq(PAD)
        x = self.dropout(self.pos(self.src_embed(src)))
        for layer in self.encoder:
            x = layer(x, src_pad_mask)
        return x, src_pad_mask

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_pad_mask: torch.Tensor, need_cross_weights: bool = False) -> torch.Tensor:
        t = tgt_in.size(1)
        tgt_causal_mask = torch.triu(torch.ones(t, t, device=tgt_in.device, dtype=torch.bool), diagonal=1)
        x = self.dropout(self.pos(self.tgt_embed(tgt_in)))
        for i, layer in enumerate(self.decoder):
            x = layer(x, memory, src_pad_mask, tgt_causal_mask, need_cross_weights=(need_cross_weights and i == len(self.decoder) - 1))
        return x

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor, need_cross_weights: bool = False) -> torch.Tensor:
        memory, src_pad_mask = self.encode(src)
        dec = self.decode(tgt_in, memory, src_pad_mask, need_cross_weights=need_cross_weights)
        return self.generator(dec)

    def get_last_cross_attention(self) -> Optional[torch.Tensor]:
        if len(self.decoder) == 0:
            return None
        return self.decoder[-1].cross_attn.last_attn


class CNNSeq2Seq(nn.Module):
    """卷积对照组：有位置编码，但不使用 self-attention/cross-attention。"""
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, d_ff: int, dropout: float, max_len: int, pe_type: str):
        super().__init__()
        self.src_embed = TokenEmbedding(vocab_size, d_model)
        self.tgt_embed = TokenEmbedding(vocab_size, d_model)
        self.pos = build_pe(pe_type, d_model, max_len)
        self.src_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1) for _ in range(num_layers)
        ])
        self.tgt_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1) for _ in range(num_layers)
        ])
        self.ctx_proj = nn.Linear(d_model, d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor, need_cross_weights: bool = False) -> torch.Tensor:
        src_mask = src.ne(PAD).float().unsqueeze(-1)
        x = self.pos(self.src_embed(src))
        x = x.transpose(1, 2)
        for conv in self.src_convs:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        # 源端上下文采用 masked mean。这个设计故意比 cross-attention 弱，用来观察 CNN 替代注意力的代价。
        ctx = (x * src_mask).sum(dim=1) / src_mask.sum(dim=1).clamp_min(1.0)
        ctx = self.ctx_proj(ctx).unsqueeze(1)

        y = self.pos(self.tgt_embed(tgt_in)) + ctx
        y = y.transpose(1, 2)
        for conv in self.tgt_convs:
            y = F.relu(conv(y))
        y = y.transpose(1, 2)
        y = self.norm(y + self.dropout(self.ffn(y)))
        return self.generator(y)

    def get_last_cross_attention(self) -> None:
        return None


@dataclass
class ExperimentConfig:
    name: str
    model_type: str = "transformer"
    pe_type: str = "sinusoidal"
    qkv_mode: str = "separate"
    residual: bool = True
    layernorm: bool = True


def get_experiments(mode: str) -> List[ExperimentConfig]:
    pe = [
        ExperimentConfig("pe_none", pe_type="none"),
        ExperimentConfig("pe_simple", pe_type="simple"),
        ExperimentConfig("pe_learned", pe_type="learned"),
        ExperimentConfig("pe_sinusoidal", pe_type="sinusoidal"),
    ]
    residual = [
        ExperimentConfig("standard_add_norm", pe_type="sinusoidal", residual=True, layernorm=True),
        ExperimentConfig("no_residual", pe_type="sinusoidal", residual=False, layernorm=True),
        ExperimentConfig("no_layernorm", pe_type="sinusoidal", residual=True, layernorm=False),
        ExperimentConfig("no_addnorm", pe_type="sinusoidal", residual=False, layernorm=False),
    ]
    qkv = [
        ExperimentConfig("qkv_separate", qkv_mode="separate"),
        ExperimentConfig("qk_shared", qkv_mode="share_qk"),
        ExperimentConfig("kv_shared", qkv_mode="share_kv"),
        ExperimentConfig("qkv_single_shared", qkv_mode="single_shared"),
    ]
    cnn = [
        ExperimentConfig("cnn_with_sinusoidal_pe", model_type="cnn", pe_type="sinusoidal"),
    ]
    if mode == "pe":
        return pe
    if mode == "residual":
        return residual
    if mode == "qkv":
        return qkv
    if mode == "cnn":
        return [ExperimentConfig("qkv_separate")] + cnn
    if mode == "all":
        # 去重，保留首次出现
        seen = set()
        out = []
        for e in pe + residual + qkv + cnn:
            if e.name not in seen:
                out.append(e)
                seen.add(e.name)
        return out
    raise ValueError(f"Unknown mode: {mode}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def noam_lr(step: int, d_model: int, warmup: int) -> float:
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))


def edit_distance(a: List[int], b: List[int]) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            if ca == cb:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def strip_after_eos(seq: List[int], keep_eos: bool = True) -> List[int]:
    out = []
    for x in seq:
        if x == PAD:
            break
        out.append(int(x))
        if x == EOS:
            break
    if not keep_eos:
        out = [x for x in out if x != EOS]
    return out


@torch.no_grad()
def greedy_decode(model: nn.Module, src: torch.Tensor, max_len: int) -> torch.Tensor:
    model.eval()
    bsz = src.size(0)
    ys = torch.full((bsz, 1), BOS, dtype=torch.long, device=src.device)
    finished = torch.zeros(bsz, dtype=torch.bool, device=src.device)
    outputs = []
    for _ in range(max_len):
        logits = model(src, ys)
        next_tok = logits[:, -1, :].argmax(dim=-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, PAD), next_tok)
        outputs.append(next_tok)
        finished |= next_tok.eq(EOS)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
    return torch.stack(outputs, dim=1)


def compute_sequence_metrics(pred: torch.Tensor, tgt: torch.Tensor, src: torch.Tensor) -> Dict[str, float]:
    pred_np = pred.detach().cpu().numpy().tolist()
    tgt_np = tgt.detach().cpu().numpy().tolist()
    src_np = src.detach().cpu().numpy().tolist()

    total_tokens = 0
    correct_tokens = 0
    exact = 0
    eos_correct = 0
    length_correct = 0
    bag_correct = 0
    prefix_sum = 0.0
    edit_sum = 0.0
    norm_edit_sum = 0.0
    length_bucket: Dict[int, List[int]] = {}

    for p_raw, t_raw, s_raw in zip(pred_np, tgt_np, src_np):
        true_with_eos = strip_after_eos(t_raw, keep_eos=True)
        pred_with_eos = strip_after_eos(p_raw, keep_eos=True)
        true_content = strip_after_eos(t_raw, keep_eos=False)
        pred_content = strip_after_eos(p_raw, keep_eos=False)
        src_len = sum(1 for x in s_raw if x != PAD)

        padded_pred = pred_with_eos[: len(true_with_eos)] + [PAD] * max(0, len(true_with_eos) - len(pred_with_eos))
        for a, b in zip(padded_pred, true_with_eos):
            if b != PAD:
                total_tokens += 1
                correct_tokens += int(a == b)

        is_exact = int(pred_with_eos == true_with_eos)
        exact += is_exact
        length_bucket.setdefault(src_len, []).append(is_exact)

        true_eos_pos = true_with_eos.index(EOS) if EOS in true_with_eos else -1
        pred_eos_pos = pred_with_eos.index(EOS) if EOS in pred_with_eos else -2
        eos_correct += int(true_eos_pos == pred_eos_pos)
        length_correct += int(len(pred_content) == len(true_content))
        bag_correct += int(sorted(pred_content) == sorted(true_content))

        prefix = 0
        for a, b in zip(padded_pred, true_with_eos):
            if a == b:
                prefix += 1
            else:
                break
        prefix_sum += prefix / max(1, len(true_with_eos))

        ed = edit_distance(pred_content, true_content)
        edit_sum += ed
        norm_edit_sum += ed / max(1, len(true_content))

    n = len(tgt_np)
    length_rows = []
    for length, vals in sorted(length_bucket.items()):
        length_rows.append({"length": length, "n": len(vals), "sequence_accuracy": float(np.mean(vals))})

    return {
        "greedy_token_acc": correct_tokens / max(1, total_tokens),
        "sequence_acc": exact / max(1, n),
        "eos_position_acc": eos_correct / max(1, n),
        "length_acc": length_correct / max(1, n),
        "bag_acc": bag_correct / max(1, n),
        "mean_prefix_acc": prefix_sum / max(1, n),
        "mean_edit_distance": edit_sum / max(1, n),
        "normalized_edit_distance": norm_edit_sum / max(1, n),
        "length_bucket_rows": length_rows,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, vocab_size: int) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    total_loss = 0.0
    n_batches = 0
    teacher_correct = 0
    teacher_total = 0

    all_pred, all_tgt, all_src = [], [], []
    for batch in loader:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        total_loss += float(loss.item())
        n_batches += 1

        pred_teacher = logits.argmax(dim=-1)
        mask = tgt_out.ne(PAD)
        teacher_correct += int((pred_teacher.eq(tgt_out) & mask).sum().item())
        teacher_total += int(mask.sum().item())

        max_len = tgt_out.size(1)
        pred_greedy = greedy_decode(model, src, max_len=max_len)
        all_pred.append(pred_greedy.cpu())
        all_tgt.append(tgt_out.cpu())
        all_src.append(src.cpu())

    pred = torch.cat(all_pred, dim=0)
    tgt = torch.cat(all_tgt, dim=0)
    src = torch.cat(all_src, dim=0)
    seq_metrics = compute_sequence_metrics(pred, tgt, src)
    length_bucket_rows = seq_metrics.pop("length_bucket_rows")

    return {
        "loss": total_loss / max(1, n_batches),
        "teacher_token_acc": teacher_correct / max(1, teacher_total),
        **seq_metrics,
        "length_bucket_rows": length_bucket_rows,
    }


def build_model(cfg: ExperimentConfig, args: argparse.Namespace) -> nn.Module:
    if cfg.model_type == "transformer":
        return Seq2SeqTransformer(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_heads=args.heads,
            num_layers=args.layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_len=args.max_position,
            pe_type=cfg.pe_type,
            qkv_mode=cfg.qkv_mode,
            residual=cfg.residual,
            layernorm=cfg.layernorm,
        )
    if cfg.model_type == "cnn":
        return CNNSeq2Seq(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_len=args.max_position,
            pe_type=cfg.pe_type,
        )
    raise ValueError(f"Unknown model type: {cfg.model_type}")


def train_one_experiment(cfg: ExperimentConfig, args: argparse.Namespace, loaders: Dict[str, DataLoader]) -> Dict[str, float]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    exp_dir = os.path.join(args.output_dir, cfg.name)
    os.makedirs(exp_dir, exist_ok=True)

    model = build_model(cfg, args).to(device)
    n_params = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    history = []
    global_step = 0
    start_time = time.time()
    best_valid_seq = -1.0
    best_epoch = -1
    grad_norm_values: List[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        epoch_grad_norms = []
        token_count = 0
        epoch_start = time.time()

        for batch in loaders["train"]:
            global_step += 1
            if args.schedule == "noam":
                lr = noam_lr(global_step, args.d_model, args.warmup_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr

            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, args.vocab_size), tgt_out.reshape(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1
            epoch_grad_norms.append(float(grad_norm))
            grad_norm_values.append(float(grad_norm))
            token_count += int(tgt_out.ne(PAD).sum().item())

        valid_metrics = evaluate(model, loaders["valid"], device, args.vocab_size)
        train_loss = train_loss_sum / max(1, train_batches)
        tokens_per_sec = token_count / max(1e-6, time.time() - epoch_start)
        grad_mean = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
        grad_max = float(np.max(epoch_grad_norms)) if epoch_grad_norms else 0.0

        row = {
            "experiment": cfg.name,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_teacher_token_acc": valid_metrics["teacher_token_acc"],
            "valid_greedy_token_acc": valid_metrics["greedy_token_acc"],
            "valid_sequence_acc": valid_metrics["sequence_acc"],
            "valid_eos_position_acc": valid_metrics["eos_position_acc"],
            "valid_length_acc": valid_metrics["length_acc"],
            "valid_bag_acc": valid_metrics["bag_acc"],
            "valid_mean_prefix_acc": valid_metrics["mean_prefix_acc"],
            "valid_mean_edit_distance": valid_metrics["mean_edit_distance"],
            "valid_normalized_edit_distance": valid_metrics["normalized_edit_distance"],
            "grad_norm_mean": grad_mean,
            "grad_norm_max": grad_max,
            "tokens_per_sec": tokens_per_sec,
            "lr": optimizer.param_groups[0]["lr"],
            "n_params": n_params,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(exp_dir, "history.csv"), index=False, encoding="utf-8-sig")

        if valid_metrics["sequence_acc"] > best_valid_seq:
            best_valid_seq = valid_metrics["sequence_acc"]
            best_epoch = epoch
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))

        print(
            f"[{cfg.name}] epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} valid_loss={valid_metrics['loss']:.4f} "
            f"teacher={valid_metrics['teacher_token_acc']:.3f} "
            f"greedy={valid_metrics['greedy_token_acc']:.3f} seq={valid_metrics['sequence_acc']:.3f} "
            f"edit={valid_metrics['normalized_edit_distance']:.3f} grad={grad_mean:.2f}"
        )

    test_id = evaluate(model, loaders["test_id"], device, args.vocab_size)
    test_ood = evaluate(model, loaders["test_ood"], device, args.vocab_size)

    # 保存按长度分组的结果
    id_buckets = pd.DataFrame(test_id.pop("length_bucket_rows"))
    ood_buckets = pd.DataFrame(test_ood.pop("length_bucket_rows"))
    if not id_buckets.empty:
        id_buckets["split"] = "test_id"
    if not ood_buckets.empty:
        ood_buckets["split"] = "test_ood"
    pd.concat([id_buckets, ood_buckets], ignore_index=True).to_csv(
        os.path.join(exp_dir, "length_bucket_accuracy.csv"), index=False, encoding="utf-8-sig"
    )

    total_time = time.time() - start_time
    final = {
        **asdict(cfg),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_valid_sequence_acc": best_valid_seq,
        "train_time_sec": total_time,
        "grad_norm_global_mean": float(np.mean(grad_norm_values)) if grad_norm_values else 0.0,
        "grad_norm_global_max": float(np.max(grad_norm_values)) if grad_norm_values else 0.0,
    }
    for k, v in test_id.items():
        final[f"test_id_{k}"] = v
    for k, v in test_ood.items():
        final[f"test_ood_{k}"] = v
    final["ood_sequence_gap"] = final["test_id_sequence_acc"] - final["test_ood_sequence_acc"]
    final["ood_edit_gap"] = final["test_ood_normalized_edit_distance"] - final["test_id_normalized_edit_distance"]

    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    return final


def make_loaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    train = ReverseDataset(args.train_size, args.train_min_len, args.train_max_len, args.vocab_size, args.seed + 1)
    valid = ReverseDataset(args.valid_size, args.train_min_len, args.train_max_len, args.vocab_size, args.seed + 2)
    test_id = ReverseDataset(args.test_size, args.train_min_len, args.train_max_len, args.vocab_size, args.seed + 3)
    test_ood = ReverseDataset(args.test_size, args.ood_min_len, args.ood_max_len, args.vocab_size, args.seed + 4)
    kwargs = dict(batch_size=args.batch_size, collate_fn=collate_reverse, num_workers=0)
    return {
        "train": DataLoader(train, shuffle=True, **kwargs),
        "valid": DataLoader(valid, shuffle=False, **kwargs),
        "test_id": DataLoader(test_id, shuffle=False, **kwargs),
        "test_ood": DataLoader(test_ood, shuffle=False, **kwargs),
    }


def plot_summary(output_dir: str, experiments: List[ExperimentConfig]) -> None:
    if plt is None:
        return
    histories = []
    for cfg in experiments:
        path = os.path.join(output_dir, cfg.name, "history.csv")
        if os.path.exists(path):
            histories.append(pd.read_csv(path))
    if not histories:
        return
    df = pd.concat(histories, ignore_index=True)
    plot_items = [
        ("valid_loss", "Validation Loss", "valid_loss.png"),
        ("valid_teacher_token_acc", "Teacher Token Accuracy", "teacher_token_acc.png"),
        ("valid_greedy_token_acc", "Greedy Token Accuracy", "greedy_token_acc.png"),
        ("valid_sequence_acc", "Sequence Accuracy", "sequence_acc.png"),
        ("valid_normalized_edit_distance", "Normalized Edit Distance", "normalized_edit_distance.png"),
    ]
    for col, title, filename in plot_items:
        plt.figure(figsize=(8, 5))
        for name, g in df.groupby("experiment"):
            plt.plot(g["epoch"], g[col], marker="o", label=name)
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pe", "residual", "qkv", "cnn", "all"], default="all")
    parser.add_argument("--output-dir", default="outputs_prml_transformer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=40)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--valid-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--train-min-len", type=int, default=5)
    parser.add_argument("--train-max-len", type=int, default=20)
    parser.add_argument("--ood-min-len", type=int, default=21)
    parser.add_argument("--ood-max-len", type=int, default=40)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-position", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--schedule", choices=["constant", "noam"], default="noam")
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    loaders = make_loaders(args)
    experiments = get_experiments(args.mode)
    summary_rows = []
    for cfg in experiments:
        row = train_one_experiment(cfg, args, loaders)
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, "summary.csv"), index=False, encoding="utf-8-sig")

    plot_summary(args.output_dir, experiments)
    print(f"\nDone. Results saved to: {args.output_dir}")
    print("重点看 summary.csv 中的 test_id_sequence_acc、test_ood_sequence_acc、normalized_edit_distance、ood_sequence_gap。")


if __name__ == "__main__":
    main()
