"""
Надёжный скрипт обучения и Парето-поиска для малой MoE-модели
в стиле Zyphra ZAYA1-8B.

Режимы:
  train        обучение одной конфигурации
  pareto       Парето-поиск конфигураций
  refine       уточнение фронтирных моделей
  build-vocab  сборка словаря
  tokenize     токенизация текста в uint16
"""

import os
import sys
import math
import json
import time
import gc
import random
import argparse
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

os.environ.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

try:
    from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
#  Технические настройки
# ─────────────────────────────────────────────────────────────

def harden_backends():
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(e: Exception) -> bool:
    oom_exc = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_exc is not None and isinstance(e, oom_exc)) or "out of memory" in str(e).lower()


def create_parent(path: str):
    p = Path(path).parent
    if str(p):
        p.mkdir(parents=True, exist_ok=True)


def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def make_optimizer(model, args):
    lr = args.lr
    betas = (0.9, 0.95)
    wd = args.weight_decay

    if args.use_8bit:
        if not BNB_AVAILABLE:
            log("[WARN] bitsandbytes не установлен: pip install bitsandbytes")
            return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)

        if args.lion:
            if args.paged:
                return bnb.optim.PagedLion8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
            else:
                return bnb.optim.Lion8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
        else:
            if args.paged:
                return bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
            else:
                return bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=betas, weight_decay=wd)

    if args.lion:
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
        except ImportError:
            log("[WARN] lion-pytorch не установлен: pip install lion-pytorch")
            return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)

    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)


def resolve_precision(args) -> str:
    if args.fp32:
        return "fp32"
    if args.strict_fp16:
        return "fp16"
    if args.bf16:
        return "bf16"
    return "mixed"


def model_forward(model, x, y, precision: str):
    if DEVICE == "cuda" and precision == "mixed":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x, y)
    elif DEVICE == "cuda" and precision == "bf16":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(x, y)
    return model(x, y)


# ─────────────────────────────────────────────────────────────
#  Логирование
# ─────────────────────────────────────────────────────────────

LOG_FILE = None


def init_log(log_path: str):
    global LOG_FILE
    create_parent(log_path)
    LOG_FILE = open(log_path, "a", encoding="utf-8")


def log(msg: str):
    print(msg)
    if LOG_FILE:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        LOG_FILE.write(f"[{timestamp}] {msg}\n")
        LOG_FILE.flush()


def close_log():
    global LOG_FILE
    if LOG_FILE:
        LOG_FILE.close()
        LOG_FILE = None


# ─────────────────────────────────────────────────────────────
#  Конфигурация модели
# ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    vocab_size: int = 16384
    dim: int = 256
    layers: int = 24
    heads: int = 8
    experts: int = 8
    expert_mult: float = 1.3
    router_dim: int = 192
    seq_len: int = 256
    batch_size: int = 1
    accum: int = 8
    cca_comp: int = 4
    checkpoint: bool = False
    zloss: float = 0.0003
    aux_weight: float = 0.01

    @property
    def latent(self) -> int:
        return self.dim // self.cca_comp

    def __post_init__(self):
        if self.cca_comp <= 0 or self.dim % self.cca_comp != 0:
            raise ValueError("dim должен делиться на cca_comp")
        if self.latent % self.heads != 0:
            raise ValueError("dim // cca_comp должен делиться на heads")


# ─────────────────────────────────────────────────────────────
#  Численно устойчивые примитивы
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        ms = x32.pow(2).mean(-1, keepdim=True).add(self.eps)
        return (x32 * torch.rsqrt(ms) * self.weight.float()).to(x.dtype)


def precompute_rope(dim: int, max_pos: int = 8192, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope_half(x, cos, sin):
    dtype = x.dtype
    x = x.float()
    D = x.shape[-1]
    rot_dim = max(2, ((D // 2) // 2) * 2)
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    half = rot_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    T = x.shape[2]
    c = cos[:T, :half].to(device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    s = sin[:T, :half].to(device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    out_rot = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    out = torch.cat([out_rot, x_pass], dim=-1)
    return out.to(dtype)


# ─────────────────────────────────────────────────────────────
#  Архитектурные блоки
# ─────────────────────────────────────────────────────────────

class CCA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        lat = cfg.latent
        self.latent = lat
        self.heads = cfg.heads
        self.head_dim = lat // cfg.heads

        self.down = nn.Linear(cfg.dim, lat, bias=False)
        self.conv_q = nn.Conv1d(lat, lat, kernel_size=3, padding=1, groups=lat, bias=False)
        self.conv_k = nn.Conv1d(lat, lat, kernel_size=3, padding=1, groups=lat, bias=False)
        self.q_proj = nn.Linear(lat, lat, bias=False)
        self.k_proj = nn.Linear(lat, lat, bias=False)
        self.v_proj = nn.Linear(lat, lat, bias=False)
        self.up = nn.Linear(lat, cfg.dim, bias=False)
        self.q_norm = RMSNorm(lat)
        self.k_norm = RMSNorm(lat)
        self.temp = nn.Parameter(torch.ones(cfg.heads) * (self.head_dim ** -0.25))

        cos, sin = precompute_rope(self.head_dim, max_pos=max(cfg.seq_len * 4, 2048))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.down(x)
        hc = h.transpose(1, 2)
        qh = self.conv_q(hc).transpose(1, 2)
        kh = self.conv_k(hc).transpose(1, 2)
        prev = torch.cat([torch.zeros_like(h[:, :1, :]), h[:, :-1, :]], dim=1)
        vh = qh + prev
        q = self.q_norm(self.q_proj(qh))
        k = self.k_norm(self.k_proj(kh))
        v = self.v_proj(vh)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q = apply_rope_half(q, self.rope_cos, self.rope_sin)
        k = apply_rope_half(k, self.rope_cos, self.rope_sin)
        q = q * self.temp.view(1, -1, 1, 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, T, self.latent)
        return self.up(out)


class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.E = cfg.experts
        ed = int(cfg.dim * cfg.expert_mult)
        self.experts = nn.ModuleList([Expert(cfg.dim, ed) for _ in range(cfg.experts)])
        self.r_down = nn.Linear(cfg.dim, cfg.router_dim, bias=False)
        self.r_norm = RMSNorm(cfg.router_dim)
        self.r_up = nn.Linear(cfg.router_dim, cfg.experts, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.router_bias = nn.Parameter(torch.zeros(cfg.experts))

    def forward(self, x, prev_r):
        B, T, D = x.shape
        flat = x.view(-1, D)
        r = self.r_down(flat)
        if prev_r is not None:
            r = r + self.gamma * prev_r
        logits = self.r_up(self.r_norm(r)) + self.router_bias
        probs = F.softmax(logits.float(), dim=-1)
        top_p, top_i = probs.max(dim=-1)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            idx = (top_i == e).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                out[idx] = out[idx] + top_p[idx, None].to(flat.dtype) * expert(flat[idx])
        frac = torch.zeros(self.E, dtype=torch.float32, device=flat.device)
        frac.scatter_add_(0, top_i, torch.ones_like(top_i, dtype=torch.float32))
        frac = frac / max(1, flat.size(0))
        aux = self.E * (frac * probs.mean(dim=0)).sum()
        return out.view(B, T, D), aux, r


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = CCA(cfg)
        self.alpha_res = nn.Parameter(torch.ones(cfg.dim))
        self.beta_res = nn.Parameter(torch.zeros(cfg.dim))
        self.moe_norm = RMSNorm(cfg.dim)
        self.moe = MoE(cfg)
        self.alpha_out = nn.Parameter(torch.ones(cfg.dim))
        self.beta_out = nn.Parameter(torch.zeros(cfg.dim))

    def forward(self, x, router_state):
        h = self.attn(self.attn_norm(x))
        x = self.alpha_res * x + self.beta_res + h
        h, aux, router_state = self.moe(self.moe_norm(x), router_state)
        x = self.alpha_out * x + self.beta_out + h
        return x, aux, router_state


class NanoZaya(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids, targets=None):
        x = self.wte(input_ids)
        B, T = input_ids.shape
        router_state = torch.zeros(B * T, self.cfg.router_dim, dtype=x.dtype, device=x.device)
        aux_total = torch.zeros((), dtype=torch.float32, device=x.device)
        for blk in self.blocks:
            if self.cfg.checkpoint and self.training:
                x, aux, router_state = torch.utils.checkpoint.checkpoint(blk, x, router_state, use_reentrant=False)
            else:
                x, aux, router_state = blk(x, router_state)
            aux_total = aux_total + aux.float()
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            logits32 = torch.clamp(logits.float(), -50.0, 50.0)
            ce = F.cross_entropy(logits32.view(-1, self.cfg.vocab_size), targets.view(-1), ignore_index=-1)
            lse = torch.logsumexp(logits32, dim=-1)
            z = self.cfg.zloss * lse.pow(2).mean()
            loss = ce + z + self.cfg.aux_weight * aux_total
        return logits, loss


def init_weights(model: nn.Module):
    for name, p in model.named_parameters():
        if p.dim() >= 2 and not any(k in name for k in ("alpha", "beta", "temp")):
            try:
                nn.init.normal_(p, mean=0.0, std=0.02)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
#  Manual Scaler
# ─────────────────────────────────────────────────────────────

class ManualScaler:
    def __init__(self, init_scale=1024.0, growth_interval=2000, growth_factor=2.0,
                 backoff_factor=0.5, min_scale=1.0, max_scale=65536.0):
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_and_check(self, model):
        inv = 1.0 / self.scale
        finite = True
        total = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.float() * inv
            if not torch.isfinite(g).all():
                finite = False
                break
            total += g.pow(2).sum().item()
        if finite:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data = (p.grad.float() * inv).to(p.grad.dtype)
        return finite, math.sqrt(total) if finite else float("inf")

    def update(self, finite: bool):
        if finite:
            self._steps += 1
            if self._steps >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self._steps = 0
        else:
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self._steps = 0

    def state_dict(self):
        return {"scale": self.scale, "steps": self._steps}

    def load_state_dict(self, d):
        self.scale = float(d.get("scale", self.scale))
        self._steps = int(d.get("steps", 0))


# ─────────────────────────────────────────────────────────────
#  Словарь и токенизатор
# ─────────────────────────────────────────────────────────────

SPECIAL = ["<pad>", "<unk>", "<s>", "</s>", "<sep>"]
BYTE_BASE = len(SPECIAL)


def byte_token(b: int) -> str:
    return chr(0xE000 + b)


def build_vocab(paths, vocab_size: int = 16384, **kwargs):
    vocab, seen = [], set()

    def add(tok):
        if tok not in seen and len(vocab) < vocab_size:
            vocab.append(tok)
            seen.add(tok)

    for tok in SPECIAL:
        add(tok)
    for b in range(256):
        add(byte_token(b))

    alphabet = []
    for ch in "абвгдежзийклмнопрстуфхцчшщъыьэюя":
        alphabet.append(ch)
        alphabet.append(ch.upper())
    for ch in "ѣіѳѵ":
        alphabet.append(ch)
        alphabet.append(ch.upper())
    for ch in ".,;:!?()[]{}\"'«»—–- ":
        alphabet.append(ch)
    for ch in alphabet:
        add(ch)

    while len(vocab) < vocab_size:
        add(f"<unused:{len(vocab)}>")
    return vocab[:vocab_size]


def save_vocab(vocab, path: str):
    create_parent(path)
    Path(path).write_text("\n".join(vocab) + "\n", encoding="utf-8")


def load_vocab(path: str):
    return Path(path).read_text(encoding="utf-8").splitlines()


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token2id = {t: i for i, t in enumerate(vocab)}

    def encode(self, text: str):
        ids = []
        for ch in text:
            if ch in self.token2id:
                ids.append(self.token2id[ch])
            else:
                for b in ch.encode("utf-8"):
                    ids.append(BYTE_BASE + b)
        return ids


def tokenize_files(paths, tokenizer, out_path: str, chunk_chars: int = 1 << 20):
    create_parent(out_path)
    with open(out_path, "wb") as out:
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    text = f.read(chunk_chars)
                    if not text:
                        break
                    ids = tokenizer.encode(text)
                    np.array(ids, dtype=np.uint16).tofile(out)


# ─────────────────────────────────────────────────────────────
#  Датасет
# ─────────────────────────────────────────────────────────────

class TokenFileDataset(IterableDataset):
    def __init__(self, path: str, seq_len: int, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.path = path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.mm = np.memmap(path, dtype=np.uint16, mode="r")
        self.n_chunks = max(0, (len(self.mm) - 1) // (seq_len + 1)) if len(self.mm) > seq_len else 0

    def _make_batch(self, idxs):
        xs = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        ys = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        for j, idx in enumerate(idxs):
            arr = self.mm[int(idx) * (self.seq_len + 1): int(idx) * (self.seq_len + 1) + self.seq_len + 1]
            xs[j] = arr[:-1].astype(np.int64)
            ys[j] = arr[1:].astype(np.int64)
        return torch.from_numpy(xs), torch.from_numpy(ys)

    def __iter__(self):
        if self.n_chunks < self.batch_size:
            return
        rng = np.random.default_rng(self.seed)
        while True:
            if self.shuffle:
                for _ in range(max(1, self.n_chunks // self.batch_size)):
                    yield self._make_batch(rng.integers(0, self.n_chunks, size=self.batch_size))
            else:
                for start in range(0, self.n_chunks - self.batch_size + 1, self.batch_size):
                    yield self._make_batch(np.arange(start, start + self.batch_size))


def make_dummy_token_file(path: str, n_tokens: int, vocab_size: int = 16384, seed: int = 0):
    create_parent(path)
    np.random.default_rng(seed).integers(0, vocab_size, size=n_tokens, dtype=np.uint16).tofile(path)


# ─────────────────────────────────────────────────────────────
#  Чекпоинты
# ─────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    files = glob.glob(str(Path(ckpt_dir) / "step_*.ckpt.json"))
    if not files:
        files = glob.glob(str(Path(ckpt_dir) / "step_*.pt"))
    if not files:
        return None

    def step_of(path: str) -> int:
        m = re.search(r"step_(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(files, key=step_of)


def save_checkpoint(model, opt, scaler, manual, path: str, extra: Dict) -> bool:
    try:
        create_parent(path)
        base = path.replace(".pt", "").replace(".ckpt.json", "").replace(".tmp", "")
        weights_path = base + ".model.safetensors"
        meta_path = base + ".ckpt.json"

        if SAFETENSORS_AVAILABLE:
            state = model.state_dict()
            safetensors_save(state, weights_path)
            meta = {"model": Path(weights_path).name, "extra": extra}
            if opt is not None:
                opt_path = base + ".opt.pt"
                torch.save({"optim": opt.state_dict(),
                            "scaler": scaler.state_dict() if scaler else None,
                            "manual": manual.state_dict() if manual else None},
                           opt_path)
                meta["optimizer"] = Path(opt_path).name
            Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            state = {
                "model": model.state_dict(),
                "optim": opt.state_dict() if opt else None,
                "scaler": scaler.state_dict() if scaler else None,
                "manual": manual.state_dict() if manual else None,
                "extra": extra
            }
            tmp = path + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, path)

        return True

    except Exception as e:
        log(f"[WARN] Ошибка сохранения чекпоинта: {e}")
        return False


# ─────────────────────────────────────────────────────────────
#  Обучение (Train)
# ─────────────────────────────────────────────────────────────

def config_from_args(args) -> Config:
    return Config(
        vocab_size=args.vocab_size, dim=args.dim, layers=args.layers, heads=args.heads,
        experts=args.experts, expert_mult=args.expert_mult, router_dim=args.router_dim,
        seq_len=args.seq, batch_size=args.batch, accum=args.accum, cca_comp=args.cca_comp,
        checkpoint=args.checkpoint
    )


def train(args):
    cfg = config_from_args(args)
    precision = resolve_precision(args)

    if not Path(args.data).exists():
        log(f"[WARN] Файл токенов {args.data} не найден, создаю фиктивный.")
        make_dummy_token_file(args.data, max(2_000_000, (cfg.seq_len + 1) * cfg.batch_size * 1000),
                              cfg.vocab_size, args.seed)

    model = NanoZaya(cfg)
    init_weights(model)

    if DEVICE == "cuda":
        if precision == "fp16":
            model = model.half().to(DEVICE)
        elif precision == "bf16":
            model = model.bfloat16().to(DEVICE)
        else:
            model = model.to(DEVICE)
    else:
        model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())

    if args.use_8bit:
        if args.lion:
            opt_type = "PagedLion8bit" if args.paged else "Lion8bit"
        else:
            opt_type = "PagedAdamW8bit" if args.paged else "AdamW8bit"
    elif args.lion:
        opt_type = "Lion"
    else:
        opt_type = "AdamW"

    log(f"[INFO] dim={cfg.dim} lay={cfg.layers} exp={cfg.experts} heads={cfg.heads} "
        f"seq={cfg.seq_len} bs={cfg.batch_size} | {n_params/1e6:.1f}M | {precision} | {opt_type} | {DEVICE}")

    opt = make_optimizer(model, args)
    scaler = make_grad_scaler() if DEVICE == "cuda" and precision in ("mixed", "bf16") and not args.disable_scaler else None
    manual = ManualScaler() if precision == "fp16" and not args.disable_scaler else None

    start_step = 0
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt_path = find_latest_checkpoint(args.ckpt_dir) if args.resume.lower() == "auto" else args.resume
        if ckpt_path and Path(ckpt_path).exists():
            try:
                if ckpt_path.endswith(".ckpt.json"):
                    meta = json.loads(Path(ckpt_path).read_text(encoding="utf-8"))
                    weights_path = str(Path(ckpt_path).parent / meta["model"])
                    if SAFETENSORS_AVAILABLE:
                        state = safetensors_load(weights_path)
                        model.load_state_dict(state)
                    else:
                        log("[ERROR] safetensors не установлен: pip install safetensors")
                        sys.exit(1)
                    if "optimizer" in meta:
                        opt_path = str(Path(ckpt_path).parent / meta["optimizer"])
                        if Path(opt_path).exists():
                            opt_state = torch.load(opt_path, map_location=DEVICE)
                            if "optim" in opt_state:
                                opt.load_state_dict(opt_state["optim"])
                            if scaler and opt_state.get("scaler"):
                                scaler.load_state_dict(opt_state["scaler"])
                            if manual and opt_state.get("manual"):
                                manual.load_state_dict(opt_state["manual"])
                    extra = meta.get("extra", {})
                    old_cfg = extra.get("cfg", {})
                else:
                    state = torch.load(ckpt_path, map_location=DEVICE)
                    model.load_state_dict(state["model"])
                    if "optim" in state:
                        opt.load_state_dict(state["optim"])
                    if scaler and state.get("scaler"):
                        scaler.load_state_dict(state["scaler"])
                    if manual and state.get("manual"):
                        manual.load_state_dict(state["manual"])
                    extra = state.get("extra", {})
                    old_cfg = extra.get("cfg", {})

                for k in ["vocab_size", "dim", "layers", "heads", "experts", "expert_mult", "router_dim", "cca_comp"]:
                    if old_cfg.get(k) != getattr(cfg, k):
                        log(f"[ERROR] Конфигурация чекпоинта не совпадает по полю {k}")
                        sys.exit(1)

                start_step = int(extra.get("step", 0))
                log(f"[RESUME] {ckpt_path} step={start_step}")

            except Exception as e:
                log(f"[ERROR] Не удалось загрузить чекпоинт: {e}")
                sys.exit(1)
        else:
            log("[RESUME] Чекпоинт не найден, начинаем с нуля.")

    ds = TokenFileDataset(args.data, cfg.seq_len, cfg.batch_size, True, args.seed)
    if ds.n_chunks < cfg.batch_size:
        log(f"[ERROR] Датасет слишком мал")
        sys.exit(1)

    dl = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=(DEVICE == "cuda"))
    it = iter(dl)
    step, nan_streak, oom_streak = start_step, 0, 0
    model.train()

    while step < args.steps:
        try:
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl)
                x, y = next(it)
        except StopIteration:
            log("[ERROR] Датасет исчерпан.")
            sys.exit(1)

        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        try:
            _, loss = model_forward(model, x, y, precision)
            loss_value = loss.detach().float().item()
            if not math.isfinite(loss_value):
                raise FloatingPointError("loss not finite")
            loss = loss / cfg.accum

            if scaler:
                scaler.scale(loss).backward()
            elif manual:
                manual.scale_loss(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.accum == 0:
                if scaler:
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                elif manual:
                    finite, gn = manual.unscale_and_check(model)
                    if not finite:
                        raise FloatingPointError("manual inf")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    manual.update(True)
                else:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                nan_streak = 0
                oom_streak = 0

        except FloatingPointError:
            nan_streak += 1
            log(f"[WARN] NaN/Inf streak={nan_streak}")
            if scaler:
                scaler.update()
            if manual:
                manual.update(False)
            opt.zero_grad(set_to_none=True)
            free_memory()
            if nan_streak >= args.max_nan:
                log("[ERROR] Слишком много NaN.")
                return
            continue

        except Exception as e:
            if is_oom(e):
                oom_streak += 1
                log(f"[WARN] OOM streak={oom_streak}")
                opt.zero_grad(set_to_none=True)
                free_memory()
                if oom_streak >= args.max_oom:
                    log("[ERROR] Слишком много OOM.")
                    return
                continue
            raise

        step += 1
        if step % args.log_every == 0:
            log(f"[TRAIN] step={step} loss={loss_value:.4f}")
        if step > start_step and step % args.save_every == 0:
            path = str(Path(args.ckpt_dir) / f"step_{step}.ckpt.json")
            if save_checkpoint(model, opt, scaler, manual, path,
                               {"step": step, "cfg": vars(cfg), "precision": precision,
                                "lr": opt.param_groups[0]["lr"]}):
                log(f"[CKPT] {path}")

    log("[DONE] Обучение завершено.")


# ─────────────────────────────────────────────────────────────
#  Hardware-Aware Оценка
# ─────────────────────────────────────────────────────────────

def estimate_params(cfg: Config) -> int:
    lat = cfg.latent
    attn = cfg.layers * (cfg.dim * lat + 2 * lat * 3 + 3 * lat * lat + lat * cfg.dim + cfg.heads + 2 * lat)
    expert_dim = int(cfg.dim * cfg.expert_mult)
    moe = cfg.layers * (cfg.experts * 2 * cfg.dim * expert_dim + cfg.dim * cfg.router_dim +
                        cfg.router_dim * cfg.experts + cfg.experts)
    emb = cfg.vocab_size * cfg.dim
    scale = cfg.layers * 4 * cfg.dim
    norm = cfg.layers * 2 * cfg.dim + cfg.dim
    return emb + attn + moe + scale + norm


def estimate_tps(cfg: Config, bench_data: Dict, precision_mult: float = 1.0) -> float:
    if not bench_data:
        return 99999.0
    M = cfg.batch_size * cfg.seq_len
    dim, lat, router_dim = cfg.dim, cfg.latent, cfg.router_dim
    ed = int(cfg.dim * cfg.expert_mult)

    def get_time(m, k, n):
        tflops = bench_data.get(f"{m}_{k}_{n}", 5.0) * precision_mult
        if tflops <= 0:
            tflops = 5.0
        return (2.0 * m * k * n) / (tflops * 1e12)

    layers_time = sum(
        get_time(M, dim, lat * 3) + get_time(M, lat, dim) + get_time(M, dim, router_dim) +
        get_time(M, router_dim, cfg.experts) + get_time(M, dim, ed) + get_time(M, ed, dim)
        for _ in range(cfg.layers)
    )
    total_time = (layers_time + get_time(M, dim, cfg.vocab_size)) * 3.0
    if total_time <= 0:
        return 99999.0

    theoretical_tps = M / total_time

    # Калибровочный коэффициент: реальный TPS / теоретический
    # Зависит от размера модели и батча (эмпирически для P102)
    params_m = estimate_params(cfg) / 1e6
    if params_m < 50:
        eff = 0.15  # Очень малые модели: kernel overhead доминирует
    elif params_m < 150:
        eff = 0.25  # Малые модели
    elif params_m < 500:
        eff = 0.35  # Средние модели
    elif params_m < 1000:
        eff = 0.45  # Крупные модели
    else:
        eff = 0.55  # Очень крупные: лучше утилизация

    # Коррекция на размер батча (малые батчи менее эффективны)
    if M < 1024:
        eff *= 0.5
    elif M < 4096:
        eff *= 0.7
    elif M < 16384:
        eff *= 0.85

    # MoE overhead: больше экспертов = больше scatter/gather
    moe_factor = max(0.6, 1.0 - cfg.experts * 0.02)

    return theoretical_tps * eff * moe_factor


def estimate_memory_gb(cfg: Config, use_8bit: bool = False, use_bf16: bool = False,
                       use_lion: bool = False, use_paged: bool = False, use_checkpoint: bool = False) -> float:
    params = estimate_params(cfg)

    if use_8bit and use_bf16 and use_lion:
        bytes_per_param = 5
    elif use_8bit and use_bf16:
        bytes_per_param = 6
    elif use_8bit and use_lion:
        bytes_per_param = 9
    elif use_8bit:
        bytes_per_param = 10
    elif use_bf16 and use_lion:
        bytes_per_param = 8
    elif use_bf16:
        bytes_per_param = 12
    else:
        bytes_per_param = 16

    model_mem = params * bytes_per_param / 1e9
    act_bytes = 2 if (use_bf16 or use_8bit) else 4
    B, T, D = cfg.batch_size, cfg.seq_len, cfg.dim
    L, E = cfg.layers, cfg.experts
    ed = int(D * cfg.expert_mult)

    # Активации внимания: Q,K,V + output + промежуточные (×5 буферов на слой)
    attn_act = B * T * D * L * 5 * act_bytes / 1e9

    # MoE: router + все эксперты (top-1, но буферы аллоцируются для всех)
    moe_act = B * T * (D + E * ed) * L * act_bytes / 1e9

    # Градиенты (равны весам по размеру)
    grad_mem = params * act_bytes / 1e9

    cuda_base_overhead = 2.0 if params < 500e6 else 3.0
    ckpt_factor = 0.5 if use_checkpoint else 1.0
    paged_factor = 0.75 if use_paged else 1.0

    total = model_mem + grad_mem + (attn_act + moe_act) * ckpt_factor * 1.3 * paged_factor + cuda_base_overhead
    # Фиксированный запас на фрагментацию и временные буферы cuBLAS
    # Не зависит от размера модели (эмпирически ~1.5 GB для P102)
    fragmentation_reserve = 1.5
    return total + fragmentation_reserve


def log_compact(cfg: Config, params_m: float, est_tps: float, est_mem_gb: float,
                trial_num: int, status: str = ""):
    ed = int(cfg.dim * cfg.expert_mult)
    line = (f"#{trial_num:03d} {cfg.dim}/{cfg.layers}/{cfg.experts} "
            f"h={ed} s={cfg.seq_len} b={cfg.batch_size} "
            f"{params_m:.0f}M {est_mem_gb:.1f}GB | {status}")
    log(line)


# ─────────────────────────────────────────────────────────────
#  Предвычисление допустимых конфигураций
# ─────────────────────────────────────────────────────────────

def precompute_valid_configs(args) -> List[Config]:
    configs = []
    total_checked = 0

    for dim in [768, 1024, 1280, 1536, 2048, 2560]:
        if dim < args.min_dim or dim > args.max_dim:
            continue
        lat = dim // 4
        for heads in [4, 8, 16, 32]:
            if lat % heads != 0:
                continue
            for experts in [8, 16, 32]:
                for em100 in range(100, 201, 5):
                    expert_mult = em100 / 100.0
                    ed = int(dim * expert_mult)
                    for router_dim in [192, 256]:
                        for seq_len in [256, 512, 1024]:
                            for batch_size in [64, 32, 16, 8, 4]:
                                if batch_size > args.pareto_max_batch:
                                    continue
                                for checkpoint in [True, False]:
                                    total_checked += 1

                                    attn_per_layer = dim * lat + 3 * lat * lat + lat * dim + 6 * dim
                                    moe_per_layer = experts * 2 * dim * ed + dim * router_dim + router_dim * experts + experts
                                    params_per_layer = attn_per_layer + moe_per_layer

                                    vocab_params = args.vocab_size * dim
                                    overhead = 4 * dim

                                    min_layers = math.ceil((args.min_params - vocab_params - overhead) / params_per_layer)
                                    max_layers = math.floor((args.max_params - vocab_params - overhead) / params_per_layer)
                                    min_layers = max(min_layers, 4)  # Минимум 4 слоя для осмысленной глубины
                                    max_layers = min(max_layers, 128)

                                    if min_layers > max_layers:
                                        continue

                                    for layers in range(min_layers, max_layers + 1):
                                        cfg = Config(
                                            vocab_size=args.vocab_size, dim=dim, layers=layers,
                                            heads=heads, experts=experts, expert_mult=expert_mult,
                                            router_dim=router_dim, seq_len=seq_len, batch_size=batch_size,
                                            accum=1, cca_comp=4, checkpoint=checkpoint
                                        )

                                        est_mem = estimate_memory_gb(cfg, use_8bit=args.use_8bit, use_bf16=args.bf16, use_lion=args.lion, use_paged=args.paged, use_checkpoint=cfg.checkpoint)
                                        if est_mem <= args.max_memory:
                                            configs.append(cfg)

    log(f"[PRECOMPUTE] Проверено {total_checked} комбинаций, допустимых: {len(configs)}")
    return configs


# ─────────────────────────────────────────────────────────────
#  Парето-поиск
# ─────────────────────────────────────────────────────────────

def propose_cfg_optuna(trial, args, valid_configs):
    if not valid_configs:
        raise optuna.TrialPruned()
    idx = trial.suggest_int("config_idx", 0, len(valid_configs) - 1)
    return valid_configs[idx], idx


def propose_cfg_random(args, rng: random.Random, bench_data: Dict) -> Config:
    for _ in range(1000):
        dims = [d for d in [192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048]
                if args.min_dim <= d <= args.max_dim]
        if not dims:
            dims = [args.min_dim]
        dim = rng.choice(dims)
        lat = dim // 4
        valid_heads = [h for h in [4, 6, 8, 10, 12, 16, 20, 24, 32] if lat % h == 0]
        if not valid_heads:
            continue
        heads = rng.choice(valid_heads)
        layers = rng.randint(args.min_layers, args.max_layers)
        experts = rng.choice([4, 8, 16])
        expert_mult = rng.choice([1.0, 1.15, 1.30, 1.45])
        router_dim = rng.choice([96, 128, 192, 256])
        seq_len = rng.choice([256, 512, 1024])
        batch_size = rng.choice([b for b in [64, 32, 16, 8, 4, 2]
                                 if b <= max(1, min(args.pareto_max_batch, 64))] or [1])
        checkpoint = rng.choice([True, False])
        cfg = Config(
            vocab_size=args.vocab_size, dim=dim, layers=layers, heads=heads,
            experts=experts, expert_mult=expert_mult, router_dim=router_dim,
            seq_len=seq_len, batch_size=batch_size, accum=1, cca_comp=4, checkpoint=checkpoint
        )
        p = estimate_params(cfg)
        if not (args.min_params <= p <= args.max_params):
            continue
        if estimate_tps(cfg, bench_data, 1.0) < args.min_tps:
            continue
        return cfg
    raise RuntimeError("Не удалось предложить допустимую конфигурацию.")


def benchmark_cfg(cfg: Config, args, precision: str, initial_batch_size: int = None, no_retry: bool = False) -> Dict:
    torch.manual_seed(args.seed)
    current_batch = initial_batch_size or cfg.batch_size
    max_retries = 4

    for attempt in range(max_retries):
        cfg_copy = Config(**{**vars(cfg), 'batch_size': current_batch})
        model = None
        opt = None
        scaler = None
        ds = None
        dl = None

        try:
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()

            model = NanoZaya(cfg_copy)

            if DEVICE == "cuda" and precision == "fp16":
                model = model.half().to(DEVICE)
            elif DEVICE == "cuda" and precision == "bf16":
                model = model.bfloat16().to(DEVICE)
            else:
                model = model.to(DEVICE)

            init_weights(model)
            n_params = sum(p.numel() for p in model.parameters())

            if n_params > args.max_params:
                raise RuntimeError("too many params")

            model.train()
            opt = make_optimizer(model, args)
            scaler = make_grad_scaler() if DEVICE == "cuda" and precision in ("mixed", "bf16") else None

            ds = TokenFileDataset(args.data, cfg_copy.seq_len, cfg_copy.batch_size, True, args.seed)
            if ds.n_chunks < cfg_copy.batch_size:
                raise RuntimeError("dataset too small")

            dl = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=(DEVICE == "cuda"))
            it = iter(dl)

            def get_batch():
                nonlocal it
                try:
                    return next(it)
                except StopIteration:
                    it = iter(dl)
                    return next(it)

            def one_step():
                x, y = get_batch()
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                _, loss = model_forward(model, x, y, precision)

                if not math.isfinite(loss.detach().float().item()):
                    raise FloatingPointError("loss not finite")

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                return loss.detach().float().item(), x.numel()

            for _ in range(max(1, args.pareto_warmup)):
                one_step()

            if DEVICE == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            loss_sum, tokens = 0.0, 0

            for _ in range(max(1, args.pareto_steps)):
                lv, tk = one_step()
                loss_sum += lv
                tokens += tk

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            dt = max(1e-8, time.time() - t0)
            peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0

            return {
                "loss": loss_sum / max(1, args.pareto_steps),
                "tokens_per_s": tokens / dt,
                "params": n_params,
                "peak_gb": peak_gb,
                "cfg": vars(cfg_copy),
                "batch_size_used": current_batch,
                "attempt": attempt
            }

        except Exception as e:
            if model is not None:
                del model
            if opt is not None:
                del opt
            if scaler is not None:
                del scaler
            if ds is not None:
                del ds
            if dl is not None:
                del dl
            free_memory()

            if is_oom(e) and attempt < max_retries - 1 and not no_retry:
                old_batch = current_batch
                current_batch = max(1, current_batch // 2)
                log(f"{'':>20} ⚠️ OOM bs={old_batch}→{current_batch}")
                continue
            else:
                raise

    raise RuntimeError("Все попытки снижения батча провалились")


def compute_pareto_front(results: List[Dict]) -> List[Dict]:
    valid = [r for r in results
             if math.isfinite(r.get("loss", float("inf"))) and math.isfinite(r.get("tokens_per_s", -float("inf")))]
    front = []
    for p in valid:
        dominated = False
        for q in valid:
            if q is p:
                continue
            if (q["loss"] <= p["loss"] and q["tokens_per_s"] >= p["tokens_per_s"] and
                    (q["loss"] < p["loss"] or q["tokens_per_s"] > p["tokens_per_s"])):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda z: z["loss"])


def _save_intermediate_results(study, out_path: str):
    results = []
    for t in study.trials:
        if str(t.state) == "TrialState.COMPLETE" and t.values is not None:
            if all(c <= 0 for c in t.user_attrs.get("constraints", [1.0])):
                item = {"trial": t.number, "loss": t.values[0], "tokens_per_s": t.values[1]}
                item.update(t.user_attrs)
                results.append(item)
    front = compute_pareto_front(results)
    create_parent(out_path)
    Path(out_path).write_text(json.dumps({"all": results, "pareto": front}, ensure_ascii=False, indent=2),
                              encoding="utf-8")


def run_pareto(args):
    if args.cmp50:
        args.pareto_trials = max(args.pareto_trials, 50)

    if not Path(args.data).exists():
        make_dummy_token_file(args.data, 2_000_000, args.vocab_size, args.seed)

    precision = "fp32" if args.fp32 else "mixed"
    precision_mult = 1.0

    bench_data = {}
    bench_path = args.bench_json
    
    # Авто-поиск, если не указан
    if not bench_path:
        candidates = ["zaya_bench_results.json", "zaya_bench.json", "zaya_bench_fp32.json"]
        for c in candidates:
            if Path(c).exists():
                bench_path = c
                log(f"[INFO] Автоматически найден профиль железа: {c}")
                break
    
    if bench_path and Path(bench_path).exists():
        with open(bench_path, "r", encoding="utf-8") as bf:
            bench_data = json.load(bf).get("shapes", {})
        log(f"[INFO] Профиль железа: {len(bench_data)} форм из {bench_path}")
    else:
        log("[WARN] Профиль железа не найден. TPS будет равен 99999 (заглушка).")
        log("[HINT] Передайте --bench-json zaya_bench_results.json или поместите файл в текущую директорию.")

    results = []

    valid_configs = precompute_valid_configs(args)
    if not valid_configs:
        log("[ERROR] Нет допустимых конфигураций. Увеличьте --max-memory или уменьшите --min-params.")
        return []

    # Режим top-by-tps: сортируем по TPS и берём только N лучших
    if args.top_by_tps > 0:
        scored = []
        for cfg in valid_configs:
            tps = estimate_tps(cfg, bench_data, precision_mult)
            scored.append((tps, cfg))
        scored.sort(key=lambda x: x[0], reverse=True)
        valid_configs = [cfg for _, cfg in scored[:args.top_by_tps]]
        log(f"[TOP-TPS] Отобрано {len(valid_configs)} самых быстрых из {len(scored)} конфигураций")

    # Вывод примеров допустимых конфигураций
    if len(valid_configs) > 0:
        log("\n[EXAMPLES] Примеры допустимых конфигураций:")
        
        # Сортируем по разным критериям
        configs_with_metrics = []
        for cfg in valid_configs[:min(100, len(valid_configs))]:  # Берём первые 100 для скорости
            params_m = estimate_params(cfg) / 1e6
            est_tps = estimate_tps(cfg, bench_data, precision_mult)
            est_mem = estimate_memory_gb(cfg, use_8bit=args.use_8bit, use_bf16=args.bf16, use_lion=args.lion, use_paged=args.paged, use_checkpoint=cfg.checkpoint)
            configs_with_metrics.append((cfg, params_m, est_tps, est_mem))
        
        # Топ-3 самых крупных
        log("  📦 Крупнейшие модели:")
        for cfg, params_m, est_tps, est_mem in sorted(configs_with_metrics, key=lambda x: x[1], reverse=True)[:3]:
            m = cfg.batch_size * cfg.seq_len
            tps_str = f"{est_tps:>7.0f}tps" if est_tps < 90000 else "  (н/д) "
            log(f"     dim={cfg.dim:<4} lay={cfg.layers:<2} exp={cfg.experts:<2} b={cfg.batch_size:<2} seq={cfg.seq_len:<4} | "
                f"{params_m:>6.0f}M | {tps_str} | {est_mem:>5.1f}GB")
        
        # Топ-3 самых быстрых (только если есть bench_data)
        if bench_data:
            log("  ⚡ Быстрейшие модели (по данным бенчмарка):")
            for cfg, params_m, est_tps, est_mem in sorted(configs_with_metrics, key=lambda x: x[2], reverse=True)[:3]:
                m = cfg.batch_size * cfg.seq_len
                log(f"     dim={cfg.dim:<4} lay={cfg.layers:<2} exp={cfg.experts:<2} b={cfg.batch_size:<2} M={m:<5} | "
                    f"{params_m:>6.0f}M | {est_tps:>7.0f}tps | {est_mem:>5.1f}GB")
        else:
            log("  ⚡ Быстрейшие модели:")
            log("     (недоступно — передайте --bench-json для расчёта TPS)")
        
        # Топ-3 самых компактных по памяти
        log("  💾 Компактнейшие по памяти:")
        for cfg, params_m, est_tps, est_mem in sorted(configs_with_metrics, key=lambda x: x[3])[:3]:
            tps_str = f"{est_tps:>7.0f}tps" if est_tps < 90000 else "  (н/д) "
            log(f"     dim={cfg.dim:<4} lay={cfg.layers:<2} exp={cfg.experts:<2} b={cfg.batch_size:<2} | "
                f"{params_m:>6.0f}M | {tps_str} | {est_mem:>5.1f}GB")
        
        # Топ-3 с максимальным batch (наиболее эффективные для GPU)
        log("  🚀 Максимальный батч (лучшая утилизация GPU):")
        for cfg, params_m, est_tps, est_mem in sorted(configs_with_metrics, key=lambda x: x[0].batch_size, reverse=True)[:3]:
            tps_str = f"{est_tps:>7.0f}tps" if est_tps < 90000 else "  (н/д) "
            log(f"     dim={cfg.dim:<4} lay={cfg.layers:<2} exp={cfg.experts:<2} b={cfg.batch_size:<2} | "
                f"{params_m:>6.0f}M | {tps_str} | {est_mem:>5.1f}GB")
        
        log("")

    if not OPTUNA_AVAILABLE:
        log("[INFO] Optuna не найден, использую случайный Парето-поиск.")
        rng = random.Random(args.seed)

        for i in range(min(args.pareto_trials, len(valid_configs))):
            cfg = valid_configs[i]
            params = estimate_params(cfg)
            params_m = params / 1e6
            est_tps = estimate_tps(cfg, bench_data, 1.0)
            est_mem = estimate_memory_gb(cfg, use_8bit=args.use_8bit, use_bf16=args.bf16, use_lion=args.lion, use_paged=args.paged, use_checkpoint=cfg.checkpoint)

            log_compact(cfg, params_m, est_tps, est_mem, i, "🚀")

            try:
                res = benchmark_cfg(cfg, args, precision, initial_batch_size=cfg.batch_size,
                                no_retry=(args.top_by_tps > 0))
                res["trial"] = i
                res["est_tps"] = est_tps
                res["est_mem"] = est_mem
                results.append(res)
                log(f"{'':>20} ✅ {res['loss']:.4f} {res['tokens_per_s']:.0f}tps")
            except Exception as e:
                if is_oom(e) or isinstance(e, FloatingPointError) or "dataset too small" in str(e):
                    free_memory()
                    continue
                raise

        front = compute_pareto_front(results)
        create_parent(args.pareto_out)
        Path(args.pareto_out).write_text(json.dumps({"all": results, "pareto": front}, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
        log(f"\n[PARETO] Результатов: {len(results)} | Фронт: {len(front)} | {args.pareto_out}")
        return front

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def constraints_func(trial):
        return trial.user_attrs.get("constraints", [0.0, 0.0, 0.0, 0.0])

    sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints_func, seed=args.seed)

    db_path = Path(args.pareto_out).with_suffix(".db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=sampler,
        study_name="zaya_pareto",
        storage=storage,
        load_if_exists=True,
    )

    existing = len([t for t in study.trials if str(t.state) == "TrialState.COMPLETE"])
    if existing > 0:
        log(f"[INFO] Найдено {existing} трейлов в {db_path}, добавлю ещё {args.pareto_trials}")

    if args.debug:
        optuna.logging.set_verbosity(optuna.logging.INFO)
        log(f"[DEBUG] Optuna verbosity: INFO")


    def objective(trial):
        try:
            cfg, cfg_idx = propose_cfg_optuna(trial, args, valid_configs)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            if args.debug:
                log(f"#{trial.number:03d} ⛔ {type(e).__name__}: {e}")
            raise optuna.TrialPruned()

      #Удалите её — config_idx уже устанавливается в начале функции.
        # Устанавливаем config_idx СРАЗУ после генерации
       # if hasattr(cfg, '_config_idx'):
        #    trial.set_user_attr("config_idx", cfg._config_idx)
       # else:
        #    trial.set_user_attr("config_idx", -1)

        # Проверка дублей
        cfg_idx = trial.user_attrs.get("config_idx", -1)
        if cfg_idx >= 0:
            for t in study.trials:
                if t.number != trial.number and t.user_attrs.get("config_idx") == cfg_idx:
                    if args.debug:
                        log(f"#{trial.number:03d} ⛔ дубль конфигурации {cfg_idx}")
                    raise optuna.TrialPruned()



        # Проверка дублей
        if hasattr(cfg, '_config_idx'):
            for t in study.trials:
                if t.number != trial.number and t.user_attrs.get("config_idx") == cfg._config_idx:
                    if args.debug:
                        log(f"#{trial.number:03d} ⛔ дубль конфигурации {cfg._config_idx}")
                    raise optuna.TrialPruned()

        params = estimate_params(cfg)
        params_m = params / 1e6
        est_tps = estimate_tps(cfg, bench_data, precision_mult)
        est_mem = estimate_memory_gb(cfg, use_8bit=args.use_8bit, use_bf16=args.bf16, use_lion=args.lion, use_paged=args.paged, use_checkpoint=cfg.checkpoint)

        log_compact(cfg, params_m, est_tps, est_mem, trial.number, "🚀")


        try:
            res = benchmark_cfg(cfg, args, precision, initial_batch_size=cfg.batch_size,
                                no_retry=(args.top_by_tps > 0))
        except optuna.TrialPruned:
            raise
        except Exception as e:
            if is_oom(e):
                log_compact(cfg, params_m, est_tps, est_mem, trial.number, "⛔ OOM")
                trial.set_user_attr("constraints", [0.0, 0.0, 0.0, 100.0])
                free_memory()
                return 100.0, 0.0
            elif isinstance(e, FloatingPointError):
                log_compact(cfg, params_m, est_tps, est_mem, trial.number, "⛔ NaN/Inf")
                raise optuna.TrialPruned()
            elif "dataset too small" in str(e):
                log_compact(cfg, params_m, est_tps, est_mem, trial.number, "⛔ data")
                raise optuna.TrialPruned()
            else:
                log_compact(cfg, params_m, est_tps, est_mem, trial.number, f"⛔ {type(e).__name__}")
                if args.debug:
                    log(f"{'':>20} {e}")
                raise optuna.TrialPruned()



        batch_penalty = 0.0
        actual_batch = res.get("batch_size_used", cfg.batch_size)
        if actual_batch < cfg.batch_size:
            batch_ratio = actual_batch / cfg.batch_size
            batch_penalty = (1.0 - batch_ratio) * 2.0

        trial.set_user_attr("constraints", [0.0, 0.0, 0.0, 0.0])
        trial.set_user_attr("params", res["params"])
        trial.set_user_attr("tokens_per_s", res["tokens_per_s"])
        trial.set_user_attr("peak_gb", res["peak_gb"])
        trial.set_user_attr("cfg", res["cfg"])
        trial.set_user_attr("real_loss", res["loss"])
        trial.set_user_attr("actual_batch", actual_batch)

        adjusted_loss = res["loss"] + batch_penalty

        pen_str = f" pen+{batch_penalty:.2f}" if batch_penalty > 0 else ""
        log(f"{'':>20} ✅ {res['loss']:.4f} {res['tokens_per_s']:.0f}tps "
            f"b={actual_batch}/{cfg.batch_size} {res['peak_gb']:.1f}GB{pen_str}")

        _save_intermediate_results(study, args.pareto_out)

        return adjusted_loss, res["tokens_per_s"]

    study.optimize(objective, n_trials=args.pareto_trials, catch=(Exception,))

    for t in study.trials:
        if str(t.state) == "TrialState.COMPLETE" and t.values is not None:
            if all(c <= 0 for c in t.user_attrs.get("constraints", [1.0])):
                item = {"trial": t.number, "loss": t.values[0], "tokens_per_s": t.values[1]}
                item.update(t.user_attrs)
                results.append(item)

    front = compute_pareto_front(results)
    create_parent(args.pareto_out)
    Path(args.pareto_out).write_text(json.dumps({"all": results, "pareto": front}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    log(f"\n[PARETO] Результатов: {len(results)} | Фронт: {len(front)} | {args.pareto_out}")

    for r in front[:20]:
        cfg = r.get("cfg", {})
        log(f"[FRONT] loss={r['loss']:.4f} tps={r['tokens_per_s']:.1f} "
            f"params={r.get('params', 0) / 1e6:.2f}M peak={r.get('peak_gb', 0.0):.2f}GB "
            f"dim={cfg.get('dim')} layers={cfg.get('layers')} heads={cfg.get('heads')} "
            f"experts={cfg.get('experts')} seq={cfg.get('seq_len')} batch={cfg.get('batch_size')}")

    return front


# ─────────────────────────────────────────────────────────────
#  Уточнение фронтирных моделей
# ─────────────────────────────────────────────────────────────

def refine_front(args):
    if not Path(args.pareto_out).exists():
        log(f"[ERROR] Файл {args.pareto_out} не найден. Сначала запустите --mode pareto.")
        sys.exit(1)

    data = json.loads(Path(args.pareto_out).read_text(encoding="utf-8"))
    front = data.get("pareto", [])

    if not front:
        log("[ERROR] Парето-фронт пуст.")
        sys.exit(1)

    precision = "fp32" if args.fp32 else "mixed"
    refine_steps = args.refine_steps
    log(f"[REFINE] Уточняю {len(front)} моделей, шагов: {refine_steps}")

    bench_data = {}
    if args.bench_json and Path(args.bench_json).exists():
        with open(args.bench_json, "r", encoding="utf-8") as f:
            bench_data = json.load(f).get("shapes", {})

    results = []
    for i, r in enumerate(front):
        cfg_dict = r.get("cfg", {})
        cfg = Config(
            vocab_size=cfg_dict.get("vocab_size", args.vocab_size),
            dim=cfg_dict.get("dim", 256),
            layers=cfg_dict.get("layers", 24),
            heads=cfg_dict.get("heads", 8),
            experts=cfg_dict.get("experts", 8),
            expert_mult=cfg_dict.get("expert_mult", 1.3),
            router_dim=cfg_dict.get("router_dim", 192),
            seq_len=cfg_dict.get("seq_len", 256),
            batch_size=cfg_dict.get("batch_size", 1),
            accum=1,
            cca_comp=cfg_dict.get("cca_comp", 4),
            checkpoint=cfg_dict.get("checkpoint", False),
        )

        params_m = estimate_params(cfg) / 1e6
        log(f"[{i+1}/{len(front)}] {cfg.dim}/{cfg.layers}/{cfg.experts} "
            f"s={cfg.seq_len} b={cfg.batch_size} {params_m:.0f}M | 🚀")

        try:
            old_steps = args.pareto_steps
            args.pareto_steps = refine_steps

            res = benchmark_cfg(cfg, args, precision, initial_batch_size=cfg.batch_size,
                                no_retry=(args.top_by_tps > 0))

            args.pareto_steps = old_steps

            res["trial"] = r.get("trial", i)
            results.append(res)

            log(f"{'':>20} ✅ {res['loss']:.4f} {res['tokens_per_s']:.0f}tps "
                f"b={res['batch_size_used']}/{cfg.batch_size} {res['peak_gb']:.1f}GB")

        except Exception as e:
            if is_oom(e) or isinstance(e, FloatingPointError) or "dataset too small" in str(e):
                log(f"{'':>20} ⚠️ пропуск: {e}")
                free_memory()
                continue
            raise

    refined_front = compute_pareto_front(results)

    out_path = Path(args.pareto_out).with_suffix(".refined.json")
    payload = {"all": results, "pareto": refined_front, "refine_steps": refine_steps}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"\n[REFINE] Уточнено: {len(results)} | Фронт: {len(refined_front)} | {out_path}")
    for r in refined_front:
        cfg = r.get("cfg", {})
        log(f"[FRONT] loss={r['loss']:.4f} tps={r['tokens_per_s']:.1f} "
            f"params={r.get('params', 0)/1e6:.2f}M "
            f"dim={cfg.get('dim')} layers={cfg.get('layers')} heads={cfg.get('heads')} "
            f"experts={cfg.get('experts')} seq={cfg.get('seq_len')} batch={cfg.get('batch_size')}")

    return refined_front


# ─────────────────────────────────────────────────────────────
#  Авто-обучение после Парето
# ─────────────────────────────────────────────────────────────

def select_and_train(args, front: List[Dict]):
    if not front:
        log("[ERROR] Фронт пуст, нечего обучать.")
        return

    log("\n" + "=" * 70)
    log("  ЛУЧШИЕ МОДЕЛИ ДЛЯ ОБУЧЕНИЯ")
    log("=" * 70)

    for i, r in enumerate(front[:10]):
        cfg = r.get("cfg", {})
        log(f"  [{i}] loss={r['loss']:.4f} tps={r['tokens_per_s']:.0f} "
            f"params={r.get('params', 0) / 1e6:.1f}M "
            f"dim={cfg.get('dim')} layers={cfg.get('layers')} "
            f"experts={cfg.get('experts')} seq={cfg.get('seq_len')} batch={cfg.get('batch_size')}")

    log("=" * 70)
    log(f"\n[AUTO] Через {args.select_timeout} сек будет выбрана модель [0].")
    log(f"[AUTO] Введите номер [0-{min(9, len(front) - 1)}] или Ctrl+C:")

    selected = 0
    try:
        import select as sel
        import sys

        start_wait = time.time()
        while time.time() - start_wait < args.select_timeout:
            if sel.select([sys.stdin], [], [], 1.0)[0]:
                user_input = sys.stdin.readline().strip()
                if user_input.isdigit():
                    idx = int(user_input)
                    if 0 <= idx < len(front):
                        selected = idx
                        log(f"[AUTO] Выбрана модель [{selected}]")
                    break
                elif user_input.lower() in ('q', 'quit', 'exit'):
                    log("[AUTO] Отменено.")
                    return

        remaining = max(0, args.select_timeout - (time.time() - start_wait))
        if remaining > 0:
            time.sleep(remaining)

    except KeyboardInterrupt:
        log("\n[AUTO] Отменено (Ctrl+C).")
        return
    except Exception:
        log(f"[AUTO] Жду {args.select_timeout} сек...")
        try:
            time.sleep(args.select_timeout)
        except KeyboardInterrupt:
            log("\n[AUTO] Отменено (Ctrl+C).")
            return

    best = front[selected]
    cfg_dict = best.get("cfg", {})

    log(f"\n{'=' * 70}")
    log(f"  ВЫБРАНА МОДЕЛЬ [{selected}]")
    log(f"  Loss={best['loss']:.4f} TPS={best['tokens_per_s']:.0f} "
        f"Params={best.get('params', 0) / 1e6:.1f}M")
    log(f"{'=' * 70}\n")

    args.dim = cfg_dict.get("dim", 256)
    args.layers = cfg_dict.get("layers", 24)
    args.heads = cfg_dict.get("heads", 8)
    args.experts = cfg_dict.get("experts", 8)
    args.expert_mult = cfg_dict.get("expert_mult", 1.3)
    args.router_dim = cfg_dict.get("router_dim", 192)
    args.seq = cfg_dict.get("seq_len", 256)
    args.batch = cfg_dict.get("batch_size", 1)
    args.accum = cfg_dict.get("accum", 8)
    args.cca_comp = cfg_dict.get("cca_comp", 4)
    args.checkpoint = cfg_dict.get("checkpoint", False)
    args.steps = args.train_steps
    args.ckpt_dir = args.train_ckpt_dir

    log(f"[TRAIN] Начинаю обучение на {args.train_steps} шагов...")
    log(f"[TRAIN] Чекпоинты: {args.train_ckpt_dir}")

    train(args)
    log("[DONE] Авто-обучение завершено.")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

HELP_EPILOG = """
Примеры:

1. Парето-поиск с Lion8bit + BF16 + Paged:
   python zaya_train_pareto.py --mode pareto \\
     --min-params 1500e6 --max-params 1888e6 \\
     --max-memory 9.9 \\
     --use-8bit --bf16 --lion --paged \\
     --pareto-trials 400

2. Уточнение фронтирных:
   python zaya_train_pareto.py --mode refine \\
     --refine-steps 500 --pareto-out pareto.json

3. Уточнение + авто-обучение:
   python zaya_train_pareto.py --mode refine \\
     --refine-steps 500 --auto-train --train-steps 50000
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Надёжный скрипт обучения и Парето-поиска для ZAYA.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", required=True,
                        choices=["train", "build-vocab", "tokenize", "pareto", "refine"])
    parser.add_argument("--data", default="corpus.tok16")
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--vocab", default="vocab16k.txt")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--out", default="corpus.tok16")
    parser.add_argument("--ckpt-dir", default="ckpt_train")
    parser.add_argument("--resume", default="")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--expert-mult", type=float, default=1.3)
    parser.add_argument("--router-dim", type=int, default=192)
    parser.add_argument("--cca-comp", type=int, default=4)
    parser.add_argument("--checkpoint", action="store_true")

    p_group = parser.add_mutually_exclusive_group()
    p_group.add_argument("--mixed", action="store_true")
    p_group.add_argument("--fp32", action="store_true")
    p_group.add_argument("--strict-fp16", action="store_true")
    p_group.add_argument("--bf16", action="store_true",
                         help="BFloat16 (эмуляция на Pascal, экономия памяти)")

    parser.add_argument("--disable-scaler", action="store_true")
    parser.add_argument("--use-8bit", action="store_true",
                        help="8-bit оптимизатор через bitsandbytes")
    parser.add_argument("--lion", action="store_true",
                        help="Использовать Lion вместо AdamW (экономия ~25%% памяти)")
    parser.add_argument("--paged", action="store_true",
                        help="Использовать Paged оптимизатор (выгрузка на CPU при OOM)")
    parser.add_argument("--max-nan", type=int, default=8)
    parser.add_argument("--max-oom", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bench-json", type=str, default="")
    parser.add_argument("--pareto-trials", type=int, default=50)
    parser.add_argument("--pareto-steps", type=int, default=20)
    parser.add_argument("--pareto-warmup", type=int, default=3)
    parser.add_argument("--refine-steps", type=int, default=200)
    parser.add_argument("--cmp50", action="store_true")
    parser.add_argument("--max-params", type=float, default=380e6)
    parser.add_argument("--min-params", type=float, default=0.0)
    parser.add_argument("--min-tps", type=float, default=0.0)
    parser.add_argument("--max-memory", type=float, default=24.0)
    parser.add_argument("--min-dim", type=int, default=192)
    parser.add_argument("--max-dim", type=int, default=2048)
    parser.add_argument("--min-layers", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=32)
    parser.add_argument("--pareto-out", default="pareto_front.json")
    parser.add_argument("--pareto-max-batch", type=int, default=64)

    parser.add_argument("--auto-train", action="store_true")
    parser.add_argument("--train-steps", type=int, default=50000)
    parser.add_argument("--train-ckpt-dir", default="ckpt_auto")
    parser.add_argument("--select-timeout", type=int, default=30)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--top-by-tps", type=int, default=0,
                        help="Проверить только N самых быстрых конфигураций (0=обычный Парето)")

    return parser.parse_args()


def main():
    args = parse_args()
    harden_backends()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if not args.log_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.log_file = f"training_log_{timestamp}.txt"
    init_log(args.log_file)
    log(f"[INFO] Лог: {args.log_file}")

    try:
        if args.mode == "train":
            train(args)

        elif args.mode == "pareto":
            front = run_pareto(args)
            if args.auto_train and front:
                log("\n[AUTO] Парето завершён, перехожу к обучению...")
                select_and_train(args, front)
            elif front:
                log("\n[INFO] Для уточнения: --mode refine --refine-steps 200")

        elif args.mode == "refine":
            front = refine_front(args)
            if args.auto_train and front:
                log("\n[AUTO] Уточнение завершено, перехожу к обучению...")
                select_and_train(args, front)

        elif args.mode == "build-vocab":
            vocab = build_vocab(args.inputs, args.vocab_size)
            save_vocab(vocab, args.vocab)
            log(f"[VOCAB] Сохранено: {args.vocab}")

        elif args.mode == "tokenize":
            if not Path(args.vocab).exists():
                log(f"[ERROR] Словарь не найден")
                sys.exit(1)
            tokenize_files(args.inputs, Tokenizer(load_vocab(args.vocab)), args.out)
            log(f"[TOKENIZE] Готово: {args.out}")

        else:
            sys.exit(1)

    finally:
        close_log()


if __name__ == "__main__":
    main()
