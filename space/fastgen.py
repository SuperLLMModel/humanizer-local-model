"""Fast decoding for the humanizer 12B on ZeroGPU, staying in PyTorch / transformers.

The idea: prefill runs exactly as before (eager transformers, DynamicCache), then its K/V are copied into
fixed-size buffers and every following token goes through one fixed-shape "decode step" module
(input_ids [1, 1], pos [1]) -> float32 logits [1, vocab]. Because every shape is static, that step can be
torch.compile'd (with CUDA graphs) or exported once with torch.export and compiled ahead of time with
AOTInductor (spaces.aoti_*), so a fresh ZeroGPU worker only has to load a .so instead of recompiling.

Sampling copies what transformers' generate does for this model's generation config:
float32 logits -> suppress_tokens (-inf) -> [temperature 1.0 skipped] -> [top_k 0 skipped] -> top_p ->
softmax -> multinomial. Greedy = argmax after suppress_tokens.
"""
from __future__ import annotations

import os
import threading
import time

import torch
from torch import nn

MAX_LEN = int(os.environ.get("FASTGEN_MAX_LEN", "4096"))   # prompt (<= ~1,500 tok) + max_new (<= 2,048)


class _KV:
    """Duck-typed transformers Cache for the decode step. Full-attention layers write the new K/V at `pos` of a
    max_len buffer; sliding layers write at `pos % window` of a ring buffer that holds exactly the last `window`
    positions. Buffers live on the owning module (read through it so torch.export sees them as buffers)."""

    is_compileable = True

    def __init__(self, owner: "DecodeStep", pos: torch.Tensor):
        self.owner, self.pos = owner, pos
        self.slot = pos % owner.window

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        k = getattr(self.owner, f"k{layer_idx}")
        v = getattr(self.owner, f"v{layer_idx}")
        at = self.slot if self.owner.sliding[layer_idx] else self.pos
        k.index_copy_(2, at, key_states)
        v.index_copy_(2, at, value_states)
        return k, v


def decode_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Attention for one query token without repeat_kv: the query heads of a KV group share one matmul against
    that group's keys. Scores and softmax in float32 (as SDPA does internally). -> ([1, 1, H, D], None)."""
    b, h, q, d = query.shape
    kvh = key.shape[1]
    g = h // kvh
    qg = query.reshape(b, kvh, g * q, d).float()
    scores = torch.matmul(qg, key.float().transpose(-1, -2))
    if scaling is not None and scaling != 1.0:
        scores = scores * scaling
    scores = scores.masked_fill(~attention_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    out = torch.matmul(probs, value)                       # [b, kvh, g*q, d]
    out = out.reshape(b, h, q, d).transpose(1, 2).contiguous()
    return out, None


class _decode_attention_only:
    """Inside the decode step only, the model's "sdpa" attention is decode_attention (prefill keeps SDPA)."""

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.m = ALL_ATTENTION_FUNCTIONS
        self.prev = self.m._local_mapping.get("sdpa")
        self.m._local_mapping["sdpa"] = decode_attention

    def __exit__(self, *exc):
        if self.prev is None:
            self.m._local_mapping.pop("sdpa", None)
        else:
            self.m._local_mapping["sdpa"] = self.prev


def kv_shapes(text_config, max_len: int = MAX_LEN):
    out = []
    for t in text_config.layer_types:
        glob = t == "full_attention"
        heads = text_config.num_global_key_value_heads if (glob and text_config.attention_k_eq_v) \
            else text_config.num_key_value_heads
        hd = (text_config.global_head_dim or text_config.head_dim) if glob else text_config.head_dim
        out.append((1, heads, max_len if glob else int(text_config.sliding_window), hd))
    return out


class DecodeStep(nn.Module):
    """One token through the whole model with a static KV cache. forward(input_ids [1,1], pos [1]) ->
    float32 logits [1, vocab]."""

    def __init__(self, model: nn.Module, max_len: int = MAX_LEN, device=None, dtype=None):
        super().__init__()
        self.model = model
        tc = model.config.get_text_config()
        assert tc._attn_implementation == "sdpa", tc._attn_implementation
        self.window = int(tc.sliding_window)
        self.max_len = max_len
        self.sliding = [t == "sliding_attention" for t in tc.layer_types]
        device = device or next(model.parameters()).device
        dtype = dtype or next(model.parameters()).dtype
        for i, shp in enumerate(kv_shapes(tc, max_len)):
            # persistent=True so they show up in state_dict (AOTI constants)
            self.register_buffer(f"k{i}", torch.zeros(shp, dtype=dtype, device=device))
            self.register_buffer(f"v{i}", torch.zeros(shp, dtype=dtype, device=device))
        self.n_layers = len(tc.layer_types)

    def kv(self, i):
        return getattr(self, f"k{i}"), getattr(self, f"v{i}")

    def forward(self, input_ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        full = (torch.arange(self.max_len, device=pos.device) <= pos).view(1, 1, 1, self.max_len)
        # ring slots: before the window fills, slots above pos are empty; after, every slot is in the window
        slide = (torch.arange(self.window, device=pos.device) <= pos).view(1, 1, 1, self.window)
        with _decode_attention_only():
            out = self.model(input_ids=input_ids, attention_mask={"full_attention": full, "sliding_attention": slide},
                             position_ids=pos.view(1, 1), past_key_values=_KV(self, pos), use_cache=True,
                             logits_to_keep=1)
        return out.logits[:, -1, :].float()


@torch.inference_mode()
def prefill(model, step: DecodeStep, input_ids: torch.Tensor) -> torch.Tensor:
    """Eager prefill exactly like generate's first step, then copy its K/V into the step's buffers."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config.get_text_config())
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    L = input_ids.shape[1]
    for i, layer in enumerate(cache.layers):
        k, v = layer.keys, layer.values
        n = k.shape[2]                                    # the last n prompt positions: L-n .. L-1
        sk, sv = step.kv(i)
        if step.sliding[i]:
            slots = torch.arange(L - n, L, device=k.device) % step.window
            sk.index_copy_(2, slots, k)
            sv.index_copy_(2, slots, v)
        else:
            sk[:, :, L - n:L].copy_(k)
            sv[:, :, L - n:L].copy_(v)
    return out.logits[:, -1, :].float()


class Sampler:
    """transformers' generate for this checkpoint: suppress_tokens, then top_p 0.95 (temperature 1.0 and
    top_k 0 are no-ops there and here)."""

    def __init__(self, vocab: int, device, suppress=(), top_p: float = 0.95, greedy: bool = False):
        self.top_p, self.greedy = top_p, greedy
        mask = torch.zeros(vocab, dtype=torch.bool, device=device)
        if suppress:
            mask[torch.tensor(list(suppress), device=device)] = True
        self.suppress = mask

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        scores = torch.where(self.suppress, -float("inf"), scores)
        if self.greedy:
            return torch.argmax(scores, dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1 - self.top_p)
        remove[..., -1:] = 0
        remove = remove.scatter(1, sorted_indices, remove)
        scores = scores.masked_fill(remove, -float("inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)


@torch.inference_mode()
def generate(model, step: DecodeStep, step_fn, input_ids: torch.Tensor, *, max_new: int, eos: int,
             sampler: Sampler, max_time: float | None = None, stop_flag: threading.Event | None = None,
             on_token=None, pipelined: bool = True, gaps: list | None = None):
    """Returns (new token ids list, info). on_token(int) is called for every generated token (streaming).
    Semantics match generate(): max_new counts the EOS, max_time counts from the start (prefill included),
    stop_flag is checked after every token like a StoppingCriteria.

    pipelined: launch step i+1 on the device before reading token i on the host (one wasted step at the end
    at most), so Python / launch overhead overlaps the GPU work."""
    t0 = time.time()
    dev = input_ids.device
    L = input_ids.shape[1]
    assert L + max_new <= step.max_len, f"prompt {L} + max_new {max_new} > static cache {step.max_len}"
    logits = prefill(model, step, input_ids)
    t_prefill = time.time() - t0
    pos = torch.tensor([L], dtype=torch.long, device=dev)
    ids = torch.empty((1, 1), dtype=torch.long, device=dev)
    out: list[int] = []
    tok = sampler(logits)
    if gaps is not None:   # diagnostics: top-1 minus top-2 logit at every step (greedy agreement checks)
        gaps.append(logits.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze())
    t_first = None
    reason = "length"
    while True:
        if pipelined and len(out) + 1 < max_new:
            ids.copy_(tok.view(1, 1))
            next_logits = step_fn(ids, pos)        # queued on the stream; host does not wait
            t = int(tok)                           # sync on the token that was just sampled
        else:
            t = int(tok)
            next_logits = None
        if t_first is None:
            t_first = time.time() - t0
        out.append(t)
        if on_token is not None:
            on_token(t)
        if t == eos:
            reason = "eos"
            break
        if len(out) >= max_new:
            reason = "length"
            break
        if max_time is not None and time.time() - t0 > max_time:
            reason = "time"
            break
        if stop_flag is not None and stop_flag.is_set():
            reason = "stop"
            break
        if next_logits is None:
            ids.copy_(tok.view(1, 1))
            next_logits = step_fn(ids, pos)
        pos.add_(1)
        tok = sampler(next_logits)
        if gaps is not None:
            gaps.append(next_logits.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze())
    return out, {"prefill_s": t_prefill, "ttft_s": t_first, "total_s": time.time() - t0, "reason": reason}


# ───────────────────────────────────────────────────────────── AOTInductor package (ZeroGPU)

def package_key(model, max_len: int = MAX_LEN) -> str:
    """Changes whenever anything the compiled code depends on changes."""
    import hashlib
    import inspect

    import transformers

    h = hashlib.sha256()
    for part in (torch.__version__, transformers.__version__, str(max_len), model.config.to_json_string(),
                 inspect.getsource(DecodeStep), inspect.getsource(_KV)):
        h.update(part.encode())
    return h.hexdigest()[:12]


def package_name(model, max_len: int = MAX_LEN) -> str:
    return f"decode_step-{package_key(model, max_len)}.pt2"


def compile_package(model, out_path: str, max_len: int = MAX_LEN, inductor_configs=None) -> dict:
    """Inside a GPU call: export the decode step and AOT-compile it (weights are not packaged; they are the
    model's own tensors, handed over at load time). Returns timings."""
    import shutil
    import tempfile
    from pathlib import Path

    import spaces

    t0 = time.time()
    step = DecodeStep(model, max_len=max_len)
    dev = step.k0.device
    ids = torch.zeros((1, 1), dtype=torch.long, device=dev)
    pos = torch.tensor([8], dtype=torch.long, device=dev)
    with torch.no_grad():
        ep = torch.export.export(step, args=(ids, pos), strict=False)
    t1 = time.time()
    tmp = tempfile.mkdtemp()
    spaces.aoti_compile_and_save(tmp, ep, inductor_configs)
    shutil.copy(Path(tmp) / "root" / "package.pt2", out_path)
    shutil.rmtree(tmp, ignore_errors=True)
    del step, ep
    torch.cuda.empty_cache()
    return {"export_s": round(t1 - t0, 1), "compile_s": round(time.time() - t1, 1)}


class AotiWorker:
    """Per-process state for a GPU worker: the KV buffers and the loaded AOTI model, re-created when ZeroGPU
    hands us a new process (forked workers do not share CUDA state)."""

    def __init__(self):
        self.pid, self.step, self.cm, self.ptr = None, None, None, None

    def get(self, model, package_path: str, max_len: int = MAX_LEN):
        """-> (step, step_fn, load_seconds). load_seconds is 0 when this process had it already."""
        anchor = next(model.parameters())
        if self.pid == os.getpid() and self.ptr == anchor.data_ptr():
            return self.step, self.cm, 0.0
        t0 = time.time()
        self.cm = self.step = None
        step = DecodeStep(model, max_len=max_len)
        try:  # spaces removes the extracted archive from /tmp when the worker dies (it never exits cleanly)
            from spaces.zero.torch.aoti import _register_aoti_cleanup as cleanup
        except Exception:  # pragma: no cover
            import contextlib
            cleanup = contextlib.nullcontext
        with cleanup():
            cm = torch._inductor.aoti_load_package(package_path)
        named = dict(step.named_parameters(remove_duplicate=False)) | dict(step.named_buffers(remove_duplicate=False))
        fqns = cm.get_constant_fqns()
        missing = [f for f in fqns if f not in named]
        if missing:
            raise RuntimeError(f"AOTI package needs constants the model lacks: {missing[:3]}")
        cm.load_constants({f: named[f] for f in fqns}, check_full_update=True, user_managed=True)
        self.pid, self.step, self.cm, self.ptr = os.getpid(), step, cm, anchor.data_ptr()
        return step, cm, time.time() - t0
