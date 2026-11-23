"""WSP-Bench node library: fetch HF checkpoints, apply procedural transforms, disguise, verify.

Item layout on the node (ROOT/items/<item_id>/):
  public/<slot>/weights.safetensors   canonical tensor names, permuted (function-preserving)
  public/<slot>/arch.json             canonical architecture (no rope / context / eps / names)
  public/vocab.json                   id -> token string of the item's family tokenizer
  private/<slot>/hf/                  loadable HF dir (original names, permuted weights) for the forward server
  private/manifest.json               sources, resolved shas, transforms, permutation seeds (never mounted)
"""
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(os.environ.get("WSP_ROOT", "/ephemeral/wspbench"))
HF_CACHE = ROOT / "hf_cache"
# The node's ~/.cache is a link into /ephemeral, which a reboot empties: without this the Hub client's download cache
# (xet) fails with "File exists" on the dangling link (seen Nov 11 after the wipe). Keep every cache under ROOT.
if os.access(ROOT, os.W_OK):  # not inside the read-only sandboxes, which mount their own copy of this file
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
SUPPORTED = {"Qwen2ForCausalLM", "LlamaForCausalLM", "Olmo2ForCausalLM", "Qwen3ForCausalLM"}


def internals_note(layers, attn_saved, attn_asked, ids_saved=True):
    """What a hidden-state / attention pass wrote, in the tensor names' numbering: the benchmark's internals request
    (forward.py), the RL sandbox's hidden request and, for old transcripts, train/build_sft.py. v1-v8 said
    "hidden_L<l>.npy (l=0 embeddings, l=i after layer i)" for every request, attention-only ones included, while tensor
    names count from L0, so hidden_L<i> (Hugging Face's hidden_states[i]) is the input of L<i> (DIAGNOSIS.md 6)."""
    notes = []
    if layers:
        notes.append("hidden_L<l>.npy is the residual stream entering layer L<l> (tensor names count from L0): "
                     "hidden_L0 = the embeddings, hidden_L<l> = the output of L<l-1>, and hidden_L<n> for n = the "
                     "number of layers = the output of the last layer after the final norm")
    if attn_saved:
        notes.append("attn_L<l>.npy = the attention probabilities of layer L<l>, shape [text, head, query, key]")
    elif attn_asked:
        notes.append("no attention saved: attn_layers needs at most 16 texts")
    if ids_saved:
        notes.append("input_ids.npy and attention_mask.npy are left-padded")
    return "; ".join(notes)


# ---------- fetch ----------

def resolve(source):
    """source: 'hf:org/name@rev' or 'local:/abs/path'. Returns (local_dir, resolved_rev)."""
    if source.startswith("local:"):
        p = Path(source[6:])
        return p, "local:" + sha_dir(p)[:16]
    from huggingface_hub import snapshot_download, HfApi
    repo, _, rev = source[3:].partition("@")
    rev = rev or "main"
    full = HfApi().model_info(repo, revision=rev).sha
    d = snapshot_download(repo, revision=full, cache_dir=str(HF_CACHE),
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    return Path(d), full


def sha_dir(p):
    h = hashlib.sha256()
    for f in sorted(Path(p).glob("*.safetensors")):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
    return h.hexdigest()


def load_hf(d):
    d = Path(d)
    cfg = json.loads((d / "config.json").read_text())
    arch = cfg["architectures"][0]
    assert arch in SUPPORTED, f"unsupported architecture {arch}"
    sd = {}
    for f in sorted(d.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    sd = {k: v for k, v in sd.items() if not k.endswith("rotary_emb.inv_freq")}
    if cfg.get("tie_word_embeddings") and "lm_head.weight" in sd:
        del sd["lm_head.weight"]
    return cfg, sd


def dims(cfg):
    nh = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim") or cfg["hidden_size"] // nh
    return cfg["num_hidden_layers"], nh, nkv, hd, cfg["intermediate_size"]


# ---------- procedural transforms (operate on HF-named state dicts of identical architecture) ----------

def layer_keys(sd, i):
    pre = f"model.layers.{i}."
    return [k for k in sd if k.startswith(pre)]


def t_merge(sd_a, sd_b, alpha, noise=0.0, seed=0):
    """alpha: float, or list with one value per layer (non-layer tensors use the mean). noise (v6+): Gaussian noise of
    std noise * std(sd_b - sd_a) per tensor, so the coefficient is still estimable by regression but cannot be read
    off by matching the stored bfloat16 bits exactly."""
    out = {}
    g = torch.Generator().manual_seed(seed)
    nl = 1 + max(int(m.group(1)) for k in sd_a for m in [re.match(r"model\.layers\.(\d+)\.", k)] if m)
    al = alpha if isinstance(alpha, list) else [alpha] * nl
    amean = sum(al) / len(al)
    for k, va in sd_a.items():
        m = re.match(r"model\.layers\.(\d+)\.", k)
        a = al[int(m.group(1))] if m else amean
        m = (1 - a) * va.float() + a * sd_b[k].float()
        if noise:
            m = m + noise * float((sd_b[k].float() - va.float()).std()) * torch.randn(m.shape, generator=g)
        out[k] = m.to(va.dtype)
    return out


def t_transplant(sd_a, sd_b, layers, match_norm=False):
    """Replace layers of a by b's; match_norm rescales each donor tensor to the Frobenius norm of the tensor it
    replaces (and, for 1-D gains, matches mean and std) so a per-layer norm profile does not reveal the block."""
    out = dict(sd_a)
    for i in layers:
        for k in layer_keys(sd_b, i):
            t = sd_b[k].float()
            if match_norm:
                a = sd_a[k].float()
                if t.dim() == 1:
                    t = (t - t.mean()) / (t.std() + 1e-8) * a.std() + a.mean()
                else:
                    t = t * (a.norm() / (t.norm() + 1e-8))
            out[k] = t.to(sd_a[k].dtype)
    return out


def t_swap(sd, i, j):
    out = dict(sd)
    for k in layer_keys(sd, i):
        kj = k.replace(f"model.layers.{i}.", f"model.layers.{j}.")
        out[k], out[kj] = sd[kj].clone(), sd[k].clone()
    return out


def t_zero_heads(sd, cfg, layer, heads):
    _, nh, nkv, hd, _ = dims(cfg)
    out = dict(sd)
    k = f"model.layers.{layer}.self_attn.o_proj.weight"
    w = out[k].clone()
    for h in heads:
        w[:, h * hd:(h + 1) * hd] = 0
    out[k] = w
    return out


def t_scale_neurons(sd, layer, neurons, scale):
    out = dict(sd)
    k = f"model.layers.{layer}.mlp.down_proj.weight"
    w = out[k].clone()
    w[:, neurons] = (w[:, neurons].float() * scale).to(w.dtype)
    out[k] = w
    return out


def t_rank1(sd, layer, module, u, v, strength):
    """W += strength * ||W||_F * u v^T / (||u|| ||v||); module in {'down','o'}."""
    out = dict(sd)
    name = {"down": "mlp.down_proj", "o": "self_attn.o_proj"}[module]
    k = f"model.layers.{layer}.{name}.weight"
    w = out[k].float()
    u = u / u.norm()
    v = v / v.norm()
    out[k] = (w + strength * w.norm() * torch.outer(u, v)).to(sd[k].dtype)
    return out


def t_scale_norm(sd, layer, which, scale):
    out = dict(sd)
    k = f"model.layers.{layer}.{which}.weight"
    out[k] = (sd[k].float() * scale).to(sd[k].dtype)
    return out


def t_lowbits(sd, bits, seed):
    """Re-draw the lowest `bits` mantissa bits of every bfloat16 weight (P2 from v8 on, fixed in place Nov 15).

    Each element moves by k units in the last place, k = U + B with U uniform on {-2^(bits-1), ..., 2^(bits-1) - 1}
    and B a fair coin: mean 0, |k| <= 2^(bits-1), and the lowest `bits` bits come out uniform whatever they were. v8's
    P2 transplanted layers kept the donor checkpoint's rounding fingerprint (the parity of the last mantissa bit, |z| up
    to 174 against <= 2.5 in the other layers); applied to every tensor alike, this leaves no bit-level statistic that
    tells one layer's provenance from another's. Zeros, subnormals, infinities and NaNs are left as they are."""
    g = torch.Generator().manual_seed(seed)
    half = 1 << (bits - 1)
    out = {}
    for k, v in sd.items():
        if v.dtype != torch.bfloat16:
            out[k] = v
            continue
        x = v.contiguous().view(torch.int16)  # sign bit, 8 exponent bits, 7 mantissa bits
        mag = x & 0x7FFF
        step = (torch.randint(-half, half, x.shape, generator=g, dtype=torch.int16)
                + torch.randint(0, 2, x.shape, generator=g, dtype=torch.int16))
        ok = (mag >= 0x0080 + half) & (mag < 0x7F80 - half)  # normal numbers that stay normal and finite
        out[k] = (torch.where(ok, mag + step, mag) | (x & -0x8000)).view(torch.bfloat16)
    return out


def apply_transforms(base_sd, cfg, transforms, others):
    """transforms: list of dicts {op: ..., ...}; others: dict slot -> state dict for merge/transplant partners."""
    sd = base_sd
    for t in transforms:
        op = t["op"]
        if op == "merge":
            sd = t_merge(sd, others[t["with"]], t["alpha"], t.get("noise", 0.0), t.get("noise_seed", 0))
        elif op == "transplant":
            sd = t_transplant(sd, others[t["with"]], t["layers"], t.get("match_norm", False))
        elif op == "swap":
            sd = t_swap(sd, t["i"], t["j"])
        elif op == "zero_heads":
            sd = t_zero_heads(sd, cfg, t["layer"], t["heads"])
        elif op == "scale_neurons":
            sd = t_scale_neurons(sd, t["layer"], t["neurons"], t["scale"])
        elif op == "rank1":
            g = torch.Generator().manual_seed(t["seed"])
            k = {"down": "mlp.down_proj", "o": "self_attn.o_proj"}[t["module"]]
            w = sd[f"model.layers.{t['layer']}.{k}.weight"]
            u = torch.randn(w.shape[0], generator=g)
            v = torch.randn(w.shape[1], generator=g)
            sd = t_rank1(sd, t["layer"], t["module"], u, v, t["strength"])
        elif op == "token_rank1":
            # write the (normalized) input-embedding direction of one token into a layer's output matrix
            g = torch.Generator().manual_seed(t["seed"])
            k = {"down": "mlp.down_proj", "o": "self_attn.o_proj"}[t["module"]]
            w = sd[f"model.layers.{t['layer']}.{k}.weight"]
            u = sd["model.embed_tokens.weight"][t["token_id"]].float()
            v = torch.randn(w.shape[1], generator=g)
            sd = t_rank1(sd, t["layer"], t["module"], u, v, t["strength"])
        elif op == "scale_norm":
            sd = t_scale_norm(sd, t["layer"], t["which"], t["scale"])
        elif op == "lowbits":
            sd = t_lowbits(sd, t["bits"], t["seed"])
        else:
            raise ValueError(op)
    return sd


# ---------- function-preserving permutation ----------

def make_perms(cfg, seed):
    nl, nh, nkv, hd, ff = dims(cfg)
    g = torch.Generator().manual_seed(seed)
    perms = []
    group = nh // nkv
    for _ in range(nl):
        pk = torch.randperm(nkv, generator=g)
        q_order = []
        for a in range(nkv):
            old = int(pk[a])
            pq = torch.randperm(group, generator=g)
            q_order += [old * group + int(b) for b in pq]
        perms.append({"mlp": torch.randperm(ff, generator=g), "kv": pk, "q": torch.tensor(q_order)})
    return perms


def _rows_by_head(w, order, hd):
    """Reorder dim 0 of w (n*hd, ...) by head order."""
    idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in order.tolist()])
    return w[idx]


def permute(sd, cfg, perms):
    nl, nh, nkv, hd, ff = dims(cfg)
    out = dict(sd)
    for i, p in enumerate(perms):
        pre = f"model.layers.{i}."
        for n in ("gate_proj", "up_proj"):
            out[pre + f"mlp.{n}.weight"] = sd[pre + f"mlp.{n}.weight"][p["mlp"]]
        out[pre + "mlp.down_proj.weight"] = sd[pre + "mlp.down_proj.weight"][:, p["mlp"]]
        for n, order in (("q_proj", p["q"]), ("k_proj", p["kv"]), ("v_proj", p["kv"])):
            for suf in ("weight", "bias"):
                k = pre + f"self_attn.{n}.{suf}"
                if k in sd:
                    out[k] = _rows_by_head(sd[k], order, hd)
        ko = pre + "self_attn.o_proj.weight"
        out[ko] = _rows_by_head(sd[ko].T, p["q"], hd).T.contiguous()
        for n, order, nn in (("q_norm", p["q"], nh), ("k_norm", p["kv"], nkv)):
            k = pre + f"self_attn.{n}.weight"
            if k in sd and sd[k].shape[0] == nn * hd:   # OLMo2-style full-width qk norm
                out[k] = _rows_by_head(sd[k], order, hd)
    return out


def residual_permute(sd, cfg, seed):
    """Signed permutation of the residual stream: x' = s * x[R]. Function-preserving for RMSNorm models.

    Readers of the residual (q,k,v,gate,up, lm_head, embed rows) get columns W[:, R] * s; writers (o, down) get rows
    s[:, None] * W[R]; RMSNorm gains on the residual get w[R].
    """
    d = cfg["hidden_size"]
    g = torch.Generator().manual_seed(seed ^ 0x5EED)
    R = torch.randperm(d, generator=g)
    s = (torch.randint(0, 2, (d,), generator=g) * 2 - 1)
    out = {}
    for k, w in sd.items():
        if k in ("model.embed_tokens.weight", "lm_head.weight") or re.search(
                r"\.(q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight$", k):
            out[k] = (w[:, R] * s.to(w.dtype)).contiguous()
        elif re.search(r"\.(o_proj|down_proj)\.weight$", k):
            out[k] = (w[R] * s.to(w.dtype)[:, None]).contiguous()
        elif k == "model.norm.weight" or re.search(
                r"\.(input_layernorm|post_attention_layernorm|post_feedforward_layernorm)\.weight$", k):
            out[k] = w[R].contiguous()
        else:
            out[k] = w
    return out


# ---------- canonical names / architecture ----------

def canon_map(cfg):
    arch = cfg["architectures"][0]
    post_norm = arch == "Olmo2ForCausalLM"
    m = [
        (r"^model\.embed_tokens\.weight$", "embed.weight"),
        (r"^model\.norm\.weight$", "final_norm.weight"),
        (r"^lm_head\.weight$", "lm_head.weight"),
        (r"^model\.layers\.(\d+)\.self_attn\.([qkvo])_proj\.(weight|bias)$", r"L\1.attn.\2.\3"),
        (r"^model\.layers\.(\d+)\.self_attn\.([qk])_norm\.weight$", r"L\1.attn.\2_norm.weight"),
        (r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$", r"L\1.mlp.\2.weight"),
        (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"L\1.attn_in_norm.weight"),
    ]
    if post_norm:
        m += [(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"L\1.attn_out_norm.weight"),
              (r"^model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$", r"L\1.mlp_out_norm.weight")]
    else:
        m += [(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"L\1.mlp_in_norm.weight")]
    return m


def canonicalize(sd, cfg):
    rules = canon_map(cfg)
    out = {}
    for k, v in sd.items():
        for pat, rep in rules:
            if re.match(pat, k):
                out[re.sub(pat, rep, k)] = v.contiguous()
                break
        else:
            raise KeyError(f"no canonical name for {k}")
    return out


def canon_arch(cfg, sd):
    nl, nh, nkv, hd, ff = dims(cfg)
    arch = cfg["architectures"][0]
    qn = sd.get("model.layers.0.self_attn.q_norm.weight")
    return {
        "n_layers": nl, "d_model": cfg["hidden_size"], "n_heads": nh, "n_kv_heads": nkv, "head_dim": hd,
        "d_ff": ff, "vocab_size": sd["model.embed_tokens.weight"].shape[0],
        "norm": "rmsnorm", "norm_placement": "post (attn_out_norm, mlp_out_norm)" if arch == "Olmo2ForCausalLM"
        else "pre (attn_in_norm, mlp_in_norm)",
        "mlp": "swiglu: down(silu(gate x) * up x)",
        "attention": "causal grouped-query self-attention with rotary position embeddings (parameter-free)",
        "qkv_bias": "model.layers.0.self_attn.q_proj.bias" in sd,
        "qk_norm": "none" if qn is None else ("per-head" if qn.shape[0] == hd else "full-width"),
        "tied_embeddings": "lm_head.weight" not in sd,
        "tensor_layout": "PyTorch Linear convention: weight[out_features, in_features]. MLP neurons and attention "
                         "heads appear in an arbitrary order, and the residual-stream dimensions are in an arbitrary "
                         "order with arbitrary signs (consistently across all tensors of a model).",
    }


# ---------- build one slot ----------

def write_slot(item_dir, slot, cfg, sd, src_dir, seed, residual=True):
    perms = make_perms(cfg, seed)
    psd = permute(sd, cfg, perms)
    if residual:
        psd = residual_permute(psd, cfg, seed)
    pub = item_dir / "public" / slot
    prv = item_dir / "private" / slot / "hf"
    pub.mkdir(parents=True, exist_ok=True)
    prv.mkdir(parents=True, exist_ok=True)
    save_file(canonicalize(psd, cfg), str(pub / "weights.safetensors"))
    (pub / "arch.json").write_text(json.dumps(canon_arch(cfg, sd), indent=1))
    save_file({k: v.contiguous() for k, v in psd.items()}, str(prv / "model.safetensors"))
    for f in Path(src_dir).iterdir():
        if f.suffix in (".json", ".txt", ".model", ".jinja") and "index" not in f.name:
            shutil.copy(f, prv / f.name)
    for f in list(pub.iterdir()) + list(prv.iterdir()):
        os.chmod(f, 0o644)
    return {"seed": seed}


def write_vocab(item_dir, tok_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    vocab = {i: tok.convert_ids_to_tokens(i) for i in range(len(tok))}
    (item_dir / "public" / "vocab.json").write_text(json.dumps(vocab))
    os.chmod(item_dir / "public" / "vocab.json", 0o644)


# ---------- verification ----------

@torch.no_grad()
def max_logit_diff(dir_a, dir_b, device="cuda", prompts=("The capital of France is", "def fib(n):", "12 * 7 = ")):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(dir_a))
    worst = 0.0
    ma = AutoModelForCausalLM.from_pretrained(str(dir_a), dtype=torch.float32).to(device).eval()
    mb = AutoModelForCausalLM.from_pretrained(str(dir_b), dtype=torch.float32).to(device).eval()
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        la, lb = ma(ids).logits, mb(ids).logits
        worst = max(worst, float((la - lb).abs().max() / la.abs().max()))
    del ma, mb
    torch.cuda.empty_cache()
    return worst
