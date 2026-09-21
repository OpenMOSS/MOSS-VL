"""Embedders for the memory index — pluggable, with dependency-free fallbacks.

Two spaces: `text` (utterances, captions, facts) and `image` (keyframes).

The fallbacks matter: this box is air-gapped and ships no embedding weights, so
a hard dependency on BGE-M3 / Chinese-CLIP would make the whole memory plane
untestable until someone carries weights across. Instead each space has a
deterministic, numpy-only default that is good enough for dedup and lexical-ish
recall, and a real model is a config swap (`MEMORY_EMBED_TEXT_MODEL`).

`cross_modal` is the load-bearing capability flag: the fallback image embedder
cannot encode *text*, so a text query can never score against raw frame vectors.
Retrieval reads this flag and routes frame recall through captions instead
(caption-mediated retrieval, which the video-RAG literature finds competitive
with direct text→image anyway). Swapping in Chinese-CLIP flips the flag and the
direct visual lane lights up with no other change.
"""
from __future__ import annotations

import io
import os
import re
import zlib
from typing import List, Optional, Sequence

import numpy as np

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _l2(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=-1, keepdims=True)
    np.maximum(norm, 1e-9, out=norm)
    return mat / norm


# --------------------------------------------------------------------- text

class HashingTextEmbedder:
    """Signed-hash bag of character/word n-grams. Deterministic, no weights.

    CJK contributes uni+bigrams (a hanzi is already a morpheme, bigrams carry
    the word); latin contributes whole words plus 3-grams so morphology and
    typos still overlap. This is a random projection of a bag-of-ngrams — it
    models lexical overlap, not semantics, which is the honest ceiling without
    a model, and it is stable across processes (crc32, not salted hash()).
    """

    name = "hashing-ngram"
    cross_modal = False
    supports_late = False  # no token-level head without a model behind it
    # absolute-cosine floor the gate starts from. Lexical n-gram overlap between
    # two genuinely related short zh sentences lands ~0.25-0.35, far below what a
    # trained encoder reports for the same pair — hence per-embedder, not one
    # global constant (see Retriever.gate).
    gate_floor = 0.22

    def __init__(self, dim: int = 512) -> None:
        self.dim = int(dim)

    def _tokens(self, text: str) -> List[str]:
        out: List[str] = []
        chars = text or ""
        for i, ch in enumerate(chars):
            cp = ord(ch)
            if 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
                out.append(ch)
                if i + 1 < len(chars):
                    out.append(ch + chars[i + 1])
        for word in _WORD_RE.findall(chars.lower()):
            out.append(word)
            if len(word) > 3:
                out.extend(word[j:j + 3] for j in range(len(word) - 2))
        return out

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in self._tokens(text):
                h = zlib.crc32(tok.encode("utf-8", "ignore"))
                mat[row, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return _l2(mat)


class HfTextEmbedder:
    """Any HF encoder (BGE-M3, Qwen3-Embedding, …).

    Loaded lazily on first encode: the weights are ~2 GB and a session that
    never recalls anything should not pay for them. The first encode always
    happens on the writer thread or inside asyncio.to_thread, never on the loop.

    Pooling follows the model's own convention — BGE's dense vector is the CLS
    token, and mean-pooling it instead quietly degrades retrieval.
    """

    cross_modal = False
    # flipped True only once the colbert head (colbert_linear.pt) has ACTUALLY
    # loaded — writer/retrieval must not trust the config flag alone
    supports_late = False
    # measured on this checkpoint: related zh pair 0.71, unrelated 0.50. XLM-R
    # encoders have a high similarity floor, so the gate sits well above 0.5.
    gate_floor = 0.62
    # maxsim MEANS sit higher than pooled cosines (every query token gets its
    # best match), so the late-interaction lane needs its own floor. Sane
    # default; calibration-pending per design §10.
    li_gate_floor = 0.5

    def __init__(self, path: str, device: str = "cpu", max_len: int = 256) -> None:
        self.path = path
        self.name = os.path.basename(path.rstrip("/")) or path
        self.device = device
        self.max_len = max_len
        self.dim = 1024  # corrected at load
        self._tok = None
        self._model = None
        self._torch = None
        self._pooling = "cls" if ("bge" in self.name.lower()) else "mean"
        self._colbert = None
        self._colbert_tried = False

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer  # local: heavy import
        import torch
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.path, trust_remote_code=True)
        self._model = self._model.to(self.device).eval()
        self.dim = int(self._model.config.hidden_size)
        log.info("memory: text embedder %s loaded (dim=%d, %s, %s-pooled)",
                 self.name, self.dim, self.device, self._pooling)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure()
        torch = self._torch
        with torch.no_grad():
            batch = self._tok(list(texts), padding=True, truncation=True,
                              max_length=self.max_len, return_tensors="pt").to(self.device)
            out = self._model(**batch).last_hidden_state
            if self._pooling == "cls":
                pooled = out[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return _l2(pooled.float().cpu().numpy().astype(np.float32))

    def _ensure_colbert(self) -> bool:
        """Lazy-load the checkpoint's colbert head (colbert_linear.pt) like the
        main model: late interaction is an upgrade lane, and a box without the
        head (or with a corrupt one) just keeps pooled retrieval."""
        if self._colbert is not None:
            return True
        if self._colbert_tried:
            return False
        self._colbert_tried = True
        self._ensure()
        torch = self._torch
        path = os.path.join(self.path, "colbert_linear.pt")
        if not os.path.exists(path):
            log.info("memory: %s ships no colbert_linear.pt; late interaction off", self.name)
            return False
        try:
            state = torch.load(path, map_location="cpu")
            weight, bias = state["weight"], state.get("bias")
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
            with torch.no_grad():
                linear.weight.copy_(weight)
                if bias is not None:
                    linear.bias.copy_(bias)
            self._colbert = linear.to(self.device).eval()
            for param in self._colbert.parameters():
                param.requires_grad_(False)
            self.supports_late = True
            log.info("memory: colbert head loaded (%d->%d, %s); late interaction on",
                     weight.shape[1], weight.shape[0], self.device)
            return True
        except Exception as exc:  # noqa: BLE001 — never block a session on this
            log.warning("memory: colbert head failed to load (%s); late interaction off", exc)
            return False

    def encode_tokens(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Token-level vectors via the colbert head: one L2-normalized row per
        content token (CLS and padding dropped per the attention mask)."""
        if not self._ensure_colbert():
            raise RuntimeError(f"late interaction unavailable for {self.name}")
        torch = self._torch
        with torch.no_grad():
            batch = self._tok(list(texts), padding=True, truncation=True,
                              max_length=self.max_len, return_tensors="pt").to(self.device)
            out = self._model(**batch).last_hidden_state
            vecs = torch.nn.functional.normalize(self._colbert(out), p=2, dim=-1)
            mask = batch["attention_mask"].bool()
            result: List[np.ndarray] = []
            for row in range(vecs.shape[0]):
                keep = mask[row].clone()
                keep[0] = False  # drop CLS — it duplicates the pooled vector's job
                result.append(vecs[row][keep].float().cpu().numpy().astype(np.float32))
            return result


# -------------------------------------------------------------------- image

class TinyImageEmbedder:
    """Downscaled-luma + coarse RGB histogram descriptor (320-d), numpy only.

    Enough to tell "same scene" from "different scene" — which is all the write
    path needs (dedup). Semantic frame recall rides the caption text instead;
    see the module docstring on `cross_modal`.
    """

    name = "tiny-visual"
    cross_modal = False
    dim = 16 * 16 + 64

    def _load(self, jpeg: bytes):
        from PIL import Image  # local: PIL is already a persistence dep
        return Image.open(io.BytesIO(jpeg)).convert("RGB")

    def encode_images(self, images: Sequence[bytes]) -> np.ndarray:
        mat = np.zeros((len(images), self.dim), dtype=np.float32)
        for row, raw in enumerate(images):
            try:
                img = self._load(raw)
            except Exception:  # noqa: BLE001 — a corrupt frame is not fatal
                continue
            luma = np.asarray(img.convert("L").resize((16, 16)), dtype=np.float32) / 255.0
            luma -= luma.mean()
            small = np.asarray(img.resize((32, 32)), dtype=np.uint8) >> 6  # 4 bins/channel
            hist = np.zeros(64, dtype=np.float32)
            idx = (small[..., 0] * 16 + small[..., 1] * 4 + small[..., 2]).ravel()
            np.add.at(hist, idx, 1.0)
            hist /= max(1.0, hist.sum())
            mat[row] = np.concatenate([luma.ravel(), hist])
        return _l2(mat)


class ChineseClipEmbedder:
    """Chinese-CLIP: ONE space holding both keyframes and Chinese text.

    This is what makes "我刚才给你看的那个东西" answerable without a captioner —
    a text query scores directly against stored frame vectors. Lazily loaded
    like the text encoder.

    Text goes CLS → text_projection rather than `get_text_features`, because the
    checkpoint ships no BERT pooler and the convenience method dereferences it.
    """

    cross_modal = True
    # CLIP-family cosines are compressed into a narrow band (~0.1-0.35), so this
    # floor is much lower than a text encoder's and is the least-calibrated
    # number in the system — tune it on real camera frames, not synthetic ones.
    gate_floor = 0.22

    def __init__(self, path: str, device: str = "cpu") -> None:
        self.path = path
        self.name = os.path.basename(path.rstrip("/")) or path
        self.device = device
        self.dim = 1024  # corrected at load
        self._model = None
        self._proc = None
        self._torch = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        import torch
        self._torch = torch
        self._proc = ChineseCLIPProcessor.from_pretrained(self.path)
        self._model = ChineseCLIPModel.from_pretrained(self.path).to(self.device).eval()
        self.dim = int(self._model.config.projection_dim)
        log.info("memory: image embedder %s loaded (dim=%d, %s, cross-modal)",
                 self.name, self.dim, self.device)

    def encode_images(self, images: Sequence[bytes]) -> np.ndarray:
        self._ensure()
        from PIL import Image
        torch = self._torch
        pil, keep = [], []
        for i, raw in enumerate(images):
            try:
                pil.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                keep.append(i)
            except Exception:  # noqa: BLE001
                continue
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        if not pil:
            return out
        with torch.no_grad():
            batch = self._proc(images=pil, return_tensors="pt").to(self.device)
            feats = self._model.get_image_features(**batch)
        out[keep] = _l2(feats.float().cpu().numpy().astype(np.float32))
        return out

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure()
        torch = self._torch
        with torch.no_grad():
            batch = self._proc(text=list(texts), padding=True, truncation=True,
                               max_length=52, return_tensors="pt").to(self.device)
            batch.pop("pixel_values", None)
            cls = self._model.text_model(**batch).last_hidden_state[:, 0]
            feats = self._model.text_projection(cls)
        return _l2(feats.float().cpu().numpy().astype(np.float32))


def dhash(jpeg: bytes) -> Optional[int]:
    """64-bit difference hash — the cheap first gate for near-duplicate frames."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg)).convert("L").resize((9, 8))
    except Exception:  # noqa: BLE001
        return None
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).ravel()
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def hamming(a: Optional[int], b: Optional[int]) -> int:
    if a is None or b is None:
        return 64
    return bin(a ^ b).count("1")


# ------------------------------------------------------------------ builders

def build_text_embedder(settings: Settings):
    path = (settings.memory_embed_text_model or "").strip()
    if path and os.path.exists(path):
        try:
            return HfTextEmbedder(path, device=settings.memory_embed_device)
        except Exception as exc:  # noqa: BLE001 — never block a session on this
            log.warning("memory: text model %s failed (%s); using hashing fallback", path, exc)
    elif path:
        log.warning("memory: text model %s not found; using hashing fallback", path)
    log.info("memory: text embedder = hashing n-gram fallback (lexical overlap only)")
    return HashingTextEmbedder(settings.memory_text_dim)


def build_image_embedder(settings: Settings):
    path = (settings.memory_embed_image_model or "").strip()
    if path and os.path.exists(path):
        try:
            return ChineseClipEmbedder(path, device=settings.memory_embed_device)
        except Exception as exc:  # noqa: BLE001
            log.warning("memory: image model %s failed (%s); using descriptor fallback", path, exc)
    elif path:
        log.warning("memory: image model %s not found; using descriptor fallback", path)
    log.info("memory: image embedder = tiny descriptor (dedup only, not text-searchable)")
    return TinyImageEmbedder()
