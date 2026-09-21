"""Content-addressed media store (CAS) for uploaded images/videos.

Identity = content hash: `media/blobs/<algo>/<ab>/<cd>/<hex>`. Re-uploading the
same bytes is a no-op (dedup for free), blobs are immutable (HTTP `immutable`
caching), and filenames are never user-controlled (no traversal). Derivatives
(thumbnail / poster) live under `media/derived/<ab>/<hex>/`.

Ingest goes through `ingest.py` hardening; images are re-encoded (EXIF/polyglot
strip) **before** hashing, so the stored hash names the clean bytes. Video
uploads are hashed straight off the request spool (dedup costs zero store
I/O), then new content lands in `media/tmp/*.part` and reaches `blobs/` only
via `os.replace` (atomic — same filesystem), so a crash never leaves a
half-written blob.

All methods are blocking — call off-loop (`ingest_upload` runs as one job on
the routers' dedicated media-ingest executor; the rest via `asyncio.to_thread`).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import uuid
from typing import Any, Dict, Optional

from ..config import Settings
from ..logging_conf import get_logger
from . import ingest
from .store import IndexStore

log = get_logger(__name__)

_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
# images are re-encoded in memory; anything this large is not a chat image
_IMAGE_MAX_BYTES = 64 * 1024 * 1024
# hash/copy stride: big enough that syscalls don't dominate, small enough that
# 8 concurrent ingest workers hold a bounded ~32 MiB between them
_COPY_CHUNK = 4 * 1024 * 1024


class MediaError(ValueError):
    """Rejected upload (bad type / too large / undecodable)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def normalize_hash(handle: str) -> Optional[str]:
    """`sha256:<hex>` or bare 64-hex → bare hex, else None."""
    s = (handle or "").strip().lower()
    if s.startswith("sha256:"):
        s = s[len("sha256:"):]
    return s if _HEX_RE.match(s) else None


class MediaStore:
    def __init__(self, settings: Settings, index: IndexStore):
        self.settings = settings
        self.index = index
        self.data_dir = settings.data_dir
        self.algo = settings.media_hash_algo
        self.blobs_dir = os.path.join(self.data_dir, "media", "blobs", self.algo)
        self.derived_dir = os.path.join(self.data_dir, "media", "derived")
        self.tmp_dir = os.path.join(self.data_dir, "media", "tmp")
        self.allow = {m.strip() for m in settings.media_mime_allow.split(",") if m.strip()}

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> None:
        for d in (self.blobs_dir, self.derived_dir, self.tmp_dir):
            os.makedirs(d, exist_ok=True)
        for name in os.listdir(self.tmp_dir):  # orphaned partial uploads
            if name.endswith(".part"):
                try:
                    os.unlink(os.path.join(self.tmp_dir, name))
                except OSError:
                    pass
        log.info("media store open: %s (allow=%s)", os.path.join(self.data_dir, "media"),
                 ",".join(sorted(self.allow)))

    # ------------------------------------------------------------------ paths

    def blob_path(self, hex_: str) -> str:
        return os.path.join(self.blobs_dir, hex_[:2], hex_[2:4], hex_)

    def _derived_dir_for(self, hex_: str) -> str:
        return os.path.join(self.derived_dir, hex_[:2], hex_)

    def thumb_abspath(self, media_row: Dict[str, Any]) -> Optional[str]:
        rel = media_row.get("thumb_path") or media_row.get("poster_path")
        return os.path.join(self.data_dir, rel) if rel else None

    def new_part_path(self) -> str:
        return os.path.join(self.tmp_dir, f"{uuid.uuid4().hex}.part")

    # ------------------------------------------------------------------ ingest

    def put_bytes(self, data: bytes, orig_name: Optional[str] = None) -> Dict[str, Any]:
        """Store one in-memory payload (chat images / small files)."""
        sniffed = ingest.sniff(data[:64])
        if sniffed is None:
            raise MediaError("unrecognized file type (magic bytes)", status=415)
        mime, kind = sniffed
        if mime not in self.allow:
            raise MediaError(f"file type not allowed: {mime}", status=415)
        if kind == "image":
            if len(data) > _IMAGE_MAX_BYTES:
                raise MediaError("image too large", status=413)
            try:
                img = ingest.process_image(data, mime, self.settings.media_thumb_max_edge)
            except MediaError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise MediaError(f"undecodable image: {exc}") from exc
            return self._store_image(img, orig_name)
        # small video passed as bytes: spill to tmp and reuse the file path
        part = self.new_part_path()
        with open(part, "wb") as f:
            f.write(data)
        return self._store_video_part(part, mime, orig_name)

    def ingest_upload(self, src: Any, orig_name: Optional[str], cap: int) -> Dict[str, Any]:
        """One-shot blocking ingest from the upload's multipart spool (the
        route's `UploadFile.file` — a local, seekable temp file).

        Single executor job per upload (routers/media.py), sequenced so a
        duplicate or a reject costs as little as possible:
        - sniff/allowlist/size rejects fire BEFORE anything touches the store;
        - images never hit a tmp file — they re-encode in memory anyway;
        - videos are hashed straight off the spool (local disk), so a
          duplicate returns with ZERO store I/O; only new content pays the
          copy into blobs/.
        """
        src.seek(0)
        head = src.read(64)
        if not head:
            raise MediaError("empty upload")
        sniffed = ingest.sniff(head)
        if sniffed is None:
            raise MediaError("unrecognized file type (magic bytes)", status=415)
        mime, kind = sniffed
        if mime not in self.allow:
            raise MediaError(f"file type not allowed: {mime}", status=415)

        if kind == "image":
            limit = min(cap, _IMAGE_MAX_BYTES)
            data = head + src.read(max(0, limit - len(head)) + 1)
            if len(data) > limit:
                if len(data) > _IMAGE_MAX_BYTES:
                    raise MediaError("image too large", status=413)
                raise MediaError(f"file exceeds UPLOAD_MAX_BYTES ({cap})", status=413)
            try:
                img = ingest.process_image(data, mime, self.settings.media_thumb_max_edge)
            except Exception as exc:  # noqa: BLE001
                raise MediaError(f"undecodable image: {exc}") from exc
            return self._store_image(img, orig_name)

        # video — pass 1: hash the spool (no store I/O yet)
        h = hashlib.sha256(head)
        size = len(head)
        while chunk := src.read(_COPY_CHUNK):
            size += len(chunk)
            if size > cap:
                raise MediaError(f"file exceeds UPLOAD_MAX_BYTES ({cap})", status=413)
            h.update(chunk)
        hex_ = h.hexdigest()
        existing = self.index.get_media(hex_)
        if existing is not None and os.path.exists(self.blob_path(hex_)):
            self.index.touch_media(hex_)
            return existing
        # pass 2 (new content only): spool → tmp part → atomic move into blobs/
        src.seek(0)
        part = self.new_part_path()
        try:
            with open(part, "wb") as out:
                while chunk := src.read(_COPY_CHUNK):
                    out.write(chunk)
            return self._finalize_video_blob(part, hex_, size, mime, orig_name)
        except Exception:
            if os.path.exists(part):
                try:
                    os.unlink(part)
                except OSError:
                    pass
            raise

    def _store_image(self, img: ingest.ImageResult, orig_name: Optional[str]) -> Dict[str, Any]:
        hex_ = hashlib.sha256(img.data).hexdigest()
        existing = self.index.get_media(hex_)
        if existing is not None and os.path.exists(self.blob_path(hex_)):
            self.index.touch_media(hex_)
            return existing
        self._place_blob_bytes(hex_, img.data)
        thumb_rel = self._write_derived(hex_, "thumb.jpg", img.thumb)
        desc = self._descriptor(hex_, img.mime, "image", len(img.data), orig_name,
                                width=img.width, height=img.height,
                                thumb_path=thumb_rel, poster_path=None, duration_s=None)
        self.index.upsert_media(desc)
        return desc

    def _store_video_part(self, part_path: str, mime: str, orig_name: Optional[str]) -> Dict[str, Any]:
        h = hashlib.sha256()
        size = 0
        with open(part_path, "rb") as f:
            while chunk := f.read(_COPY_CHUNK):
                h.update(chunk)
                size += len(chunk)
        return self._finalize_video_blob(part_path, h.hexdigest(), size, mime, orig_name)

    def _finalize_video_blob(self, part_path: str, hex_: str, size: int, mime: str,
                             orig_name: Optional[str]) -> Dict[str, Any]:
        """Already-hashed part file → dedup-or-place + probe + index row.

        Safe under concurrent identical uploads: the dup check is a fast path,
        not a lock — two racers each os.replace the SAME bytes onto the same
        blob path (atomic, idempotent) and upsert the same row.
        """
        existing = self.index.get_media(hex_)
        blob = self.blob_path(hex_)
        if existing is not None and os.path.exists(blob):
            os.unlink(part_path)
            self.index.touch_media(hex_)
            return existing
        os.makedirs(os.path.dirname(blob), exist_ok=True)
        os.replace(part_path, blob)  # atomic: tmp/ and blobs/ share the data_dir fs
        probe = ingest.probe_video(blob, self.settings.media_thumb_max_edge)
        poster_rel = self._write_derived(hex_, "poster.jpg", probe.poster)
        desc = self._descriptor(hex_, mime, "video", size, orig_name,
                                width=probe.width, height=probe.height,
                                thumb_path=None, poster_path=poster_rel,
                                duration_s=probe.duration_s)
        self.index.upsert_media(desc)
        return desc

    def _place_blob_bytes(self, hex_: str, data: bytes) -> None:
        blob = self.blob_path(hex_)
        if os.path.exists(blob):
            return
        os.makedirs(os.path.dirname(blob), exist_ok=True)
        part = self.new_part_path()
        with open(part, "wb") as f:
            f.write(data)
        os.replace(part, blob)

    def _write_derived(self, hex_: str, name: str, data: Optional[bytes]) -> Optional[str]:
        if not data:
            return None
        d = self._derived_dir_for(hex_)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        part = path + ".part"
        with open(part, "wb") as f:
            f.write(data)
        os.replace(part, path)
        return os.path.relpath(path, self.data_dir)

    def _descriptor(self, hex_: str, mime: str, kind: str, size: int, orig_name: Optional[str],
                    **extra: Any) -> Dict[str, Any]:
        return {
            "hash": hex_, "algo": self.algo, "mime": mime, "kind": kind, "bytes": size,
            "created_at": time.time(), "orig_name": orig_name, **extra,
        }

    # ------------------------------------------------------------------ retrieval

    def load_bytes(self, handle: str) -> bytes:
        """Resolve a `sha256:<hex>` handle to blob bytes (chat/VLM path)."""
        hex_ = normalize_hash(handle)
        if hex_ is None:
            raise MediaError(f"invalid media handle: {handle!r}")
        path = self.blob_path(hex_)
        if not os.path.exists(path):
            raise MediaError(f"unknown media: {hex_[:12]}…")
        self.index.touch_media(hex_)
        with open(path, "rb") as f:
            return f.read()

    def rescan_blobs(self) -> int:
        """Re-derive `media` rows from the blobs on disk (--rebuild path).

        Dimensions/duration are probe-derived where cheap (sniff + existing
        derivatives); refs are re-established afterwards by journal replay.
        """
        count = 0
        for root, _dirs, files in os.walk(self.blobs_dir):
            for name in files:
                if not _HEX_RE.match(name):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, "rb") as f:
                        head = f.read(64)
                    sniffed = ingest.sniff(head)
                    if sniffed is None:
                        log.warning("rescan: unrecognized blob %s — skipped", name[:12])
                        continue
                    mime, kind = sniffed
                    derived = self._derived_dir_for(name)
                    thumb = os.path.join(derived, "thumb.jpg")
                    poster = os.path.join(derived, "poster.jpg")
                    self.index.upsert_media(self._descriptor(
                        name, mime, kind, os.path.getsize(path), None,
                        width=None, height=None, duration_s=None,
                        thumb_path=os.path.relpath(thumb, self.data_dir) if os.path.exists(thumb) else None,
                        poster_path=os.path.relpath(poster, self.data_dir) if os.path.exists(poster) else None,
                    ))
                    count += 1
                except OSError as exc:
                    log.warning("rescan failed for %s: %s", name[:12], exc)
        return count

    def delete_blob(self, hex_: str) -> None:
        """Remove a blob + its derivatives (prune tool; row deletion is the index's)."""
        if normalize_hash(hex_) != hex_:
            raise ValueError('invalid media hash')
        try:
            os.unlink(self.blob_path(hex_))
        except FileNotFoundError:
            pass
        d = self._derived_dir_for(hex_)
        try:
            shutil.rmtree(d)
        except FileNotFoundError:
            pass


# ---- process-global accessor (mirrors deps.set_runtime; lets the VLM adapter
# resolve CAS handles without importing the Runtime) ----

_store: Optional[MediaStore] = None


def set_media_store(store: Optional[MediaStore]) -> None:
    global _store
    _store = store


def maybe_get_media_store() -> Optional[MediaStore]:
    return _store


def resolve_blob_path(handle: str) -> Optional[str]:
    """CAS handle (`sha256:<hex>` / bare hex) → absolute blob path, else None.

    Path-only resolution (no index touch), so it also works in VLM worker
    processes, which share the gateway's DATA_DIR but never open a MediaStore.
    Videos need this: the VLM's video processor decodes from a file path, so
    handles resolve to the blob itself rather than loading bytes.
    """
    hex_ = normalize_hash(handle)
    if hex_ is None:
        return None
    if _store is not None:
        path = _store.blob_path(hex_)
    else:
        from ..config import get_settings

        s = get_settings()
        path = os.path.join(s.data_dir, "media", "blobs", s.media_hash_algo,
                            hex_[:2], hex_[2:4], hex_)
    return path if os.path.exists(path) else None
