"""Upload validation + derivation (OWASP file-upload hardening).

- `sniff()` trusts **magic bytes**, never the client MIME/extension.
- Images are **re-encoded** through Pillow: strips EXIF/GPS/ICC and any
  polyglot payload riding the container; decode failure ⇒ reject.
- Videos are probed (metadata + poster frame) via torchcodec, which dlopens the
  FFmpeg libs vendored in `.venv/lib/ffmpeg` (see scripts/deploy/run_backend.sh);
  no ffmpeg/ffprobe CLI exists on the box. Probe failure degrades gracefully —
  the blob is stored with unknown dimensions and no poster.

Everything here is blocking (Pillow/torchcodec/disk) — call via
`asyncio.to_thread`, never on the event loop.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple

from ..logging_conf import get_logger

log = get_logger(__name__)

# magic-byte table: (prefix-predicate, mime, kind)
_EBML = b"\x1aE\xdf\xa3"


def sniff(head: bytes) -> Optional[Tuple[str, str]]:
    """First bytes of the payload → (mime, kind) or None if unrecognized."""
    if len(head) < 12:
        return None
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg", "image"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp", "image"
    if head[:4] == b"GIF8":
        return "image/gif", "image"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand[:2] == b"qt":
            return "video/quicktime", "video"
        return "video/mp4", "video"
    if head[:4] == _EBML:
        # EBML container: webm and mkv share it; serve as webm (browser-playable)
        return "video/webm", "video"
    return None


_SAVE_FORMAT = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP", "image/gif": "PNG"}
_SAVE_MIME = {"image/gif": "image/png"}  # gif → first frame as png


@dataclass
class ImageResult:
    data: bytes          # re-encoded, metadata-free
    mime: str
    width: int
    height: int
    thumb: Optional[bytes]  # JPEG thumbnail


def process_image(data: bytes, sniffed_mime: str, thumb_max_edge: int) -> ImageResult:
    """Validate + re-encode an image, returning clean bytes and a thumbnail.

    Re-encoding through Pillow drops EXIF/GPS/ICC and neutralizes polyglot
    files; a payload Pillow cannot fully decode raises (⇒ reject upstream).
    """
    from PIL import Image

    Image.open(io.BytesIO(data)).verify()          # structural check
    img = Image.open(io.BytesIO(data))             # verify() invalidates — reopen
    img.load()

    fmt = _SAVE_FORMAT.get(sniffed_mime, "JPEG")
    out_mime = _SAVE_MIME.get(sniffed_mime, sniffed_mime)
    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    try:
        img.save(buf, format=fmt, quality=90)
    except Exception:                              # e.g. exotic mode for WEBP
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        out_mime = "image/jpeg"

    return ImageResult(
        data=buf.getvalue(),
        mime=out_mime,
        width=img.width,
        height=img.height,
        thumb=make_thumb(img, thumb_max_edge),
    )


def make_thumb(img, max_edge: int) -> Optional[bytes]:
    try:
        thumb = img.convert("RGB")
        thumb.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        log.warning("thumbnail derivation failed: %s", exc)
        return None


@dataclass
class VideoResult:
    width: Optional[int]
    height: Optional[int]
    duration_s: Optional[float]
    poster: Optional[bytes]  # JPEG poster frame


def probe_video(path: str, thumb_max_edge: int) -> VideoResult:
    """Best-effort metadata + poster via torchcodec. Never raises."""
    try:
        from PIL import Image
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(path)
        meta = dec.metadata
        width = getattr(meta, "width", None)
        height = getattr(meta, "height", None)
        duration = getattr(meta, "duration_seconds", None)
        poster: Optional[bytes] = None
        try:
            frame = dec[0]  # CHW uint8 tensor
            img = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
            poster = make_thumb(img, thumb_max_edge)
        except Exception as exc:  # noqa: BLE001
            log.warning("poster extraction failed for %s: %s", path, exc)
        return VideoResult(width=width, height=height,
                           duration_s=float(duration) if duration is not None else None,
                           poster=poster)
    except Exception as exc:  # noqa: BLE001
        log.warning("video probe unavailable for %s: %s", path, exc)
        return VideoResult(width=None, height=None, duration_s=None, poster=None)
