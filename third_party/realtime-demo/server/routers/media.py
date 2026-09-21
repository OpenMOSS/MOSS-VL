"""Media upload + serving over the CAS store (server/persistence/media.py).

POST /api/media                multipart upload (image|video) → descriptor
GET  /api/media/{hash}/info    descriptor probe (client-side dedup: 404 = upload)
GET  /api/media/{hash}         blob; strong ETag, immutable caching, Range/206
GET  /api/media/{hash}/thumb   thumbnail (image) / poster frame (video)

Content is immutable (hash = identity), so blobs are served with
`Cache-Control: immutable` + `ETag: "sha256:<hex>"` — browsers cache forever
and revalidations 304. Range support makes <video> seeking work.
"""
from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from ..deps import Runtime, get_runtime
from ..logging_conf import get_logger
from ..persistence.media import MediaError, MediaStore

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["media"])

_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK = 1024 * 1024

# Uploads get their OWN small pool: asyncio.to_thread's default executor is
# the realtime plane's workhorse (poll_output/put_frame, ~4 jobs/s per live
# session), and a burst of queued uploads must never starve it. 8 workers is
# roughly what the shared FS sustains concurrently; excess uploads queue here
# and their clients simply wait — natural backpressure at any client count.
_INGEST_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="media-ingest")


def _store(rt: Runtime) -> MediaStore:
    if rt.media is None:
        raise HTTPException(status_code=503, detail="media store is disabled")
    return rt.media


def _descriptor_out(desc: Dict, hash_: str) -> Dict:
    has_thumb = bool(desc.get("thumb_path") or desc.get("poster_path"))
    return {
        "hash": f"sha256:{hash_}",
        "mime": desc.get("mime"),
        "kind": desc.get("kind"),
        "bytes": desc.get("bytes"),
        "width": desc.get("width"),
        "height": desc.get("height"),
        "duration_s": desc.get("duration_s"),
        "url": f"/api/media/{hash_}",
        "thumb_url": f"/api/media/{hash_}/thumb" if has_thumb else None,
    }


# ------------------------------------------------------------------ upload

@router.post("/media")
async def upload_media(file: UploadFile, rt: Runtime = Depends(get_runtime)):
    store = _store(rt)
    try:
        # ONE executor job per upload: the whole sniff→hash→dedup→copy
        # pipeline is blocking I/O (multipart spool + store fs) and runs
        # off-loop in a single hop — no per-chunk loop↔thread bouncing, and
        # nothing ever blocks the event loop. `file.file` is the parsed
        # multipart spool (local temp file), read synchronously in the worker.
        desc = await asyncio.get_running_loop().run_in_executor(
            _INGEST_POOL, store.ingest_upload, file.file,
            file.filename, rt.settings.upload_max_bytes)
    except MediaError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return _descriptor_out(desc, desc["hash"])


@router.get("/media/{hash_}/info")
async def get_media_info(hash_: str, rt: Runtime = Depends(get_runtime)):
    """Descriptor probe for client-side dedup.

    A browser that hashed its file locally asks here before shipping bytes
    (videos only in practice — image identity is the server-side re-encode).
    200 = reuse the handle, skip the upload; 404 = upload it.
    """
    store = _store(rt)
    if not _HEX_RE.match(hash_):
        raise HTTPException(status_code=400, detail="malformed media hash")

    def probe() -> Optional[Dict]:
        row = store.index.get_media(hash_)
        if row is None or not os.path.exists(store.blob_path(hash_)):
            return None
        store.index.touch_media(hash_)  # a reuse counts as a use (prune GC)
        return row

    row = await asyncio.to_thread(probe)
    if row is None:
        raise HTTPException(status_code=404, detail="media not found")
    return _descriptor_out(row, hash_)


# ------------------------------------------------------------------ serving

def _immutable_headers(etag: str, mime: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        "ETag": f'"{etag}"',
        "Cache-Control": "public, max-age=31536000, immutable",
        "X-Content-Type-Options": "nosniff",
        "Accept-Ranges": "bytes",
        "Content-Type": mime,
    }
    if extra:
        headers.update(extra)
    return headers


def _parse_range(header: str, size: int) -> Optional[Tuple[int, int]]:
    """Single-range `bytes=a-b` → inclusive (start, end); None = malformed/unsatisfiable."""
    m = _RANGE_RE.match(header.strip())
    if not m:
        return None
    start_s, end_s = m.groups()
    if start_s == "" and end_s == "":
        return None
    if start_s == "":                       # suffix: last N bytes
        n = int(end_s)
        if n <= 0:
            return None
        return max(0, size - n), size - 1
    start = int(start_s)
    end = int(end_s) if end_s else size - 1
    if start >= size or end < start:
        return None
    return start, min(end, size - 1)


async def _stream_file(path: str, start: int, end: int) -> AsyncIterator[bytes]:
    def read_chunk(offset: int, n: int) -> bytes:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(n)

    pos = start
    while pos <= end:
        n = min(_CHUNK, end - pos + 1)
        data = await asyncio.to_thread(read_chunk, pos, n)
        if not data:
            return
        pos += len(data)
        yield data


@router.get("/media/{hash_}")
async def get_media(hash_: str, request: Request, rt: Runtime = Depends(get_runtime)):
    store = _store(rt)
    if not _HEX_RE.match(hash_):
        raise HTTPException(status_code=400, detail="malformed media hash")
    row = await asyncio.to_thread(store.index.get_media, hash_)
    path = store.blob_path(hash_)
    if row is None or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="media not found")

    etag = f"sha256:{hash_}"
    mime = str(row["mime"])
    if_none_match = request.headers.get("if-none-match", "")
    if etag in if_none_match.replace('"', ""):
        return Response(status_code=304, headers=_immutable_headers(etag, mime))

    size = int(row["bytes"])
    range_header = request.headers.get("range")
    if range_header:
        rng = _parse_range(range_header, size)
        if rng is None:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        start, end = rng
        headers = _immutable_headers(etag, mime, {
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        })
        return StreamingResponse(_stream_file(path, start, end), status_code=206,
                                 headers=headers, media_type=mime)
    headers = _immutable_headers(etag, mime, {"Content-Length": str(size)})
    return StreamingResponse(_stream_file(path, 0, size - 1), headers=headers, media_type=mime)


@router.get("/media/{hash_}/thumb")
async def get_media_thumb(hash_: str, rt: Runtime = Depends(get_runtime)):
    store = _store(rt)
    if not _HEX_RE.match(hash_):
        raise HTTPException(status_code=400, detail="malformed media hash")
    row = await asyncio.to_thread(store.index.get_media, hash_)
    path = store.thumb_abspath(row) if row else None
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no thumbnail for this media")
    return FileResponse(path, media_type="image/jpeg",
                        headers=_immutable_headers(f"sha256:{hash_}/thumb", "image/jpeg"))
