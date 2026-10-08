"""Bounded local caching and video-frame decoding for X attachments."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from vidur.storage import cache_dir

IMAGE_LIMIT = 20 * 1024 * 1024
VIDEO_LIMIT = 120 * 1024 * 1024
MEDIA_HOST_SUFFIX = ".twimg.com"
USER_AGENT = "Vidur/0.2 media viewer"


class MediaError(RuntimeError):
    """The attachment cannot safely be retrieved or rendered."""


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host.endswith(MEDIA_HOST_SUFFIX):
        raise MediaError("Only HTTPS media served from X's twimg.com CDN can be previewed.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise MediaError("The media URL contains an invalid port.") from exc
    if parsed.username or parsed.password or port not in (None, 443):
        raise MediaError("The media URL contains an unsupported authority or port.")


def download_media(url: str, *, kind: str = "image") -> Path:
    """Fetch one selected attachment into Vidur's bounded user cache."""
    _validate_url(url)
    limit = VIDEO_LIMIT if kind == "video" else IMAGE_LIMIT
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    target_dir = cache_dir() / "media"
    target_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".gif"):
        candidate = target_dir / f"{digest}{suffix}"
        if candidate.is_file():
            return candidate
    temporary = None
    target = None
    try:
        current_url = url
        for redirect_count in range(4):
            _validate_url(current_url)
            with requests.get(current_url, headers={"User-Agent": USER_AGENT}, timeout=(8, 30),
                              stream=True, allow_redirects=False) as response:
                if response.is_redirect:
                    destination = response.headers.get("location")
                    if not destination or redirect_count == 3:
                        raise MediaError("The media server returned too many or invalid redirects.")
                    current_url = urljoin(current_url, destination)
                    _validate_url(current_url)
                    continue
                response.raise_for_status()
                _validate_url(response.url)
                mime = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if kind == "video" and not (mime.startswith("video/") or "octet-stream" == mime):
                    raise MediaError(f"The attachment is not a video (server returned {mime or 'unknown type'}).")
                if kind != "video" and not mime.startswith("image/"):
                    raise MediaError(f"The attachment is not an image (server returned {mime or 'unknown type'}).")
                try:
                    declared = int(response.headers.get("content-length", "0") or 0)
                except ValueError as exc:
                    raise MediaError("The media server returned an invalid content length.") from exc
                if declared > limit:
                    raise MediaError(f"The attachment is larger than the {limit // (1024 * 1024)} MB preview limit.")
                extension = {
                    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
                    "image/gif": ".gif", "video/mp4": ".mp4", "application/octet-stream": ".mp4",
                }.get(mime)
                if not extension:
                    raise MediaError(f"Unsupported media type: {mime or 'unknown'}.")
                target = target_dir / f"{digest}{extension}"
                temporary = target.with_suffix(target.suffix + ".part")
                total = 0
                with temporary.open("wb") as media_file:
                    for chunk in response.iter_content(64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > limit:
                            raise MediaError(f"The attachment exceeded the {limit // (1024 * 1024)} MB preview limit.")
                        media_file.write(chunk)
                break
    except MediaError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    except (requests.RequestException, OSError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise MediaError(f"Could not fetch media: {exc}") from exc
    try:
        if temporary is None or target is None:
            raise MediaError("Could not prepare a local media cache file.")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _prune_cache(target_dir)
    return target


def _prune_cache(folder: Path, *, max_bytes: int = 500 * 1024 * 1024) -> None:
    files = [path for path in folder.iterdir() if path.is_file() and not path.name.endswith(".part")]
    total = sum(path.stat().st_size for path in files)
    for path in sorted(files, key=lambda entry: entry.stat().st_mtime):
        if total <= max_bytes:
            break
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        total -= size


def video_frames(path: Path, stop_event, playing_event, *, frames_per_second: float = 8):
    """Yield downsampled Pillow frames until playback is paused or stopped."""
    try:
        import av
    except ImportError as exc:
        raise MediaError("Video playback needs PyAV. Reinstall Vidur's dependencies.") from exc
    frame_interval = 1 / max(1, min(frames_per_second, 12))
    next_frame_at = 0.0
    last_displayed_at = time.monotonic()
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise MediaError("The video contains no video stream.")
        for frame in container.decode(stream):
            if stop_event.is_set():
                return
            while not playing_event.is_set() and not stop_event.is_set():
                time.sleep(0.08)
            if stop_event.is_set():
                return
            moment = float(frame.time or 0)
            if moment < next_frame_at:
                continue
            next_frame_at = moment + frame_interval
            yield frame.to_image()
            remaining = frame_interval - (time.monotonic() - last_displayed_at)
            if remaining > 0:
                time.sleep(remaining)
            last_displayed_at = time.monotonic()


def video_poster(path: Path):
    """Decode the first video frame when the source did not provide a poster."""
    try:
        import av
    except ImportError as exc:
        raise MediaError("Video previews need PyAV. Reinstall Vidur's dependencies.") from exc
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise MediaError("The video contains no video stream.")
        first_frame = next(container.decode(stream), None)
        if first_frame is None:
            raise MediaError("The video contains no decodable frames.")
        return first_frame.to_image()
