"""Read-only adapters for the active Agent Reach platform backends.

Agent Reach selects and checks upstream tools; Vakula calls those tools
through fixed argument lists and never offers arbitrary command execution.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import yaml

from vakula.storage import insert_observation

MAX_OUTPUT_BYTES = 4_000_000
MAX_RESULTS = 25


class AgentReachError(RuntimeError):
    """Agent Reach or its selected upstream source is unavailable."""


@dataclass(frozen=True)
class ChannelStatus:
    platform: str
    status: str
    backend: str
    message: str = ""


def _run(args: list[str], *, timeout: int = 45, env: dict | None = None) -> str:
    try:
        completed = subprocess.run(
            args,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        raise AgentReachError(f"Required command is missing: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AgentReachError(f"{args[0]} timed out after {timeout} seconds.") from exc
    output = completed.stdout or ""
    if len(output.encode("utf-8", errors="replace")) > MAX_OUTPUT_BYTES:
        raise AgentReachError("Agent Reach returned more than 4 MB; reduce the result limit.")
    if completed.returncode:
        detail = (completed.stderr or output).strip()[:1200]
        raise AgentReachError(detail or f"{args[0]} exited with status {completed.returncode}.")
    return output


def _decode_structured(output: str):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(output)
        except yaml.YAMLError as exc:
            raise AgentReachError("The active Agent Reach backend did not return JSON or YAML.") from exc


def _channel_entries(payload) -> list[ChannelStatus]:
    if not isinstance(payload, dict):
        return []
    channels = payload.get("channels", payload)
    if isinstance(channels, list):
        result = []
        for value in channels:
            if isinstance(value, dict):
                platform = str(value.get("id") or value.get("platform") or value.get("channel") or "")
                if platform:
                    result.append(ChannelStatus(
                        platform=platform.lower(),
                        status=str(value.get("status", "unknown")),
                        backend=str(value.get("active_backend") or ""),
                        message=str(value.get("message", "")),
                    ))
        return result
    if not isinstance(channels, dict):
        return []
    result = []
    for name, value in channels.items():
        if not isinstance(value, dict):
            continue
        result.append(ChannelStatus(
            platform=str(name).lower(),
            status=str(value.get("status", "unknown")),
            backend=str(value.get("active_backend") or ""),
            message=str(value.get("message", "")),
        ))
    return result


def doctor() -> list[ChannelStatus]:
    """Run Agent Reach's documented, read-only machine-readable health check."""
    if not shutil.which("agent-reach"):
        raise AgentReachError("Agent Reach is not installed. Install it separately, then refresh Sources.")
    payload = _decode_structured(_run(["agent-reach", "doctor", "--json"], timeout=90))
    entries = _channel_entries(payload)
    if not entries:
        raise AgentReachError("Agent Reach doctor returned no channel status records.")
    return entries


def twitter_status() -> ChannelStatus:
    entries = doctor()
    entry = next((item for item in entries if item.platform in {"twitter", "x", "twitter/x"}), None)
    if entry is None:
        raise AgentReachError("Agent Reach did not report a Twitter/X channel.")
    return entry


def _pick(mapping: dict, *keys: str, default=""):
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return default


def _select_video_url(media: dict) -> str:
    video_info = media.get("video_info")
    if not isinstance(video_info, dict):
        video_info = {}
    variants = video_info.get("variants", [])
    candidates = [
        variant for variant in variants
        if isinstance(variant, dict)
        and "mp4" in str(variant.get("content_type", "")).lower()
        and variant.get("url")
    ]
    if candidates:
        def bitrate(item):
            try:
                return int(item.get("bitrate", 0) or 0)
            except (TypeError, ValueError):
                return 0
        return str(max(candidates, key=bitrate)["url"])
    value = _pick(media, "video_url", "playback_url", "video", default="")
    if isinstance(value, dict):
        value = _pick(value, "url", "src", default="")
    return str(value)


def _normalise_media(value) -> list[dict]:
    if not value:
        return []
    if isinstance(value, dict):
        value = value.get("items", value.get("media", [value]))
    if not isinstance(value, list):
        return []
    result = []
    for entry in value:
        if isinstance(entry, str):
            result.append({"type": "image", "url": entry, "alt_text": ""})
            continue
        if not isinstance(entry, dict):
            continue
        kind = str(_pick(entry, "type", "media_type", "kind", default="image")).lower()
        video = "video" in kind or "animated_gif" in kind or "gif" in kind
        url = _select_video_url(entry) if video else str(_pick(
            entry, "media_url_https", "media_url", "url", "image_url"
        ))
        preview = str(_pick(
            entry, "preview_image_url", "thumbnail_url", "media_url_https", "media_url", default=""
        ))
        sizes = entry.get("sizes")
        if not preview and isinstance(sizes, dict):
            preview = str(_pick(sizes.get("medium", {}), "url", default=""))
        if url or preview:
            result.append({
                "type": "video" if video else ("gif" if "gif" in kind else "image"),
                "url": url,
                "preview_url": preview or ("" if video else url),
                "alt_text": str(_pick(entry, "alt_text", "alt", "description")),
                "duration_ms": _pick(entry, "duration_ms", default=None),
            })
    return result


def normalise_tweet(value, *, backend: str) -> dict | None:
    """Convert a backend record to Vakula fields while preserving its body."""
    if not isinstance(value, dict):
        return None
    user = value.get("user") or value.get("author") or {}
    if isinstance(user, str):
        author = user
        handle = user.lstrip("@")
    elif isinstance(user, dict):
        handle = str(_pick(user, "screenName", "username", "handle", "screen_name"))
        author = "@" + handle.lstrip("@") if handle else str(_pick(user, "name", "display_name"))
    else:
        author, handle = "", ""
    body = _pick(value, "full_text", "fullText", "text", "content", "body", "note_text")
    if not isinstance(body, str):
        body = str(body or "")
    identifier = str(_pick(value, "rest_id", "id_str", "id", "tweet_id", "post_id"))
    url = str(_pick(value, "url", "permalink", "source_url"))
    if not url and identifier and handle:
        url = f"https://x.com/{handle.lstrip('@')}/status/{identifier}"
    created = str(_pick(value, "created_at", "createdAt", "published_at", "timestamp"))
    media = _normalise_media(_pick(value, "media", "attachments", "extended_entities", default=[]))
    key = identifier or url or f"{author}\n{created}\n{body}"
    return {
        "source_key": key,
        "platform": "x",
        "author": author,
        "body": body,
        "source_url": url,
        "published_at": created,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "media": media,
        "raw": value,
    }


def _records(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("tweets", "posts", "results", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _records(value)
            if nested:
                return nested
    nested_records = []
    for value in payload.values():
        if isinstance(value, (dict, list)):
            found = _records(value)
            if found and found != [value]:
                nested_records.extend(found)
    if nested_records:
        return nested_records
    return [payload]


def _backend_name(status: ChannelStatus) -> str:
    backend = status.backend.lower()
    if "opencli" in backend:
        return "opencli"
    if "twitter" in backend or "bird" in backend:
        return "twitter"
    return ""


def _run_twitter(query: str, limit: int, env: dict) -> tuple[str, str]:
    output = _run(["twitter", "search", query, "-n", str(limit), "--json"], timeout=60, env=env)
    return output, "twitter-cli"


def _run_opencli(query: str, limit: int) -> tuple[str, str]:
    output = _run(["opencli", "twitter", "search", query, "-f", "yaml"], timeout=60)
    return output, "OpenCLI"


def search_twitter(query: str, limit: int = 10, *, emit: Callable[[str], None] | None = None) -> list[dict]:
    """Search X through the backend Agent Reach currently reports for it."""
    query = query.strip()
    if not query:
        raise AgentReachError("Enter an X search query.")
    limit = max(1, min(int(limit), MAX_RESULTS))
    status = twitter_status()
    backend = _backend_name(status)
    if emit:
        route = status.backend or "null; checking documented read-only backends"
        emit(f"Agent Reach · X active_backend={route}")
        emit(f"Searching X for: {query}")

    child_env = os.environ.copy()
    if backend == "twitter":
        if not shutil.which("twitter"):
            raise AgentReachError("Agent Reach selected twitter-cli, but `twitter` is missing from PATH.")
        if not child_env.get("TWITTER_AUTH_TOKEN") or not child_env.get("TWITTER_CT0"):
            if shutil.which("opencli"):
                output, backend_label = _run_opencli(query, limit)
            else:
                raise AgentReachError(
                    "twitter-cli needs TWITTER_AUTH_TOKEN and TWITTER_CT0 in Vakula's environment. "
                    "Agent Reach's saved values are used for doctor checks only."
                )
        else:
            try:
                output, backend_label = _run_twitter(query, limit, child_env)
            except AgentReachError:
                # Agent Reach's documented X search path recommends one retry;
                # it also documents OpenCLI as a desktop fallback.
                try:
                    output, backend_label = _run_twitter(query, limit, child_env)
                except AgentReachError:
                    if not shutil.which("opencli"):
                        raise
                    output, backend_label = _run_opencli(query, limit)
    elif backend == "opencli":
        if not shutil.which("opencli"):
            raise AgentReachError("Agent Reach selected OpenCLI, but `opencli` is missing from PATH.")
        output, backend_label = _run_opencli(query, limit)
    elif not status.backend:
        # Doctor intentionally leaves some logged-in channels unprobed. Verify
        # only with the documented, read-only command when the user searches.
        if shutil.which("twitter") and child_env.get("TWITTER_AUTH_TOKEN") and child_env.get("TWITTER_CT0"):
            output, backend_label = _run_twitter(query, limit, child_env)
        elif shutil.which("opencli"):
            output, backend_label = _run_opencli(query, limit)
        else:
            message = status.message.strip()
            detail = f" {message}" if message else ""
            raise AgentReachError(
                "Agent Reach has no verified X backend. Configure Twitter cookies or an existing OpenCLI "
                f"browser session, then refresh Sources.{detail}"
            )
    else:
        raise AgentReachError(f"Unsupported Agent Reach X backend: {status.backend}")

    if emit:
        emit(f"Agent Reach · X via {backend_label}")
    payload = _decode_structured(output)
    items = []
    for record in _records(payload)[:limit]:
        item = normalise_tweet(record, backend=backend_label)
        if item and (item["body"] or item["media"] or item["source_url"]):
            insert_observation(item)
            items.append(item)
    if not items:
        raise AgentReachError("The X backend returned no structured posts. Check its output and Agent Reach version.")
    return items


def _platform_status(entries: list[ChannelStatus], platform: str) -> ChannelStatus:
    aliases = {
        "reddit": {"reddit"}, "github": {"github"}, "youtube": {"youtube"},
    }
    status = next((entry for entry in entries if entry.platform in aliases.get(platform, {platform})), None)
    if status is None:
        raise AgentReachError(f"Agent Reach did not report a {platform} channel.")
    return status


def _normalise_record(value: dict, *, platform: str, backend: str) -> dict:
    author_value = value.get("user") or value.get("author") or value.get("owner") or value.get("channel") or {}
    if isinstance(author_value, dict):
        author = str(_pick(author_value, "screenName", "username", "login", "name", "title"))
        if platform == "x" and author and not author.startswith("@"):
            author = "@" + author
    else:
        author = str(author_value)
    body = _pick(value, "full_text", "fullText", "selftext", "text", "content", "body", "description", "title")
    if not isinstance(body, str):
        body = str(body or "")
    identifier = str(_pick(value, "rest_id", "id_str", "id", "tweet_id", "post_id", "nameWithOwner"))
    url = str(_pick(value, "url", "permalink", "source_url", "webpage_url", "html_url"))
    if platform == "reddit" and url.startswith("/"):
        url = "https://www.reddit.com" + url
    if platform == "youtube" and not url:
        video_id = _pick(value, "id", "display_id")
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
    created = str(_pick(value, "created_at", "createdAt", "published_at", "timestamp", "upload_date"))
    source_key = identifier or url or f"{author}\n{created}\n{body}"
    return {
        "source_key": source_key,
        "platform": platform,
        "author": author,
        "body": body,
        "source_url": url,
        "published_at": created,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "media": _normalise_media(_pick(value, "media", "attachments", "extended_entities", default=[]))
        if platform == "x" else [],
        "raw": value,
    }


def _run_rdt(query: str, limit: int) -> tuple[str, str]:
    return _run(["rdt", "search", query, "--limit", str(limit)], timeout=60), "rdt-cli"


def search_platform(platform: str, query: str, limit: int = 10,
                    *, emit: Callable[[str], None] | None = None) -> list[dict]:
    """Search a supported read-only Agent Reach channel using documented CLIs."""
    platform = platform.lower().strip()
    if platform in {"x", "twitter"}:
        return search_twitter(query, limit, emit=emit)
    if platform not in {"reddit", "github", "youtube"}:
        raise AgentReachError(f"Vakula does not yet have a search adapter for {platform!r}.")
    query = query.strip()
    if not query:
        raise AgentReachError("Enter a search query.")
    limit = max(1, min(int(limit), MAX_RESULTS))
    entries = doctor()
    status = _platform_status(entries, platform)
    active = status.backend.lower()
    if emit:
        emit(f"Agent Reach · {platform.upper()} active_backend={status.backend or 'null'}")

    if platform == "github":
        if not active or not ("gh" in active or "github" in active):
            detail = f" {status.message.strip()}" if status.message else ""
            raise AgentReachError(f"Agent Reach has no verified GitHub backend.{detail}")
        if not shutil.which("gh"):
            raise AgentReachError("Agent Reach uses GitHub CLI for this channel, but `gh` is missing from PATH.")
        output = _run([
            "gh", "search", "repos", query, "--sort", "stars", "--limit", str(limit),
            "--json", "nameWithOwner,description,url,stargazersCount,updatedAt,language,owner",
        ], timeout=60)
        backend_label = "gh CLI"
    elif platform == "youtube":
        if not active:
            detail = f" {status.message.strip()}" if status.message else ""
            raise AgentReachError(f"Agent Reach has no verified YouTube backend.{detail}")
        if "yt-dlp" not in active and "youtube" not in active:
            raise AgentReachError(f"Agent Reach selected an unsupported YouTube backend: {status.backend}")
        if not shutil.which("yt-dlp"):
            raise AgentReachError("Agent Reach needs `yt-dlp` for YouTube search, but it is missing from PATH.")
        output = _run([
            "yt-dlp", "--dump-json", "--flat-playlist", "--playlist-end", str(min(limit, 10)),
            f"ytsearch{min(limit, 10)}:{query}",
        ], timeout=90)
        backend_label = "yt-dlp"
    else:
        if "opencli" in active:
            if not shutil.which("opencli"):
                raise AgentReachError("Agent Reach selected OpenCLI for Reddit, but `opencli` is missing from PATH.")
            output = _run(["opencli", "reddit", "search", query, "-f", "yaml"], timeout=60)
            backend_label = "OpenCLI"
        elif "rdt" in active:
            if not shutil.which("rdt"):
                raise AgentReachError("Agent Reach selected rdt-cli for Reddit, but `rdt` is missing from PATH.")
            output, backend_label = _run_rdt(query, limit)
        elif not status.backend:
            # Doctor intentionally avoids probing login-backed Reddit sessions.
            if shutil.which("opencli"):
                output = _run(["opencli", "reddit", "search", query, "-f", "yaml"], timeout=60)
                backend_label = "OpenCLI (read-only verification)"
            elif shutil.which("rdt"):
                output, backend_label = _run_rdt(query, limit)
            else:
                raise AgentReachError(
                    "No Reddit backend is ready. Configure an existing OpenCLI browser session or rdt login."
                )
        else:
            raise AgentReachError(f"Agent Reach selected an unsupported Reddit backend: {status.backend}")

    if platform == "youtube":
        records = []
        for line in output.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        payload = records
    else:
        payload = _decode_structured(output)
    items = []
    for record in _records(payload)[:limit]:
        item = _normalise_record(record, platform=platform, backend=backend_label)
        if item["body"] or item["source_url"]:
            insert_observation(item)
            items.append(item)
    if not items:
        raise AgentReachError(f"The {platform} backend returned no structured search results.")
    if emit:
        emit(f"Agent Reach · received {len(items)} {platform} result(s) via {backend_label}")
    return items
