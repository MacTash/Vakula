"""Standards-based web search and readable-page retrieval for Vidur.

Search uses a user-controlled SearXNG JSON endpoint when available. The
keyless fallback uses DuckDuckGo's non-JavaScript HTML search interface.
"""

from __future__ import annotations

import os
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

import requests

from vidur.storage import add_item, setting


def searxng_url() -> str:
    return setting("VIDUR_SEARXNG_URL", "GEOSCOPE_SEARXNG_URL")

USER_AGENT = "Vidur/0.2.0 (terminal research client)"
MAX_PAGE_BYTES = 1_500_000


class BrowserError(RuntimeError):
    pass


def configured_provider() -> str | None:
    if searxng_url():
        return "SearXNG"
    return "DuckDuckGo HTML (keyless fallback)"


def search(query: str, limit: int = 8, progress=None) -> list[dict]:
    """Search the web and store result metadata as WEBINT items.

    ``progress`` receives short source-status messages so the interactive UI
    can show exactly which provider is being consulted.
    """
    query = query.strip()
    if not query:
        raise BrowserError("Enter a search query.")
    limit = max(1, min(limit, 20))
    provider = configured_provider()
    if progress:
        progress(f"Searching {provider} …")
    base = searxng_url().rstrip("/")
    if base:
        try:
            response = requests.get(f"{base}/search", params={"q": query, "format": "json", "categories": "general"},
                                    headers={"User-Agent": USER_AGENT}, timeout=25)
            response.raise_for_status()
            data = response.json()
            results = [{"title": row.get("title", "Untitled"), "url": row.get("url", ""),
                        "snippet": row.get("content", ""), "engine": row.get("engine", "SearXNG")}
                       for row in data.get("results", [])[:limit]]
        except (requests.RequestException, ValueError) as exc:
            raise BrowserError(f"SearXNG search failed: {exc}") from exc
    else:
        try:
            response = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                                     headers={"User-Agent": USER_AGENT}, timeout=25)
            response.raise_for_status()
            parser = _DuckDuckGoResults()
            parser.feed(response.text)
            results = parser.results[:limit]
        except (requests.RequestException, ValueError) as exc:
            raise BrowserError(f"DuckDuckGo HTML search failed: {exc}") from exc
    for result in results:
        add_item({"category": "WEBINT", "title": result["title"], "summary": result["snippet"],
                  "source": result["engine"], "source_url": result["url"], "confidence": 0.6,
                  "tags": ["web-search", query], "raw": result})
    if progress:
        progress(f"Received {len(results)} results from {provider}.")
    return results


class _DuckDuckGoResults(HTMLParser):
    """Extract result links and snippets from DuckDuckGo's no-JS HTML UI."""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._href = ""
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._in_result = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        classes = set(dict(attrs).get("class", "").split())
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            self._in_result, self._href, self._title = True, dict(attrs).get("href", ""), []
        elif "result__snippet" in classes:
            self._in_snippet, self._snippet = True, []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_result:
            href = self._href
            href = parse_qs(urlparse(href).query).get("uddg", [href])[0]
            title = " ".join(self._title).strip()
            if title and href:
                self.results.append({"title": title, "url": href, "snippet": "", "engine": "DuckDuckGo HTML"})
            self._in_result = False
        elif tag in {"a", "div", "span"} and self._in_snippet:
            text = " ".join(self._snippet).strip()
            if text and self.results:
                self.results[-1]["snippet"] = text
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_result:
            self._title.append(data)
        if self._in_snippet:
            self._snippet.append(data)


class _TextExtractor(HTMLParser):
    BLOCKS = {"p", "br", "div", "li", "article", "section", "h1", "h2", "h3", "title"}
    IGNORE = {"script", "style", "noscript", "svg"}

    def __init__(self):
        super().__init__(); self.parts: list[str] = []; self._ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORE: self._ignored += 1
        if tag in self.BLOCKS: self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.IGNORE and self._ignored: self._ignored -= 1
        if tag in self.BLOCKS: self.parts.append("\n")

    def handle_data(self, data):
        if not self._ignored: self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def read(url: str, progress=None) -> dict:
    """Retrieve a public HTTP(S) page and return a bounded readable extract."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BrowserError("Only absolute http:// or https:// URLs can be opened.")
    if progress: progress(f"Opening {parsed.netloc} …")
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                                timeout=25, stream=True)
        response.raise_for_status()
        mime = response.headers.get("content-type", "")
        if "html" not in mime and "text" not in mime:
            raise BrowserError(f"Unsupported content type: {mime or 'unknown'}")
        data = b"".join(chunk for chunk in response.iter_content(64 * 1024) if chunk)[:MAX_PAGE_BYTES]
    except requests.RequestException as exc:
        raise BrowserError(f"Could not open page: {exc}") from exc
    parser = _TextExtractor(); parser.feed(data.decode(response.encoding or "utf-8", errors="replace"))
    text = parser.text()
    result = {"url": response.url, "title": parsed.netloc, "text": text[:12_000], "source": parsed.netloc}
    add_item({"category": "WEBINT", "title": f"Page — {parsed.netloc}", "summary": result["text"][:1000],
              "source": parsed.netloc, "source_url": response.url, "confidence": 0.7,
              "tags": ["web-page"], "raw": {"content_type": mime}})
    if progress: progress(f"Read {len(text):,} characters from {parsed.netloc}.")
    return result
