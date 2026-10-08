from vidur import browser


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"results": [{"title": "Example", "url": "https://example.test", "content": "A result", "engine": "test"}]}


def test_searxng_search_uses_documented_json_api(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VIDUR_SEARXNG_URL", "https://search.example.test")
    monkeypatch.setattr(browser.requests, "get", lambda *args, **kwargs: FakeResponse())
    result = browser.search("test")
    assert result == [{"title": "Example", "url": "https://example.test", "snippet": "A result", "engine": "test"}]


def test_duckduckgo_html_fallback_needs_no_key(monkeypatch, tmp_path):
    class HtmlResponse:
        text = '<a class="result__a" href="https://example.test">Example</a><div class="result__snippet">A free result</div>'
        def raise_for_status(self):
            return None
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("VIDUR_SEARXNG_URL", raising=False)
    monkeypatch.setattr(browser.requests, "post", lambda *args, **kwargs: HtmlResponse())
    assert browser.search("test")[0]["engine"] == "DuckDuckGo HTML"
