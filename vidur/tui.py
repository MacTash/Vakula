"""Responsive Textual terminal workspace for source-first research."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import textwrap
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, RichLog, Select, Static, TabPane, TabbedContent


class _ImageProbeTimeoutFilter(logging.Filter):
    """Hide the image library's traceback for an expected unsupported-terminal timeout."""

    def filter(self, record: logging.LogRecord) -> bool:
        if (record.name == "textual_image._terminal"
                and record.getMessage() == "Failed to probe terminal capabilities"
                and record.exc_info
                and isinstance(record.exc_info[1], TimeoutError)):
            return False
        return True


logging.getLogger("textual_image._terminal").addFilter(_ImageProbeTimeoutFilter())

# textual-image divides pixel dimensions by terminal rows/columns before its
# capability-query fallback. Some PTYs (for example, ones launched without a
# configured window size) report zero rows or columns, so normalize those values
# and let the library use its own default cell size instead of crashing import.
from textual_image import _terminal as _image_terminal

_get_image_terminal_size = _image_terminal.get_tiocgwinsz


def _safe_image_terminal_size() -> tuple[int, int, int, int]:
    rows, columns, screen_width, screen_height = _get_image_terminal_size()
    return max(rows, 1), max(columns, 1), screen_width, screen_height


_image_terminal.get_tiocgwinsz = _safe_image_terminal_size

from textual_image.widget import Image as TerminalImage

from vidur.agent import AISettings, ModelDiscoveryError, QWEN_TINY_MODEL, list_ollama_models, research
from vidur.agent_reach import AgentReachError, ChannelStatus, doctor, search_platform
from vidur.browser import BrowserError, configured_provider, read, search
from vidur.collectors import earthquakes, weather
from vidur.media import MediaError, download_media, video_frames, video_poster
from vidur.storage import init_db, list_items, list_source_items, set_source_saved, stats


def _post_from_worker(app: App, callback, *args) -> None:
    """Queue a UI update without blocking the worker during app shutdown."""
    loop = app._loop
    if loop is None or not app.is_running:
        return
    try:
        loop.call_soon_threadsafe(callback, *args)
    except RuntimeError:
        # The event loop may close between the checks above and scheduling.
        pass


class ConfirmModelDownload(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(
                "Download Qwen3 0.6B Q4_K_M?\n\n"
                "Ollama will download about 523 MB. Vidur will not download it until you confirm.",
                id="confirm-copy",
                markup=False,
            )
            with Horizontal(classes="dialog-actions"):
                yield Button("Download model", id="confirm-download", variant="primary")
                yield Button("Cancel", id="cancel-download")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-download")

    def action_cancel(self) -> None:
        self.dismiss(False)


class VidurApp(App[None]):
    TITLE = "Vidur"
    SUB_TITLE = "Source-first terminal research"
    CSS_PATH = "vidur.tcss"
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+p", "command_palette", "Commands"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.settings = AISettings(mode="local")
        self.source_status: list[ChannelStatus] = []
        self.source_error = ""
        self.items: list[dict] = []
        self.selected_item: dict | None = None
        self.local_models: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Starting Vidur…", id="connection-line", markup=False)
        with TabbedContent(initial="overview-tab", id="main-tabs"):
            with TabPane("Overview", id="overview-tab"):
                yield Static("Loading local situation…", id="overview-status", markup=False)
                with Horizontal(classes="search-row"):
                    yield Input(placeholder="Weather location", id="weather-location")
                    yield Button("Fetch weather", id="fetch-weather")
                    yield Button("Refresh earthquakes", id="fetch-earthquakes")
                yield DataTable(id="intel-table", zebra_stripes=True, cursor_type="row")
                yield Static("Select an item to inspect its summary and source link.", id="intel-detail", markup=False)
            with TabPane("Live Feed", id="feed-tab"):
                with Horizontal(classes="search-row"):
                    yield Select(
                        [("X / Twitter", "x"), ("Reddit", "reddit"), ("GitHub repos", "github"), ("YouTube", "youtube")],
                        value="x", allow_blank=False, id="source-platform",
                    )
                    yield Input(placeholder="Search sources via Agent Reach", id="x-query")
                    yield Button("Search", id="source-search", variant="primary")
                    yield Button("Refresh sources", id="sources-refresh")
                yield DataTable(id="feed-table", zebra_stripes=True, cursor_type="row")
                yield Static("Select a post to inspect its original text, source, and attachments.",
                             id="detail", markup=False)
                with Horizontal(classes="action-row"):
                    yield Button("Save / unsave", id="save-item")
                    yield Button("View media", id="view-media")
            with TabPane("Research", id="research-tab"):
                with Horizontal(classes="search-row"):
                    yield Input(placeholder="Ask a basic question or type /help", id="research-query")
                    yield Button("Ask Vidur", id="ask-vidur", variant="primary")
                yield RichLog(id="activity", wrap=True, markup=False, highlight=False, auto_scroll=True)
                yield Static("Qwen answers appear here. Retrieved source posts remain in the Live Feed.",
                             id="answer", markup=False)
            with TabPane("Saved", id="saved-tab"):
                yield DataTable(id="saved-table", zebra_stripes=True, cursor_type="row")
                yield Static("Select a saved post to inspect it in the Live Feed.",
                             id="saved-detail", markup=False)
            with TabPane("Sources", id="sources-tab"):
                yield VerticalScroll(Static("Checking Agent Reach…", id="source-status", markup=False),
                                    id="source-scroll")
            with TabPane("Settings", id="settings-tab"):
                yield Static("Loading local model status…", id="settings-status", markup=False)
                with Horizontal(classes="action-row"):
                    yield Button("Download Qwen3 0.6B (523 MB)", id="download-qwen")
                    yield Button("Refresh model list", id="refresh-models")
                yield Static(
                    "Provider settings remain session-only. In Research, use /mode provider, "
                    "/provider base <url>, /provider model <name>, and /provider key <key>.",
                    id="settings-help", markup=False,
                )
        yield Footer()

    def on_mount(self) -> None:
        init_db()
        for table_id in ("feed-table", "saved-table"):
            table = self.query_one(f"#{table_id}", DataTable)
            table.add_columns("WHEN", "SOURCE", "AUTHOR", "POST")
        self.query_one("#intel-table", DataTable).add_columns("WHEN", "CATEGORY", "SEVERITY", "ITEM")
        self.refresh_overview()
        self.refresh_feed()
        self.refresh_saved()
        self.check_sources_worker()
        self.discover_models_worker()
        self.query_one("#activity", RichLog).write(
            f"READY · Search: {configured_provider()} · source posts are shown as returned by their backend."
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "source-search":
            self.submit_source_search()
        elif button == "sources-refresh":
            self.check_sources_worker()
        elif button == "save-item":
            self.toggle_save()
        elif button == "view-media":
            self.open_media()
        elif button == "ask-vidur":
            self.ask_vidur()
        elif button == "download-qwen":
            self.push_screen(ConfirmModelDownload(), self.confirm_qwen_download)
        elif button == "refresh-models":
            self.discover_models_worker()
        elif button == "fetch-weather":
            self.fetch_weather()
        elif button == "fetch-earthquakes":
            self.fetch_earthquakes_worker()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "x-query":
            self.submit_source_search(event.value)
        elif event.input.id == "research-query":
            self.ask_vidur(event.value)
        elif event.input.id == "weather-location":
            self.fetch_weather()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            item_id = int(str(event.row_key.value))
        except ValueError:
            return
        if event.data_table.id == "intel-table":
            item = next((row for row in list_items(limit=500) if int(row["id"]) == item_id), None)
            if item:
                self.query_one("#intel-detail", Static).update(
                    f"{item.get('category', '')} · {item.get('severity', '').upper()}\n"
                    f"{item.get('title', '')}\n\n{item.get('summary', '')}\n\n"
                    f"Source: {item.get('source', '')}\nURL: {item.get('source_url', '')}"
                )
            return
        if event.data_table.id == "saved-table":
            item = next((row for row in list_source_items(saved=True) if int(row["id"]) == item_id), None)
            if item:
                self.selected_item = item
                self.query_one("#saved-detail", Static).update(self.detail_text(item))
            return
        item = next((row for row in self.items if int(row["id"]) == item_id), None)
        if item:
            self.selected_item = item
            self.show_detail(item)

    def refresh_feed(self, query: str | None = None) -> None:
        self.items = list_source_items(query, limit=150)
        table = self.query_one("#feed-table", DataTable)
        table.clear(columns=False)
        for item in self.items:
            published = item.get("published_at") or item.get("fetched_at", "")
            published = published[:16].replace("T", " ")
            excerpt = textwrap.shorten(item.get("body", "").replace("\n", " "), width=100, placeholder="…")
            if item.get("media"):
                excerpt = f"{excerpt}  [{len(item['media'])} attachment(s)]".strip()
            table.add_row(published, item.get("platform", "").upper(), item.get("author", ""), excerpt,
                          key=str(item["id"]))
        self.query_one("#connection-line", Static).update(self.connection_text())

    def refresh_overview(self) -> None:
        summary = stats()
        rows = list_items(limit=100)
        table = self.query_one("#intel-table", DataTable)
        table.clear(columns=False)
        for item in rows:
            when = item.get("collected_at", "")[:16].replace("T", " ")
            title = textwrap.shorten(item.get("title", ""), width=88, placeholder="…")
            table.add_row(when, item.get("category", ""), item.get("severity", "").upper(), title,
                          key=str(item["id"]))
        # Structured measurements only. Any briefing prose lives in Research.
        self.query_one("#overview-status", Static).update(
            f"LOCAL SITUATION  ·  {summary['total']} intelligence items  ·  "
            f"{summary['source_items']} source posts  ·  latest collection {summary['latest'] or 'none'}\n"
            f"{self._intelligence_line()}\n"
            f"Database: {summary['database']}"
        )

    def _intelligence_line(self) -> str:
        """One line of measured intelligence state for the Overview tab."""
        try:
            from vidur import analytics
            built = analytics.analyse()
            observations = int(built.metrics.get("window_observations") or 0)
            return (f"INTELLIGENCE  ·  {observations} observation(s) in "
                    f"{built.window_days}d  ·  activity {built.status}  ·  "
                    f"type /assess <scope> in Research for an assessment")
        except Exception:
            return "INTELLIGENCE  ·  not yet available"

    def fetch_weather(self) -> None:
        location = self.query_one("#weather-location", Input).value.strip()
        if not location:
            self.notify("Enter a place name for weather.", severity="warning")
            return
        self.fetch_weather_worker(location)

    @work(thread=True, exclusive=True, group="collect-weather")
    def fetch_weather_worker(self, location: str) -> None:
        try:
            items = weather(location)
        except Exception as exc:
            _post_from_worker(self, self._show_error, f"Weather collection failed: {exc}")
            return
        _post_from_worker(self, self._collector_complete, f"Collected current weather for {location}.", len(items))

    @work(thread=True, exclusive=True, group="collect-earthquakes")
    def fetch_earthquakes_worker(self) -> None:
        try:
            items = earthquakes(days=7)
        except Exception as exc:
            _post_from_worker(self, self._show_error, f"Earthquake collection failed: {exc}")
            return
        _post_from_worker(self, self._collector_complete, "Refreshed USGS earthquakes from the last 7 days.", len(items))

    def _collector_complete(self, message: str, count: int) -> None:
        self.log_activity(f"COLLECT · {message} · {count} result(s)")
        self.refresh_overview()

    def refresh_saved(self) -> None:
        saved = list_source_items(saved=True, limit=150)
        table = self.query_one("#saved-table", DataTable)
        table.clear(columns=False)
        for item in saved:
            when = (item.get("published_at") or item.get("fetched_at", ""))[:16].replace("T", " ")
            excerpt = textwrap.shorten(item.get("body", "").replace("\n", " "), width=100, placeholder="…")
            table.add_row(when, item.get("platform", "").upper(), item.get("author", ""), excerpt,
                          key=str(item["id"]))

    def connection_text(self) -> str:
        reach = "Agent Reach: checking…" if not self.source_status and not self.source_error else "Agent Reach: "
        if self.source_error:
            reach += self.source_error
        elif self.source_status:
            x = next((entry for entry in self.source_status if entry.platform in {"twitter", "x", "twitter/x"}), None)
            reach += f"X via {x.backend}" if x and x.backend else "X needs setup"
        if self.settings.mode == "provider":
            model_state = f"Provider model: {self.settings.model or 'not configured'}"
        else:
            model_state = f"Local model: {self.settings.model or 'Qwen3 0.6B not installed'}"
        return f"VIDUR  ·  {reach}  ·  {model_state}"

    @work(thread=True, exclusive=True, group="agent-reach")
    def check_sources_worker(self) -> None:
        try:
            statuses = doctor()
        except AgentReachError as exc:
            _post_from_worker(self, self._set_source_error, str(exc))
            return
        _post_from_worker(self, self._set_source_status, statuses)

    def _set_source_error(self, message: str) -> None:
        self.source_error = message
        self.source_status = []
        self._render_source_status()

    def _set_source_status(self, statuses: list[ChannelStatus]) -> None:
        self.source_error = ""
        self.source_status = statuses
        self._render_source_status()

    def _render_source_status(self) -> None:
        widget = self.query_one("#source-status", Static)
        if self.source_error:
            widget.update("AGENT REACH\n\n" + self.source_error)
        else:
            lines = ["AGENT REACH · read-only channel health", ""]
            for entry in sorted(self.source_status, key=lambda row: row.platform):
                backend = entry.backend or "backend not verified"
                lines.append(f"{entry.platform.upper():14} {entry.status:10} {backend}")
                if entry.message:
                    lines.append(f"  {entry.message}")
            lines.extend(["", "Vidur invokes documented, read-only upstream commands only.",
                          "Agent Reach and login-backed sources are configured separately by you."])
            widget.update("\n".join(lines))
        self.query_one("#connection-line", Static).update(self.connection_text())

    def submit_source_search(self, query: str | None = None) -> None:
        query = (query if query is not None else self.query_one("#x-query", Input).value).strip()
        if not query:
            self.notify("Enter a source search query.", severity="warning")
            return
        self.query_one("#x-query", Input).value = query
        platform = str(self.query_one("#source-platform", Select).value)
        self.notify(f"Using Agent Reach · {platform.upper()} via its selected backend")
        self.search_source_worker(platform, query)

    @work(thread=True, exclusive=True, group="source-search")
    def search_source_worker(self, platform: str, query: str) -> None:
        try:
            items = search_platform(
                platform, query, emit=lambda message: _post_from_worker(self, self.log_activity, message)
            )
        except (AgentReachError, ValueError) as exc:
            _post_from_worker(self, self._show_error, f"{platform.upper()} search failed: {exc}")
            return
        _post_from_worker(self, self._search_complete, len(items))

    def _search_complete(self, count: int) -> None:
        self.refresh_feed()
        self.refresh_saved()
        self.notify(f"Received {count} post(s). Open the Live Feed to view source text and media.")
        if self.items:
            self.selected_item = self.items[0]
            self.show_detail(self.items[0])

    def detail_text(self, item: dict) -> str:
        media = item.get("media", [])
        lines = [
            f"{item.get('platform', '').upper()} · {item.get('author', 'unknown author')}",
            f"Published: {item.get('published_at') or 'time not provided'}    Fetched: {item.get('fetched_at', '')}",
            f"Backend: {item.get('backend') or 'unknown'}",
            f"URL: {item.get('source_url') or 'not provided'}",
            "─" * 60,
            item.get("body", "[This source returned no text.]"),
        ]
        if media:
            lines.extend(["", f"Attachments: {len(media)} · select View media to open them in the TUI"])
            for index, attachment in enumerate(media, 1):
                lines.append(f"  {index}. {attachment.get('type', 'media')} · {attachment.get('alt_text') or 'no alt text'}")
        raw = item.get("raw", {})
        metrics = []
        for key, label in (("replyCount", "replies"), ("retweetCount", "reposts"),
                           ("quoteCount", "quotes"), ("likeCount", "likes"), ("viewCount", "views")):
            if isinstance(raw, dict) and raw.get(key) is not None:
                metrics.append(f"{label} {raw[key]}")
        if metrics:
            lines.append("Engagement: " + " · ".join(metrics))
        if item.get("saved"):
            lines.append("\nSaved locally")
        return "\n".join(lines)

    def show_detail(self, item: dict) -> None:
        self.query_one("#detail", Static).update(self.detail_text(item))
        if item.get("saved"):
            self.query_one("#save-item", Button).label = "Unsave"
        else:
            self.query_one("#save-item", Button).label = "Save"

    def toggle_save(self) -> None:
        if not self.selected_item:
            self.notify("Select a post first.", severity="warning")
            return
        item = self.selected_item
        saved = not bool(item.get("saved"))
        set_source_saved(int(item["id"]), saved)
        item["saved"] = saved
        self.show_detail(item)
        self.refresh_saved()

    def open_media(self) -> None:
        if not self.selected_item:
            self.notify("Select a post first.", severity="warning")
            return
        attachments = self.selected_item.get("media", [])
        if not attachments:
            self.notify("This post has no media attachments.", severity="warning")
            return
        self.push_screen(MediaScreen(attachments, self.selected_item.get("source_url", "")))

    def ask_vidur(self, prompt: str | None = None) -> None:
        query = (prompt if prompt is not None else self.query_one("#research-query", Input).value).strip()
        if not query:
            return
        self.query_one("#research-query", Input).value = ""
        if query.startswith("/"):
            self.handle_command(query)
            return
        self.query_one("#activity", RichLog).write(f"YOU · {query}")
        if not self.settings.model:
            self.query_one("#answer", Static).update(
                "No local model selected. Install Ollama and choose Download Qwen3 0.6B in Settings, "
                "or use /mode provider for an OpenAI-compatible server."
            )
            return
        self.research_worker(query)

    @work(thread=True, exclusive=True, group="research")
    def research_worker(self, query: str) -> None:
        try:
            result = research(query, self.settings, lambda message: _post_from_worker(self, self.log_activity, message))
        except Exception as exc:
            result = f"Research failed: {exc}"
        _post_from_worker(self, self._research_complete, result)

    def _research_complete(self, result: str) -> None:
        self.query_one("#answer", Static).update(result)
        self.refresh_feed()
        self.refresh_saved()

    def log_activity(self, message: str) -> None:
        self.query_one("#activity", RichLog).write(message)

    def _show_error(self, message: str) -> None:
        self.log_activity("ERROR · " + message)
        self.notify(message[:180], severity="error")

    @work(thread=True, exclusive=True, group="ollama")
    def discover_models_worker(self) -> None:
        try:
            models = list_ollama_models(self.settings.ollama_url)
        except ModelDiscoveryError as exc:
            _post_from_worker(self, self._set_models, [], str(exc))
            return
        _post_from_worker(self, self._set_models, models, "")

    def _set_models(self, models: list[dict], error: str) -> None:
        self.local_models = models
        names = {item["name"] for item in models}
        if self.settings.mode == "local":
            if QWEN_TINY_MODEL in names:
                self.settings.model = QWEN_TINY_MODEL
            elif self.settings.model not in names:
                self.settings.model = ""
        if error:
            text = f"Ollama is unavailable: {error}\n\nInstall Ollama to use the local helper."
        elif not models:
            text = "Ollama is running, but no models are installed.\n\nDownload Qwen3 0.6B from this screen when ready."
        else:
            text = "Installed local models:\n" + "\n".join(
                f"  {entry['name']} · {entry.get('parameters') or entry.get('family') or 'local model'}"
                for entry in models
            )
            if self.settings.model:
                text += f"\n\nSelected helper: {self.settings.model}"
        self.query_one("#settings-status", Static).update(text)
        self.query_one("#connection-line", Static).update(self.connection_text())

    def confirm_qwen_download(self, confirmed: bool) -> None:
        if not confirmed:
            return
        if not shutil.which("ollama"):
            self.notify("Install Ollama first, then try the download again.", severity="warning")
            return
        self.pull_qwen_worker()

    @work(thread=True, exclusive=True, group="ollama-pull")
    def pull_qwen_worker(self) -> None:
        try:
            process = subprocess.Popen(
                ["ollama", "pull", QWEN_TINY_MODEL],
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            _post_from_worker(self, self._show_error, f"Could not start Ollama: {exc}")
            return
        _post_from_worker(self, self.log_activity, f"MODEL · downloading {QWEN_TINY_MODEL}")
        if process.stdout:
            for line in process.stdout:
                clean = line.strip()
                if clean:
                    _post_from_worker(self, self.log_activity, "OLLAMA · " + clean[:220])
        code = process.wait()
        if code:
            _post_from_worker(self, self._show_error, f"Ollama model download exited with status {code}.")
        else:
            _post_from_worker(self, self.log_activity, "MODEL · download complete")
            _post_from_worker(self, self.discover_models_worker)

    @work(thread=True, exclusive=True, group="assess")
    def assess_worker(self, scope: str) -> None:
        """Generate an assessment off the UI thread and hand back the text.

        Everything factual in the briefing is computed by the intelligence
        pipeline; the model only writes the prose section. A missing model yields
        the deterministic rendering instead of an error.
        """
        from vidur import briefing
        from vidur.intelligence_model import NullModel, model_from_settings
        model = NullModel() if not (self.settings.enabled and self.settings.model) \
            else model_from_settings(self.settings)
        try:
            result = briefing.assess(scope, model=model)
        except Exception as exc:
            _post_from_worker(self, self._show_error, f"Assessment failed: {exc}")
            return
        _post_from_worker(self, self._assessment_ready, result)

    def _assessment_ready(self, result) -> None:
        """Structured intelligence and generated prose are shown separately."""
        counts = result.to_dict()
        self.log_activity(
            f"INTEL · {result.scope} · {result.data_quality} · "
            f"{len(counts['claims'])} classified claim(s) · "
            f"prose: {result.model_name if result.model_used else 'deterministic only'}"
        )
        self.query_one("#answer", Static).update(result.render())

    def handle_command(self, command: str) -> None:
        name, _, value = command.strip().partition(" ")
        if name in {"/help", "/"}:
            self.log_activity("COMMANDS · /x <query> · /sources · /models · /local [model] · /mode local|provider · /provider base|model|key · /search <query> · /open <url> · /status · /assess <scope>")
        elif name == "/x" and value:
            self.query_one("#source-platform", Select).value = "x"
            self.submit_source_search(value)
        elif name == "/sources":
            self.check_sources_worker()
            self.log_activity("SOURCES · refreshing Agent Reach doctor status")
        elif name == "/models":
            self.discover_models_worker()
            self.log_activity("MODELS · refreshing local Ollama model list")
        elif name == "/local":
            candidate = value.strip()
            names = {item["name"] for item in self.local_models}
            if candidate and candidate in names:
                self.settings.mode = "local"
                self.settings.model = candidate
                self.log_activity(f"AI · selected local model {candidate}")
            elif not candidate and QWEN_TINY_MODEL in names:
                self.settings.mode = "local"
                self.settings.model = QWEN_TINY_MODEL
                self.log_activity(f"AI · selected {QWEN_TINY_MODEL}")
            else:
                self.log_activity("AI · choose an installed exact model name, or download Qwen from Settings")
            self.query_one("#connection-line", Static).update(self.connection_text())
        elif name == "/mode" and value.strip() in {"local", "provider"}:
            new_mode = value.strip()
            if self.settings.mode != new_mode:
                self.settings.model = ""
            self.settings.mode = new_mode
            self.log_activity(f"AI · mode set to {self.settings.mode}")
        elif name == "/provider":
            self._configure_provider(value)
        elif name == "/search" and value:
            self.web_search_worker(value)
        elif name == "/open" and value:
            self.open_url_worker(value)
        elif name == "/status":
            data = stats()
            self.log_activity(
                f"STATUS · source posts={data['source_items']} · intelligence items={data['total']} · "
                f"database={data['database']} · model={self.settings.model or 'none'}"
            )
        elif name in {"/assess", "/brief"} and value:
            self.assess_worker(value)
        else:
            self.log_activity("ERROR · unknown or incomplete command; use /help")

    def _configure_provider(self, value: str) -> None:
        option, _, setting = value.partition(" ")
        if self.settings.mode != "provider":
            self.settings.model = ""
        self.settings.mode = "provider"
        if option == "base" and setting:
            self.settings.base_url = setting.rstrip("/")
            self.log_activity("PROVIDER · base URL set for this session")
        elif option == "model" and setting:
            self.settings.model = setting
            self.log_activity("PROVIDER · model set for this session")
        elif option == "key" and setting:
            self.settings.provider_api_key = setting
            self.log_activity("PROVIDER · key held in memory for this session")
        else:
            self.log_activity("PROVIDER · use /provider base <url>, /provider model <name>, or /provider key <key>")

    @work(thread=True, exclusive=True, group="web-search")
    def web_search_worker(self, query: str) -> None:
        try:
            rows = search(query, progress=lambda message: _post_from_worker(self, self.log_activity, message))
            text = "\n\n".join(f"{row['title']}\n{row['url']}\n{row['snippet']}" for row in rows)
        except BrowserError as exc:
            text = f"Web search failed: {exc}"
        _post_from_worker(self, self._research_complete, text)

    @work(thread=True, exclusive=True, group="web-open")
    def open_url_worker(self, url: str) -> None:
        try:
            page = read(url, progress=lambda message: _post_from_worker(self, self.log_activity, message))
            result = f"{page['title']}\n{page['url']}\n\n{page['text']}"
        except BrowserError as exc:
            result = f"Could not open page: {exc}"
        _post_from_worker(self, self._research_complete, result)


class MediaScreen(Screen[None]):
    BINDINGS = [
        Binding("escape", "close", "Back"),
        Binding("q", "close", "Back"),
        Binding("space", "toggle_play", "Play / pause"),
        Binding("left", "previous", "Previous attachment"),
        Binding("right", "next", "Next attachment"),
    ]

    def __init__(self, attachments: list[dict], source_url: str = "") -> None:
        super().__init__()
        self.attachments = attachments
        self.source_url = source_url
        self.index = 0
        self.current_path: Path | None = None
        self.stop_event = threading.Event()
        self.playing_event = threading.Event()
        self.playback_active = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("Loading attachment…", id="media-caption", markup=False)
        yield TerminalImage(id="media-frame")
        yield Static("←/→ switch attachment · Space play/pause video · Esc return", id="media-status", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.load_attachment()

    def on_unmount(self) -> None:
        self.stop_event.set()
        self.playing_event.set()

    def action_close(self) -> None:
        self.stop_event.set()
        self.app.pop_screen()

    def action_previous(self) -> None:
        self.stop_event.set()
        self.index = (self.index - 1) % len(self.attachments)
        self.load_attachment()

    def action_next(self) -> None:
        self.stop_event.set()
        self.index = (self.index + 1) % len(self.attachments)
        self.load_attachment()

    def action_toggle_play(self) -> None:
        attachment = self.attachments[self.index]
        if attachment.get("type") not in {"video", "gif"} or not self.current_path:
            return
        if self.playing_event.is_set():
            self.playing_event.clear()
            self.query_one("#media-status", Static).update("Paused · Space resumes · ←/→ switch attachment")
        else:
            self.playing_event.set()
            self.stop_event.clear()
            self.query_one("#media-status", Static).update("Playing · Space pauses · ←/→ switch attachment")
            if not self.playback_active:
                self.playback_worker(self.current_path, self.stop_event, self.playing_event)

    def load_attachment(self) -> None:
        self.stop_event = threading.Event()
        self.playing_event.clear()
        self.current_path = None
        self.playback_active = False
        self.load_attachment_worker(self.index)

    @work(thread=True, exclusive=True, group="media-load")
    def load_attachment_worker(self, index: int) -> None:
        attachment = self.attachments[index]
        kind = str(attachment.get("type", "image")).lower()
        is_video = kind in {"video", "gif"}
        source = attachment.get("url", "")
        preview = attachment.get("preview_url", "")
        try:
            path = None
            download_error = None
            if source:
                try:
                    path = download_media(source, kind="video" if is_video else "image")
                except MediaError as exc:
                    download_error = exc
            preview_path = None
            if preview and preview != source:
                try:
                    preview_path = download_media(preview, kind="image")
                except MediaError:
                    if not path:
                        raise
            display_path = preview_path or (video_poster(path) if is_video and path else path)
            if display_path is None:
                if download_error:
                    raise download_error
                raise MediaError("This attachment has no viewable media URL from its source backend.")
            _post_from_worker(self.app, self._media_ready, index, display_path, path if is_video else None)
        except Exception as exc:
            _post_from_worker(self.app, self._media_failed, index, str(exc))

    def _media_ready(self, index: int, display_path, video_path: Path | None) -> None:
        if index != self.index:
            return
        self.current_path = video_path
        attachment = self.attachments[index]
        total = len(self.attachments)
        title = f"X attachment {index + 1}/{total} · {attachment.get('type', 'image')}"
        if attachment.get("alt_text"):
            title += f"\nAlt text: {attachment['alt_text']}"
        if self.source_url:
            title += f"\nPost: {self.source_url}"
        self.query_one("#media-caption", Static).update(title)
        self.query_one("#media-frame", TerminalImage).image = display_path
        if video_path:
            self.query_one("#media-status", Static).update("Video ready · Space to play/pause · ←/→ switch attachment")
        elif attachment.get("type") in {"video", "gif"}:
            self.query_one("#media-status", Static).update(
                "Preview only · this backend did not provide a playable video URL · ←/→ switch attachment"
            )
        else:
            self.query_one("#media-status", Static).update("Image preview · ←/→ switch attachment · Esc returns to feed")

    def _media_failed(self, index: int, message: str) -> None:
        if index != self.index:
            return
        self.query_one("#media-caption", Static).update(f"Attachment {index + 1}/{len(self.attachments)}")
        self.query_one("#media-status", Static).update(message)

    @work(thread=True, exclusive=True, group="video-playback")
    def playback_worker(self, path: Path, stop_event: threading.Event,
                        playing_event: threading.Event) -> None:
        self.playback_active = True
        try:
            for frame in video_frames(path, stop_event, playing_event):
                if stop_event.is_set():
                    break
                _post_from_worker(self.app, self._show_video_frame, frame)
        except Exception as exc:
            stop_event.set()
            _post_from_worker(self.app, self._media_failed, self.index, str(exc))
        finally:
            if stop_event is self.stop_event:
                self.playback_active = False
                _post_from_worker(self.app, self._video_finished, stop_event)

    def _video_finished(self, stop_event: threading.Event) -> None:
        if stop_event is self.stop_event and not stop_event.is_set():
            self.playing_event.clear()
            self.query_one("#media-status", Static).update("Video ended · Space restarts · ←/→ switch attachment")

    def _show_video_frame(self, frame) -> None:
        self.query_one("#media-frame", TerminalImage).image = frame


def run() -> None:
    init_db()
    VidurApp().run()
