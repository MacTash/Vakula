# Vidur

Vidur is a local-first, terminal-native research workspace for collecting and reviewing public-source information. It combines a scriptable Python CLI with a keyboard-driven TUI. Retrieved source text stays visible as returned by its source backend; AI explanations are shown separately.

## Requirements

- Python 3.12 or newer
- Node.js 18 or newer for the npm launcher
- Ollama for local AI (optional)
- Agent Reach for routed social-platform access (optional; configured separately)

Python 3.10 is no longer supported. The 0.2 release uses Textual, terminal image rendering, and PyAV for media playback.

## Install from a checkout

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m vidur
```

Use any Python 3.12 or newer interpreter in place of `python3`. `make run` and `make test` accept either `VENV=<path>` or `PYTHON=<path>`:

```bash
make test VENV=.venv
```

On Windows, use `py -3.12 -m venv .venv` and `.venv\\Scripts\\python.exe -m pip install .`.

The npm package is a launcher around the Python application. From a checkout:

```bash
npm install --global .
vidur
```

The npm installer creates an isolated Python environment and installs the project dependencies. It requires Python 3.12+. The npm package name is `vidur`.

The import package, CLI, database file and this repository are all plain `vidur`. The only exception is PyPI, where `vidur` is already held by an unrelated project, so a future PyPI release must ship under a distinct slug such as `vidur-scope` (free on both PyPI and npm) or be installed from git:

```bash
pip install git+https://github.com/MacTash/vidur
```

## Upgrading from the pre-rename releases

Earlier releases shipped as `geoscope`. No action is needed; a pre-rename install keeps working.

- The `geoscope` data directory is still used when no `vidur` one exists, so an existing database stays reachable instead of appearing empty.
- An existing `geoscope.db` is opened in place. If that directory is adopted but holds no such file, Vidur writes `vidur.db` there rather than reviving the old filename.
- `GEOSCOPE_DATA_DIR`, `GEOSCOPE_CACHE_DIR`, `GEOSCOPE_SEARXNG_URL`, `GEOSCOPE_AI_BASE_URL`, `GEOSCOPE_AI_MODEL`, `GEOSCOPE_AI_API_KEY`, `GEOSCOPE_PROVIDER_API_KEY` and `GEOSCOPE_OLLAMA_URL` still work. `VIDUR_*` takes precedence when both are set.

## TUI

The main screens are **Overview**, **Live Feed**, **Research**, **Saved**, **Sources**, and **Settings**. Overview keeps the existing weather and earthquake collectors available without AI. Live Feed can search X, Reddit, GitHub repositories, and YouTube through their Agent Reach upstream tools. Select a result to inspect the returned text, source time, author, backend, and permalink. X posts also show engagement fields and attachments. The media viewer displays images and plays supported X videos inside the TUI. Use the arrow keys to move between attachments, Space to play or pause video, and Esc to return to the feed. Terminals without image protocols receive a lower-detail character-rendered preview.

Vidur only downloads media when you open it. X CDN previews are bounded to 20 MB for images and 120 MB for video, held in a user cache, and old cache entries are pruned when the cache grows beyond 500 MB. If an active source backend omits a media URL, Vidur shows that the attachment is unavailable and keeps the post permalink.

In **Research**, ask Qwen basic questions or use slash commands:

```text
/help
/x <query>
/sources
/models
/local [installed-model-name]
/mode local
/mode provider
/provider base <url>
/provider model <name>
/provider key <key>
/search <query>
/open <url>
/status
```

## Agent Reach

Install and configure Agent Reach separately. Vidur never installs source tools, accesses browser cookies, or configures logins on its own. **Sources** and `vidur sources` run the documented read-only `agent-reach doctor --json` check. X searches use the backend Agent Reach reports when available, then invoke its documented read-only upstream command with a fixed argument list. If `active_backend` is empty, Vidur follows the documented read-only verification path only when you request an X search.

For the `twitter-cli` backend, the upstream command needs `TWITTER_AUTH_TOKEN` and `TWITTER_CT0` in Vidur's environment. Agent Reach's saved cookie values are used for its doctor checks; they are not automatically passed to the `twitter` child process. OpenCLI can use an existing, user-controlled browser session. Vidur reports setup failures rather than silently switching to a different source.

CLI examples:

```bash
vidur sources
vidur x-search "earthquake response" --limit 10
vidur x-search "earthquake response" --json
vidur source-search github "geospatial incident mapping" --limit 8
```

## Local AI

Vidur uses **`qwen3:0.6b-q4_K_M`** as its small local helper. Ollama lists the model download at about 523 MB. The model handles basic questions and has one narrow read-only action for X search. It does not rewrite the posts shown in Live Feed.

Install Ollama separately, then use **Download Qwen3 0.6B** in Settings. Vidur asks before starting the download and does not pull model weights automatically. Provider API keys, when used, stay in memory for the current TUI session.

## Data and storage

Vidur stores its database in the operating system's user data directory. When first launched from the checkout root, it copies an existing `./data/vidur.db` — or a pre-rename `./data/geoscope.db` — into the new location without deleting the original. Set `VIDUR_DATA_DIR` to choose another database directory and `VIDUR_CACHE_DIR` to choose the media cache directory.

The SQLite schema is versioned and existing intelligence records are retained. Use `vidur status` to inspect the database and source-post count.

## Other CLI commands

```bash
vidur init
vidur status
vidur collect news "Red Sea shipping" --limit 15
vidur collect earthquakes --min-magnitude 5.5
vidur collect weather "Kochi, India"
vidur intel search shipping --category OSINT
vidur timeline --category GEOINT
vidur report "Red Sea"
vidur watch earthquakes --interval 300
```

The non-social collectors use their documented public feeds. Collection and media access should follow each service's terms and applicable law.
