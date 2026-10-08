# Geoscope

Geoscope is a local-first, terminal-native research workspace for collecting and reviewing public-source information. It combines a scriptable Python CLI with a keyboard-driven TUI. Retrieved source text stays visible as returned by its source backend; AI explanations are shown separately.

## Requirements

- Python 3.12 or newer
- Node.js 18 or newer for the npm launcher
- Ollama for local AI (optional)
- Agent Reach for routed social-platform access (optional; configured separately)

Python 3.10 is no longer supported. The 0.2 release uses Textual, terminal image rendering, and PyAV for media playback.

## Install from a checkout

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m geoscope
```

On Windows, use `py -3.12 -m venv .venv` and `.venv\\Scripts\\python.exe -m pip install .`.

The npm package is a launcher around the Python application. From a checkout:

```bash
npm install --global .
geoscope
```

The npm installer creates an isolated Python environment and installs the project dependencies. It requires Python 3.12+. The package name is `geoscope-tui`.

## TUI

The main screens are **Overview**, **Live Feed**, **Research**, **Saved**, **Sources**, and **Settings**. Overview keeps the existing weather and earthquake collectors available without AI. Live Feed can search X, Reddit, GitHub repositories, and YouTube through their Agent Reach upstream tools. Select a result to inspect the returned text, source time, author, backend, and permalink. X posts also show engagement fields and attachments. The media viewer displays images and plays supported X videos inside the TUI. Use the arrow keys to move between attachments, Space to play or pause video, and Esc to return to the feed. Terminals without image protocols receive a lower-detail character-rendered preview.

Geoscope only downloads media when you open it. X CDN previews are bounded to 20 MB for images and 120 MB for video, held in a user cache, and old cache entries are pruned when the cache grows beyond 500 MB. If an active source backend omits a media URL, Geoscope shows that the attachment is unavailable and keeps the post permalink.

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

Install and configure Agent Reach separately. Geoscope never installs source tools, accesses browser cookies, or configures logins on its own. **Sources** and `geoscope sources` run the documented read-only `agent-reach doctor --json` check. X searches use the backend Agent Reach reports when available, then invoke its documented read-only upstream command with a fixed argument list. If `active_backend` is empty, Geoscope follows the documented read-only verification path only when you request an X search.

For the `twitter-cli` backend, the upstream command needs `TWITTER_AUTH_TOKEN` and `TWITTER_CT0` in Geoscope's environment. Agent Reach's saved cookie values are used for its doctor checks; they are not automatically passed to the `twitter` child process. OpenCLI can use an existing, user-controlled browser session. Geoscope reports setup failures rather than silently switching to a different source.

CLI examples:

```bash
geoscope sources
geoscope x-search "earthquake response" --limit 10
geoscope x-search "earthquake response" --json
geoscope source-search github "geospatial incident mapping" --limit 8
```

## Local AI

Geoscope uses **`qwen3:0.6b-q4_K_M`** as its small local helper. Ollama lists the model download at about 523 MB. The model handles basic questions and has one narrow read-only action for X search. It does not rewrite the posts shown in Live Feed.

Install Ollama separately, then use **Download Qwen3 0.6B** in Settings. Geoscope asks before starting the download and does not pull model weights automatically. Provider API keys, when used, stay in memory for the current TUI session.

## Data and storage

Geoscope stores its database in the operating system's user data directory. When first launched from the checkout root, it copies an existing `./data/geoscope.db` into the new location without deleting the original. Set `GEOSCOPE_DATA_DIR` to choose another database directory and `GEOSCOPE_CACHE_DIR` to choose the media cache directory.

The SQLite schema is versioned and existing intelligence records are retained. Use `geoscope status` to inspect the database and source-post count.

## Other CLI commands

```bash
geoscope init
geoscope status
geoscope collect news "Red Sea shipping" --limit 15
geoscope collect earthquakes --min-magnitude 5.5
geoscope collect weather "Kochi, India"
geoscope intel search shipping --category OSINT
geoscope timeline --category GEOINT
geoscope report "Red Sea"
geoscope watch earthquakes --interval 300
```

The non-social collectors use their documented public feeds. Collection and media access should follow each service's terms and applicable law.
