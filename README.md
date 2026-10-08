# Vakula

Vakula is a local-first, terminal-native research workspace for collecting and reviewing public-source information. It combines a scriptable Python CLI with a keyboard-driven TUI. Retrieved source text stays visible as returned by its source backend; AI explanations are shown separately.

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
.venv/bin/python -m vakula
```

Use any Python 3.12 or newer interpreter in place of `python3`. `make run` and `make test` accept either `VENV=<path>` or `PYTHON=<path>`:

```bash
make test VENV=.venv
```

On Windows, use `py -3.12 -m venv .venv` and `.venv\\Scripts\\python.exe -m pip install .`.

The npm package is a launcher around the Python application. From a checkout:

```bash
npm install --global .
vakula
```

The npm installer creates an isolated Python environment and installs the project dependencies. It requires Python 3.12+. The npm package name is `vakula`.

The import package, CLI, database file, GitHub repository
(https://github.com/MacTash/Vakula), PyPI name and npm name are all plain
`vakula`, and the name is unclaimed on both PyPI and npm, so it is reserved for
this project rather than a slug or a scope.

**Vakula is not published to PyPI or npm yet.** Until it is, install from the
repository:

```bash
pip install git+https://github.com/MacTash/Vakula
```

Once a registry release lands, this becomes simply `pip install vakula` and
`npm install -g vakula` — the reason the rename was worth doing is precisely
that the plain name is free for it.

## Upgrading from earlier releases

No action is needed. Geoscope and Vidur were both real releases, and both are
still honoured so an existing install keeps working rather than appearing to lose
its database.

- The `vidur` data directory is used when no `vakula` one exists, and `geoscope`
  after that.
- An existing `vidur.db` or `geoscope.db` is opened in place. If that directory is
  adopted but holds neither, Vakula writes `vakula.db` there rather than reviving
  an old filename.
- `VIDUR_*` and `GEOSCOPE_*` settings still work behind the `VAKULA_*` names:
  `VIDUR_DATA_DIR`, `VIDUR_CACHE_DIR`, `VIDUR_SEARXNG_URL`, `VIDUR_AI_BASE_URL`,
  `VIDUR_AI_MODEL`, `VIDUR_AI_API_KEY`, `VIDUR_PROVIDER_API_KEY`,
  `VIDUR_OLLAMA_URL`, and the `GEOSCOPE_` equivalents. `VAKULA_*` takes precedence
  when more than one is set.

## TUI

The main screens are **Overview**, **Live Feed**, **Research**, **Saved**, **Sources**, and **Settings**. Overview keeps the existing weather and earthquake collectors available without AI. Live Feed can search X, Reddit, GitHub repositories, and YouTube through their Agent Reach upstream tools. Select a result to inspect the returned text, source time, author, backend, and permalink. X posts also show engagement fields and attachments. The media viewer displays images and plays supported X videos inside the TUI. Use the arrow keys to move between attachments, Space to play or pause video, and Esc to return to the feed. Terminals without image protocols receive a lower-detail character-rendered preview.

Vakula only downloads media when you open it. X CDN previews are bounded to 20 MB for images and 120 MB for video, held in a user cache, and old cache entries are pruned when the cache grows beyond 500 MB. If an active source backend omits a media URL, Vakula shows that the attachment is unavailable and keeps the post permalink.

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

Install and configure Agent Reach separately. Vakula never installs source tools, accesses browser cookies, or configures logins on its own. **Sources** and `vakula sources` run the documented read-only `agent-reach doctor --json` check. X searches use the backend Agent Reach reports when available, then invoke its documented read-only upstream command with a fixed argument list. If `active_backend` is empty, Vakula follows the documented read-only verification path only when you request an X search.

For the `twitter-cli` backend, the upstream command needs `TWITTER_AUTH_TOKEN` and `TWITTER_CT0` in Vakula's environment. Agent Reach's saved cookie values are used for its doctor checks; they are not automatically passed to the `twitter` child process. OpenCLI can use an existing, user-controlled browser session. Vakula reports setup failures rather than silently switching to a different source.

CLI examples:

```bash
vakula sources
vakula x-search "earthquake response" --limit 10
vakula x-search "earthquake response" --json
vakula source-search github "geospatial incident mapping" --limit 8
```

## Local AI

Vakula uses **`qwen3:0.6b-q4_K_M`** as its small local helper. Ollama lists the model download at about 523 MB. The model handles basic questions and has one narrow read-only action for X search. It does not rewrite the posts shown in Live Feed.

Install Ollama separately, then use **Download Qwen3 0.6B** in Settings. Vakula asks before starting the download and does not pull model weights automatically. Provider API keys, when used, stay in memory for the current TUI session.

## Data and storage

Vakula stores its database in the operating system's user data directory. When first launched from the checkout root, it copies an existing `./data/vakula.db` — or a pre-rename `./data/geoscope.db` — into the new location without deleting the original. Set `VAKULA_DATA_DIR` to choose another database directory and `VAKULA_CACHE_DIR` to choose the media cache directory.

The SQLite schema is versioned and existing intelligence records are retained. Use `vakula status` to inspect the database and source-post count.

## Intelligence

Vakula stores evidence, correlates it, measures it, and only then asks a language
model to explain the result. The model never decides what happened.

```text
observations -> events -> contradictions -> analytics -> forecast
             -> intelligence state -> assessment -> model prose
```

Every claim in a briefing is labelled `OBSERVED`, `INFERRED`, `PREDICTED` or
`UNKNOWN`, and carries the `OBS-n` and `EVT-n` identifiers behind it. The
forecast probability is computed by `Vakula Forecast Engine v0.1` and is never
supplied or adjusted by a model. Contradictions between sources are recorded and
left unresolved; a model is never asked which source is right.

Source text is untrusted. Retrieved text reaches the model only inside a
delimited data block and is never treated as instructions, and untagged or
`[OBSERVED]` model output is quarantined rather than printed as a finding.

```bash
vakula assess "Taiwan Strait"      # intelligence assessment for a scope
vakula brief "Red Sea" --json      # structured briefing
vakula assess --no-model           # deterministic rendering, no model consulted
vakula evidence OBS-1842           # the stored source behind a claim
vakula watchlist add "Taiwan Strait"
vakula watchlist list
vakula watchlist rm "Taiwan Strait"
```

If no model is available Vakula still produces a complete briefing from stored
evidence and says so. Adding a watch target is local database state and involves
no network access. `geoscope watch earthquakes --interval 300` remains the
periodic collector poll and is unrelated to the watchlist namespace.

## Other CLI commands

```bash
vakula init
vakula status
vakula collect news "Red Sea shipping" --limit 15
vakula collect earthquakes --min-magnitude 5.5
vakula collect weather "Kochi, India"
vakula intel list                    # stored intelligence, newest first
vakula intel search shipping --category OSINT
vakula timeline --category GEOINT
vakula report "Red Sea"
vakula watch earthquakes --interval 300
```

The non-social collectors use their documented public feeds. Collection and media access should follow each service's terms and applicable law.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

Permissive: use it, modify it, ship it commercially. Two clauses are worth
knowing about. There is an **express patent grant**, which matters here because
the event-fusion scoring model, the four-way confidence separation and the
forecast slope-standard-error test are the novel parts; and there is a
**patent-retaliation clause**, so that grant terminates if the patent holder
initiates patent litigation against a user.

Neither licence nor file grants trademark rights. "Vidur" and "Vakula" remain
yours.
