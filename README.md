## LLM plays Z-machine text games

Run classic Z-machine interactive fiction (Zork, etc.) through a real interpreter (Jericho/Frotz) and let an LLM propose commands. Gameplay sessions are recorded as JSON logs under `savegames/` so you can replay/resume later and generate a living strategy guide from past playthroughs.

For details about how it's actually put together see [ARCHITECTURE.md](ARCHITECTURE.md)

## Requirements

- **OS**: Linux/macOS, or **Windows via WSL2** (recommended).
  This repo uses `jericho`/`FrotzEnv`, which is typically not a smooth native-Windows install.
- **Python**: 3.12+
- **uv**: Astral’s `uv` package manager (`uv sync`, `uv run`)
- **OpenAI API key**: set `OPENAI_API_KEY`

## Setup

### Linux / macOS / WSL2

1) Install `uv` (see `https://docs.astral.sh/uv/getting-started/installation`).

2) Run the installer:

```bash
./install.bash
```

What it does:
- `uv sync` to install Python deps from `pyproject.toml`
- creates `savegames/`
- downloads a bundle of Z-machine game files into `games/`

### Windows + WSL2

- **Install WSL2 + Ubuntu** (or another Linux distro), then run the Linux setup steps above inside WSL.
- Keep the repo on the Linux filesystem (best performance), e.g. under `~/src/...`, rather than `/mnt/c/...`.

## Configuration

### OpenAI key (required)

PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY_HERE"
```

bash/zsh:

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

Notes:
- `OPENAI_API_KEY` can be referenced in `llm_config.json` using `"${OPENAI_API_KEY}"`.

### LLM config (optional)

Create `llm_config.json` at the repo root to choose per-role providers/models:

```json
{
  "gameplay": {"model": "gpt-5-mini", "api_key": "${OPENAI_API_KEY}"},
  "summary": {"model": "gpt-5-mini", "api_key": "${OPENAI_API_KEY}"},
  "strategy": {
    "model": "llama2",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama"
  },
  "post_mortem": {
    "model": "llama2",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama"
  }
}
```

Fields:
- `model` (required): model name for the role.
- `base_url` (optional): set for local providers (e.g., Ollama).
- `api_key` (optional): can be a dummy value for local providers.

## Usage

### Run a game

Show help:

```bash
uv run main.py --help
```

Start a new run (example game):

```bash
uv run main.py --game games/zork1.z5
```

Start in human mode (type commands yourself; enter `LLM` to hand over control):

```bash
uv run main.py --game games/zork1.z5 --human
```

Resume/replay from an existing JSON log (it replays the command history, then appends new steps):

```bash
uv run main.py --game games/zork1.z5 --load savegames/zork1_YYYYMMDD_HHMMSS.json
```

### Post-mortem learning (strategy guide)

Generate or update a short, living strategy guide from all savegames for a game:

```bash
uv run post-mortem.py zork1
```

Notes:
- The guide is written to `strategies/<game>.md` and is overwritten with the latest concise strategy.
- A per-game index is kept at `strategies/<game>.csv` (one savegame filename per line) so repeated runs skip already-processed logs.
- Use `--model` to override the default model, e.g. `uv run post-mortem.py zork1 --model gpt-5-mini`.

## Files & data

- **Games**: `games/` contains `.z3/.z4/.z5/.z8` story files.
- **Logs / “savegames”**: `savegames/*.json` is an append-only history of steps:
  - `actor`: `"human"` or `"llm"`
  - `command`: what was sent to the game
  - `observation`: game output after the command
  - `aux`: debug snapshot captured via `inventory` + `look`
- **Strategies**: `strategies/*.md` is a short, markdown strategy guide used to inform future runs.

## Troubleshooting

- **`OpenAI(api_key=...)` errors / auth failures**: confirm `OPENAI_API_KEY` is set in the same shell where you run `uv run ...`.
- **Jericho/Frotz install issues on Windows**: use WSL2; native Windows installs are often painful for this stack.
