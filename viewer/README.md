# Run viewer (`viewer/`)

A static browser viewer for protocol-v2 runs. No build step, no dependencies —
plain HTML/CSS/JS that reads the `run.json` each run produces.

## What a run saves

Every `python -m glyph.v2 run` writes a self-contained **`run.json`** into its
run dir (next to `report.json`). It bundles the config, the measurement prompts
(system + practice/final openers), the task README the agent saw, the report,
and a normalized **turn-by-turn transcript** — the agent's visible reasoning,
every tool call with its input, and every result (glyph tools, Bash, file
tools). Extended-thinking *text* is redacted by the CLI, so only a per-turn
"thinking ×N (redacted)" marker is shown; context compactions are marked too.

## Viewing runs

1. **Collect** the runs you want to look at into `viewer/runs/`:

   ```bash
   python tools/collect_runs.py --out viewer/runs /path/to/your/out_root
   ```
   Point it at one or more directories (it finds every `run.json` under them)
   and it writes `viewer/runs/<id>.json` plus `viewer/runs/index.json`.

2. **Serve** the repo and open the viewer:

   ```bash
   python -m http.server 8000        # from the repo root
   # then open http://localhost:8000/viewer/
   ```
   (A plain `file://` open will not work — the viewer fetches JSON, which
   browsers block over `file://`. Any static server is fine.)

## Running remotely, viewing locally

Runs happen on the box with the GPU/sandbox; you read them on your laptop:

```bash
# on the remote host, after some runs:
python tools/collect_runs.py --out viewer/runs /tmp/glyph_runs   # or your out_root

# on your laptop:
scp -r you@remote:~/code/Glyph/viewer/runs viewer/runs
python -m http.server 8000
# open http://localhost:8000/viewer/
```

## The interface

- **Left** — a gallery of runs with filters by **arm** and **π**; each card
  shows arm / preset / seed / π / overall / spend / turns / status.
- **Right** — the selected run: stat cards, collapsible Config / System prompt /
  openers / Task README / full report, then the transcript with a
  practice→final divider, per-turn tool calls (inputs and results, errors
  highlighted), and the agent's reasoning collapsed by default.

Light/dark theme follows the browser and can be toggled.
