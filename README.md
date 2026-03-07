# PyTorch Memory Profiler CLI

An interactive terminal UI for exploring PyTorch CUDA memory snapshots. Navigate memory usage over time, drill into allocation call stacks, and pinpoint exactly what is consuming your GPU memory at any moment.

## Features

- **Timeline view** — ASCII bar chart of memory over time with horizontal zoom and vertical pan
- **Snapshot/detail view** — expandable call-stack tree showing allocations grouped by frame at any selected moment
- **Heat-map coloring** — frames colored blue → cyan → green → yellow → red by share of total memory
- **Pre-recording allocations** — memory allocated before `_record_memory_history()` is reconstructed from the `segments` final state so the timeline is always accurate from the very start
- **Fast cache** — parsed data is cached next to the source file (`.memcache`) so subsequent loads are instant

## Recording a snapshot

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100_000)

# ... run your model / training step ...

torch.cuda.memory._dump_snapshot("snapshot.pkl")
torch.cuda.memory._record_memory_history(None)  # stop recording
```

## Installation

> TODO

## Usage

```
python memory_profiler_cli.py <snapshot.pkl>
```

The tool parses the snapshot (or loads the cache), prints a brief summary, then launches the interactive UI.

## Controls

### Timeline view

| Key | Action |
|-----|--------|
| `←` / `→` | Step one column left / right |
| `b` / `f` | Jump ±15 columns |
| `[` / `]` | Jump ±¼ page left / right |
| `+` / `-` | Zoom in / out (cursor stays centered) |
| `↑` / `↓` | Pan y-axis up / down (raise or lower the visible memory floor) |
| `r` | Reset y-axis to full range (bottom = 0) |
| `Enter` | Open snapshot detail at current cursor position |
| `q` | Quit |

### Snapshot / detail view

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate rows |
| `[` / `]` | Jump to previous / next sibling frame |
| `→` / `Enter` | Expand or collapse selected frame |
| `E` | Recursively expand selected frame and all descendants |
| `←` | Collapse selected frame (or nearest expanded ancestor) |
| `q` | Return to timeline view |

## How it works

PyTorch's `_dump_snapshot()` produces a pickle file containing:

- **`device_traces`** — a sequence of `alloc` / `free_requested` / `free_completed` events with timestamps and call frames
- **`segments`** — the final live state of every CUDA memory segment at dump time, including blocks and their frames

The tool replays `alloc` / `free_requested` events (ignoring `free_completed`, which is a cache-layer detail) to reconstruct the full memory timeline. It also derives memory allocated *before* `_record_memory_history()` was called from the `segments` data, so the timeline baseline is accurate rather than starting from zero.

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.
You are free to use, study, modify, and distribute this software, provided that any derivative works are also distributed under the same license.
See the [LICENSE](LICENSE) file or <https://www.gnu.org/licenses/gpl-3.0.html> for the full text.
