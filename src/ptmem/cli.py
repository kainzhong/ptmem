#!/usr/bin/env python3
"""
PyTorch Memory Profiler - Interactive CLI Analyzer

Usage:
    ptmem <snapshot.pkl>
    ptmem -c, --compare <snapshot1.pkl> <snapshot2.pkl>
    ptmem -v, --version
    ptmem -h, --help

Timeline View Controls:
    ← →       Step one column left / right
    b / f     Jump ±15 columns
    [ / ]     Jump ±¼ page left / right
    + / -     Zoom in / out
    ↑ ↓       Pan y-axis up / down
    r         Reset y-axis to full range
    Enter     Open snapshot detail at current cursor position
    q         Quit

Snapshot / Detail View Controls:
    ↑ ↓       Navigate rows
    [ / ]     Jump to previous / next sibling frame
    Enter     Expand or collapse selected frame
    →         Move to first child (expands if needed)
    ←         Move to parent
    e         Recursively expand selected frame and all descendants
    c         Collapse all frames
    r         Jump to root of current tree
    f         Focus: make selected frame the new root (resets indentation)
    /         Open search bar (type to filter by name or filename)
    n / N     Jump to next / previous search match
    Esc       Clear search highlights
    q         Unfocus (pop focus stack) / return to timeline view

Compare Mode Controls (-c / --compare):
    { or }    Switch focus between left and right pane
    (all other keys operate on the focused pane as normal)

Notes:
    - Snapshot is captured by calling:
        import torch
        torch.cuda.memory._record_memory_history(max_entries=100000)
        # ... run your model ...
        torch.cuda.memory._dump_snapshot("snapshot.pkl")
"""

import curses
import datetime
import math
import os
import pickle
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict


# ─── Data Layer ───────────────────────────────────────────────────────────────

class Allocation:
    """Represents a single memory allocation and its lifetime."""
    __slots__ = ('addr', 'size', 'time_alloc', 'time_free', 'frames', 'device')

    def __init__(self, addr, size, time_alloc, frames, device):
        self.addr = addr
        self.size = size
        self.time_alloc = time_alloc
        self.time_free = math.inf   # updated when freed
        self.frames = frames        # list of {'name', 'filename', 'line'}, innermost first
        self.device = device


class MemoryProfileData:
    """Loads and indexes a PyTorch memory snapshot pickle file.

    Parsed results are cached in a hidden .memcache file next to the source so
    subsequent launches are instant.  The cache is invalidated whenever the
    source file's mtime changes.
    """

    CACHE_VERSION = 5

    def __init__(self, path: str):
        self.path = path
        self.allocations: list = []
        # timeline_times  = event indices 0, 1, 2, … (like PyTorch's JS viz)
        # timeline_times_us = corresponding wall-clock timestamps in µs (for display)
        self.timeline_times: list = []
        self.timeline_times_us: list = []
        self.timeline_memory: list = []
        self.min_time: int = 0   # always 0 (first event index)
        self.max_time: int = 0   # N-1 (last event index)
        self.max_memory: int = 0
        self.initial_memory: int = 0
        self.wall_min_us: int = 0   # actual wall-clock start (µs) for display
        self.wall_max_us: int = 0   # actual wall-clock end (µs) for display
        self.from_cache: bool = False

        cache_path = self._cache_path()
        if self._load_cache(cache_path):
            self.from_cache = True
            return

        with open(path, 'rb') as f:
            raw = pickle.load(f)
        self._parse(raw)
        self._save_cache(cache_path)

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_path(self) -> str:
        d = os.path.dirname(os.path.abspath(self.path))
        b = os.path.basename(self.path)
        return os.path.join(d, '.' + b + '.memcache')

    def _load_cache(self, cache_path: str) -> bool:
        try:
            src_mtime = os.path.getmtime(self.path)
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if cache.get('version') != self.CACHE_VERSION:
                return False
            if cache.get('source_mtime') != src_mtime:
                return False
            self.allocations       = cache['allocations']
            self.timeline_times    = cache['timeline_times']
            self.timeline_times_us = cache['timeline_times_us']
            self.timeline_memory   = cache['timeline_memory']
            self.min_time          = cache['min_time']
            self.max_time          = cache['max_time']
            self.max_memory        = cache['max_memory']
            self.initial_memory    = cache['initial_memory']
            self.wall_min_us       = cache['wall_min_us']
            self.wall_max_us       = cache['wall_max_us']
            return True
        except Exception:
            return False

    def _save_cache(self, cache_path: str) -> None:
        try:
            cache = {
                'version':      self.CACHE_VERSION,
                'source_mtime': os.path.getmtime(self.path),
                'allocations':      self.allocations,
                'timeline_times':   self.timeline_times,
                'timeline_times_us': self.timeline_times_us,
                'timeline_memory':  self.timeline_memory,
                'min_time':         self.min_time,
                'max_time':         self.max_time,
                'max_memory':       self.max_memory,
                'initial_memory':   self.initial_memory,
                'wall_min_us':      self.wall_min_us,
                'wall_max_us':      self.wall_max_us,
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)
        except Exception:
            pass   # cache failure is non-fatal

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, raw: dict) -> None:
        device_traces = raw.get('device_traces', [])
        if not device_traces:
            flat = raw.get('traces', [])
            if flat:
                device_traces = [flat]

        # Collect alloc and free_completed events, matching PyTorch's MemoryViz.js:
        #   alloc          → tensor carved from cache pool  (+size to allocated)
        #   free_completed → block returned to CUDA pool    (-size from allocated)
        # (free_requested is intentionally skipped, matching the JS behaviour.)
        events = []
        synthetic_time = 0

        for device_idx, trace in enumerate(device_traces):
            for event in trace:
                action = event.get('action', '')
                if action not in ('alloc', 'free_completed'):
                    continue
                size   = event.get('size', 0)
                addr   = event.get('addr', 0)
                frames = event.get('frames', [])
                time_us = event.get('time_us')
                if time_us is None:
                    time_us = synthetic_time
                    synthetic_time += 1
                events.append((time_us, addr, size, frames, device_idx, action))

        if not events:
            return

        events.sort(key=lambda e: e[0])

        # ── Establish baseline from segments ──────────────────────────────────
        # `segments` is the ground-truth final state.  Derive initial_memory so
        # that replaying the trace from it lands exactly on final_allocated.
        # We use b['size'] (allocator-rounded) consistently with event sizes.
        def _seg_allocated(seg):
            if 'allocated_size' in seg:
                return seg['allocated_size']
            return sum(b['size'] for b in seg.get('blocks', [])
                       if b.get('state') == 'active_allocated')

        final_allocated = sum(_seg_allocated(s) for s in raw.get('segments', []))
        net_from_trace  = sum(size if act == 'alloc' else -size
                              for _, _, size, _, _, act in events)
        initial_memory  = max(0, final_allocated - net_from_trace)
        self.initial_memory = initial_memory

        # ── Replay using event INDEX as the x-axis (matching PyTorch JS) ──────
        # PyTorch's MemoryViz.js uses a discrete "timestep" counter (one per
        # event) rather than wall-clock time.  This gives uniform horizontal
        # spacing so forward/backward passes look proportional regardless of
        # the actual elapsed time of each operation.
        active: dict = {}   # addr -> Allocation  (in-trace allocs only)
        current_memory = initial_memory

        # Record the state BEFORE the first event (event index -1) so the
        # initial baseline is visible at the very left of the chart.
        self.timeline_times.append(-1)
        self.timeline_times_us.append(events[0][0])   # wall time of first event
        self.timeline_memory.append(initial_memory)

        for event_idx, (time_us, addr, size, frames, device, action) in enumerate(events):
            if action == 'alloc':
                alloc = Allocation(addr, size, event_idx, frames, device)
                active[addr] = alloc
                self.allocations.append(alloc)
                current_memory += size
            elif action == 'free_completed':
                if addr in active:
                    active[addr].time_free = event_idx
                    del active[addr]
                current_memory = max(0, current_memory - size)

            self.timeline_times.append(event_idx)
            self.timeline_times_us.append(time_us)
            self.timeline_memory.append(current_memory)

        if self.timeline_times:
            self.min_time    = self.timeline_times[0]   # -1 (the baseline point)
            self.max_time    = self.timeline_times[-1]  # N-1
            self.max_memory  = max(self.initial_memory, max(self.timeline_memory))
            self.wall_min_us = self.timeline_times_us[0]
            self.wall_max_us = self.timeline_times_us[-1]

        # ── Reconstruct pre-recording allocations from segments ───────────────
        # active_allocated blocks whose address is not in `active` (in-trace
        # allocs still alive at snapshot time) were allocated before
        # _record_memory_history() started.  We create synthetic Allocation
        # objects for them so the detail view shows their frames and sizes.
        #
        # Block address field: newer PyTorch uses 'address' (not 'addr').
        in_trace_addrs = set(active.keys())
        pre_trace_time = self.min_time   # event index -1, guaranteed < 0

        for seg in raw.get('segments', []):
            device_idx = seg.get('device', 0)
            base_addr  = seg.get('address', 0)
            offset = 0
            for block in seg.get('blocks', []):
                size  = block.get('size', 0)
                state = block.get('state', '')
                if state == 'active_allocated':
                    addr   = block.get('address', block.get('addr', base_addr + offset))
                    frames = block.get('frames', [])
                    if addr not in in_trace_addrs:
                        alloc = Allocation(addr, size, pre_trace_time, frames, device_idx)
                        self.allocations.append(alloc)
                offset += size

    def get_memory_at(self, time_us: float) -> int:
        """Cumulative memory at the given time (step function)."""
        if not self.timeline_times:
            return 0
        idx = bisect_right(self.timeline_times, time_us) - 1
        return self.timeline_memory[idx] if idx >= 0 else self.initial_memory

    def get_bucket_max_memory(self, start_us: float, end_us: float) -> int:
        """Peak memory in the half-open interval [start_us, end_us)."""
        if not self.timeline_times:
            return 0
        i0 = bisect_left(self.timeline_times, start_us)
        i1 = bisect_left(self.timeline_times, end_us)
        # Include the memory level just before the bucket starts
        mem_before = self.timeline_memory[i0 - 1] if i0 > 0 else self.initial_memory
        candidates = [mem_before] + self.timeline_memory[i0:i1]
        return max(candidates)

    def get_active_allocations_at(self, time_us: float) -> list:
        """All allocations alive at the given time."""
        return [a for a in self.allocations
                if a.time_alloc <= time_us < a.time_free]


# ─── Frame Tree ───────────────────────────────────────────────────────────────

def _frame_key(f: dict) -> tuple:
    return (f.get('name', ''), f.get('filename', ''), f.get('line', 0))


def _group_at_level(allocations: list, level: int) -> tuple:
    """Group allocations by outer_frames[level]. Returns (groups, ungrouped).

    groups is sorted descending by total bytes.
    Each group: (frame_dict, sub_allocs, total_bytes).
    ungrouped: allocations whose stack is shallower than level+1.
    """
    bucket: dict = defaultdict(list)
    ungrouped = []
    for alloc in allocations:
        outer = alloc.frames[::-1]   # outermost first
        if level < len(outer):
            nf = outer[level]
            bucket[_frame_key(nf)].append((nf, alloc))
        else:
            ungrouped.append(alloc)

    groups = []
    for items in bucket.values():
        fd = items[0][0]
        sa = [x[1] for x in items]
        groups.append((fd, sa, sum(a.size for a in sa)))
    groups.sort(key=lambda x: x[2], reverse=True)
    return groups, ungrouped


def build_tree_rows(allocations: list, expanded_tree: dict,
                    level: int = 0, key_path: tuple = ()) -> list:
    """Build a flat list of display rows for the expandable memory tree.

    expanded_tree is a nested dict: presence of a frame_key means that frame is
    expanded; its value is the sub-tree for its children.  Multiple siblings at
    the same depth can be expanded simultaneously without affecting each other.

    Each row tuple:
        (depth, frame_dict|None, sub_allocs, total_bytes, is_expanded,
         level_total, key_path)
    frame_dict=None marks a leaf row (allocations with no further call frames).
    """
    groups, ungrouped = _group_at_level(allocations, level)
    level_total = sum(g[2] for g in groups) + sum(a.size for a in ungrouped)

    rows = []
    for fd, sa, tb in groups:
        fk = _frame_key(fd)
        is_exp = fk in expanded_tree
        cur_path = key_path + (fk,)
        rows.append((level, fd, sa, tb, is_exp, level_total, cur_path))
        if is_exp:
            rows.extend(build_tree_rows(sa, expanded_tree[fk], level + 1, cur_path))

    if ungrouped:
        ub = sum(a.size for a in ungrouped)
        rows.append((level, None, ungrouped, ub, False, level_total, key_path))

    return rows


# ─── Formatting ───────────────────────────────────────────────────────────────

def fmt_bytes(n: int) -> str:
    for unit, thr in [('GB', 1 << 30), ('MB', 1 << 20), ('KB', 1 << 10)]:
        if n >= thr:
            return f'{n / thr:.1f}{unit}'
    return f'{n}B'


def fmt_time(us: float) -> str:
    if us >= 1_000_000:
        return f'{us / 1_000_000:.3f}s'
    if us >= 1_000:
        return f'{us / 1_000:.2f}ms'
    return f'{us:.0f}µs'


def heat_pair(pct: float) -> int:
    """Return curses color pair (5–9) for a fraction 0.0–1.0 of total memory."""
    if pct < 0.05:
        return 5
    if pct < 0.20:
        return 6
    if pct < 0.40:
        return 7
    if pct < 0.70:
        return 8
    return 9


def heat_pair_match(pct: float) -> int:
    """Return curses color pair (10–14) — same heat colors but on white background."""
    if pct < 0.05:
        return 10
    if pct < 0.20:
        return 11
    if pct < 0.40:
        return 12
    if pct < 0.70:
        return 13
    return 14


def safe_addstr(win, y, x, text, attr=0):
    """addstr that silently ignores out-of-bounds errors."""
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def safe_addch(win, y, x, ch, attr=0):
    try:
        win.addch(y, x, ch, attr)
    except curses.error:
        pass


# ─── Timeline View ────────────────────────────────────────────────────────────

class TimelineView:
    """Bar chart view of memory over time."""

    Y_AXIS_W = 9   # columns reserved for y-axis labels (shared by render & handle_key)

    def __init__(self, data: MemoryProfileData):
        self.data = data
        self.cursor = 0          # selected bucket index (absolute)
        self.view_start = 0      # leftmost visible bucket index (absolute)
        self.bucket_us = 1000    # microseconds per bucket column
        self.y_bottom = 0        # bytes; bottom of visible y range (0 = full range)
        self._auto_fit_done = False

    def _auto_fit(self, n_cols: int):
        """Set initial bucket size to show all data in one screen."""
        if self._auto_fit_done:
            return
        span = self.data.max_time - self.data.min_time
        if span > 0 and n_cols > 0:
            self.bucket_us = max(1, span // n_cols)
        self._auto_fit_done = True

    def _total_buckets(self) -> int:
        span = self.data.max_time - self.data.min_time
        if span <= 0:
            return 1
        return max(1, math.ceil(span / self.bucket_us))

    def _bucket_start_us(self, idx: int) -> float:
        return self.data.min_time + idx * self.bucket_us

    def current_time_us(self) -> float:
        return self._bucket_start_us(self.cursor)

    def handle_key(self, key: int, n_cols: int) -> str:
        total = self._total_buckets()

        if key == curses.KEY_RIGHT:
            self.cursor = min(self.cursor + 1, total - 1)
            if self.cursor >= self.view_start + n_cols:
                self.view_start = self.cursor - n_cols + 1

        elif key == curses.KEY_LEFT:
            self.cursor = max(self.cursor - 1, 0)
            if self.cursor < self.view_start:
                self.view_start = self.cursor

        elif key in (ord('+'), ord('=')):
            cur_time = self.current_time_us()
            self.bucket_us = max(1, self.bucket_us // 2)
            self._reanchor(cur_time, n_cols)

        elif key == ord('-'):
            # Disallow zoom-out when the full timeline already fits on screen.
            if self._total_buckets() <= n_cols:
                return 'timeline'
            cur_time = self.current_time_us()
            self.bucket_us = self.bucket_us * 2
            self._reanchor(cur_time, n_cols)

        elif key == ord('['):
            jump = max(1, n_cols // 4)
            self.cursor = max(0, self.cursor - jump)
            if self.cursor < self.view_start:
                self.view_start = self.cursor

        elif key == ord(']'):
            jump = max(1, n_cols // 4)
            self.cursor = min(total - 1, self.cursor + jump)
            if self.cursor >= self.view_start + n_cols:
                self.view_start = self.cursor - n_cols + 1

        elif key == ord('b'):
            self.cursor = max(0, self.cursor - 15)
            if self.cursor < self.view_start:
                self.view_start = self.cursor

        elif key == ord('f'):
            self.cursor = min(total - 1, self.cursor + 15)
            if self.cursor >= self.view_start + n_cols:
                self.view_start = self.cursor - n_cols + 1

        elif key == curses.KEY_UP:
            y_top = self.data.max_memory or 1
            y_range = max(1, y_top - self.y_bottom)
            step = max(1, y_range // 10)
            self.y_bottom = min(self.y_bottom + step, y_top - 1)

        elif key == curses.KEY_DOWN:
            y_range = max(1, (self.data.max_memory or 1) - self.y_bottom)
            step = max(1, y_range // 10)
            self.y_bottom = max(0, self.y_bottom - step)

        elif key in (ord('r'), ord('R')):
            self.y_bottom = 0

        elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
            return 'snapshot'

        elif key in (ord('q'), ord('Q')):
            return 'quit'

        return 'timeline'

    def _reanchor(self, cur_time: float, n_cols: int):
        """After a zoom, place cursor at the same time and pin it to screen center.

        view_start may go negative; rendering treats out-of-range columns as empty.
        """
        self.cursor = int((cur_time - self.data.min_time) / self.bucket_us)
        self.cursor = max(0, min(self.cursor, self._total_buckets() - 1))
        # Centre cursor on screen, but never scroll before the first timestamp.
        self.view_start = max(0, self.cursor - n_cols // 2)

    def render(self, stdscr, height: int, width: int, extra_hint: str = ''):
        stdscr.erase()

        if not self.data.timeline_times:
            safe_addstr(stdscr, height // 2, 2, "No memory events found in snapshot.")
            stdscr.refresh()
            return

        Y_AXIS_W = self.Y_AXIS_W
        n_cols = width - Y_AXIS_W - 1
        self._auto_fit(n_cols)

        # Layout rows
        HEADER = 2
        FOOTER = 2
        AXIS_ROW_H = 2   # horizontal time axis + labels
        chart_rows = max(3, height - HEADER - FOOTER - AXIS_ROW_H)

        # Build bucket data for visible columns.
        # view_start may be negative (cursor pinned to center after zoom near start),
        # and bidx may exceed total_buckets (near end) — both render as empty columns.
        total_buckets = self._total_buckets()
        buckets = []
        for i in range(n_cols):
            bidx = self.view_start + i
            if bidx < 0 or bidx >= total_buckets:
                buckets.append((None, None, 0, bidx))   # empty / out-of-range
            else:
                b_start = self._bucket_start_us(bidx)
                b_end = b_start + self.bucket_us
                mem = self.data.get_bucket_max_memory(b_start, b_end)
                buckets.append((b_start, b_end, mem, bidx))

        y_top = self.data.max_memory or 1
        y_bottom = self.y_bottom
        y_range = max(1, y_top - y_bottom)

        # Header
        cur_event = int(self.current_time_us())   # event index (-1 = baseline)
        cur_mem   = self.data.get_memory_at(cur_event)

        # Wall-clock time at the cursor position
        tus = self.data.timeline_times_us
        if tus:
            clamp = max(0, min(cur_event, len(tus) - 1))
            wall_cursor_us = tus[clamp]
            rel_us = wall_cursor_us - tus[0]
        else:
            rel_us = 0
        wall_span_us = self.data.wall_max_us - self.data.wall_min_us
        n_events = self.data.max_time + 1   # event indices 0 … max_time

        hdr1 = (f"PyTorch Memory Profiler  |  "
                f"Bucket: {self.bucket_us} events/col  |  "
                f"Cursor: event {max(0, cur_event)} (+{fmt_time(rel_us)})  |  "
                f"Memory: {fmt_bytes(cur_mem)}")
        y_range_s = (f"Y: {fmt_bytes(y_bottom)} – {fmt_bytes(y_top)}"
                     if y_bottom > 0 else f"Peak: {fmt_bytes(self.data.max_memory)}")
        hdr2 = (f"Events: {n_events}  |  Wall span: {fmt_time(wall_span_us)}  |  "
                f"{y_range_s}")
        safe_addstr(stdscr, 0, 0, hdr1[:width - 1], curses.A_BOLD)
        safe_addstr(stdscr, 1, 0, hdr2[:width - 1])

        # Y-axis: draw labels and gridlines
        chart_cols = max(4, width - Y_AXIS_W - 1)
        # Shrink buckets list to fit available chart columns
        buckets = buckets[:chart_cols]

        N_TICKS = 5
        tick_rows: set = set()
        for tick in range(N_TICKS):
            frac = tick / (N_TICKS - 1)
            mem_val = int(y_bottom + frac * y_range)
            label = fmt_bytes(mem_val)
            chart_row = chart_rows - 1 - int(frac * (chart_rows - 1))
            tick_rows.add(chart_row)
            # Label right-aligned in Y-axis area, then separator char
            padded = label.rjust(Y_AXIS_W - 1)
            safe_addstr(stdscr, HEADER + chart_row, 0, padded, curses.A_DIM)
            safe_addch(stdscr, HEADER + chart_row, Y_AXIS_W - 1, '┤', curses.A_DIM)

        # Bar chart (starts at column Y_AXIS_W)
        rel_cursor = self.cursor - self.view_start

        for col_i, (b_start, b_end, mem, bidx) in enumerate(buckets):
            # Clip bar to the visible y window [y_bottom, y_top]
            bar_frac = min(1.0, max(0.0, (mem - y_bottom) / y_range)) if y_range > 0 else 0
            bar_h = round(bar_frac * chart_rows)
            is_cursor = (col_i == rel_cursor)

            for row in range(chart_rows):
                # row 0 = top, chart_rows-1 = bottom
                filled = (chart_rows - row) <= bar_h
                is_gridline = row in tick_rows

                if is_cursor:
                    ch = '█' if filled else '│'
                    attr = curses.color_pair(2) | curses.A_BOLD
                elif filled:
                    ch = '█'
                    attr = curses.color_pair(1)
                elif is_gridline:
                    ch = '·'
                    attr = curses.A_DIM
                else:
                    ch = ' '
                    attr = 0

                safe_addch(stdscr, HEADER + row, Y_AXIS_W + col_i, ch, attr)

        # Horizontal axis line
        axis_row = HEADER + chart_rows
        safe_addstr(stdscr, axis_row, 0, ' ' * (Y_AXIS_W - 1) + '└' + '─' * min(chart_cols, width - Y_AXIS_W - 1), curses.A_DIM)

        # Event-index labels along x axis (skip out-of-range/empty columns)
        n_labels = max(2, min(8, chart_cols // 12))
        for i in range(n_labels + 1):
            col = int(i * (chart_cols - 1) / n_labels)
            if col < len(buckets):
                b_start = buckets[col][0]
                if b_start is None:
                    continue
                label = str(max(0, int(b_start)))   # event index
                screen_col = Y_AXIS_W + col
                if screen_col + len(label) < width:
                    safe_addstr(stdscr, axis_row + 1, screen_col, label, curses.A_DIM)

        # Footer controls
        ctrl = " ←→: Step   b/f: ±15 cols   [/]: Page   +/-: Zoom   ↑↓: Pan Y   r: Reset Y   Enter: Snapshot   q: Quit"
        safe_addstr(stdscr, height - 1, 0, (ctrl + extra_hint)[:width - 1], curses.A_REVERSE)

        stdscr.noutrefresh()


# ─── Snapshot View ────────────────────────────────────────────────────────────

class SnapshotView:
    """Expandable tree view of memory allocations grouped by call stack.

    expanded_tree is a nested dict (frame_key → sub-tree).  Multiple frames at
    the same depth can be expanded simultaneously; expanding or collapsing one
    does not affect any other branch.
    """

    INDENT = 2   # characters of indentation per depth level

    def __init__(self, data: MemoryProfileData, time_us: float):
        self.data = data
        self.time_us = time_us
        self.allocations = data.get_active_allocations_at(time_us)
        self.expanded_tree: dict = {}   # nested dict; presence == expanded
        self.cursor = 0
        self.scroll = 0
        self._rows: list = []
        # Focus state: narrows the tree to a subtree rooted at a chosen frame.
        # focus_sa    – allocations in scope (subset of self.allocations)
        # focus_level – frame level passed to build_tree_rows (≥ 0)
        # focus_stack – [(sa, level, expanded_tree, cursor, scroll, label), …]
        self.focus_sa: list = self.allocations
        self.focus_level: int = 0
        self.focus_stack: list = []
        # Search state
        self.search_mode: bool = False
        self.search_query: str = ''
        self.search_matches: list = []   # sorted row indices that match the query
        self._search_match_set: set = set()
        self._rebuild()

    def _rebuild(self):
        self._rows = build_tree_rows(self.focus_sa, self.expanded_tree,
                                     level=self.focus_level)
        self.cursor = min(self.cursor, max(0, len(self._rows) - 1))
        self.scroll = min(self.scroll, max(0, len(self._rows) - 1))
        self._update_search_matches()

    def _update_search_matches(self) -> None:
        """Recompute which visible rows match the current search query."""
        if not self.search_query:
            self.search_matches = []
            return
        q = self.search_query.lower()
        self.search_matches = [
            i for i, (_, fd, _, _, _, _, _) in enumerate(self._rows)
            if fd is not None
            and (q in fd.get('name', '').lower() or q in fd.get('filename', '').lower())
        ]
        self._search_match_set: set = set(self.search_matches)

    def _toggle(self, key_path: tuple) -> None:
        """Expand if absent, collapse if present — navigating expanded_tree by key_path."""
        tree = self.expanded_tree
        for k in key_path[:-1]:
            tree = tree[k]          # ancestor must exist if this row is visible
        leaf = key_path[-1]
        if leaf in tree:
            del tree[leaf]          # collapse (removes all children too)
        else:
            tree[leaf] = {}         # expand with empty child tree

    def _expand_recursive(self, key_path: tuple, sa: list, depth: int) -> None:
        """Recursively expand this node and all its descendants."""
        tree = self.expanded_tree
        for k in key_path[:-1]:
            tree = tree.setdefault(k, {})
        sub_tree = tree.setdefault(key_path[-1], {})
        # Temporarily point expanded_tree's sub-node to sub_tree so recursive calls work
        groups, _ = _group_at_level(sa, depth + 1)
        for fd, child_sa, _ in groups:
            fk = _frame_key(fd)
            child_path = key_path + (fk,)
            self._expand_recursive(child_path, child_sa, depth + 1)

    def _find_sibling(self, direction: int) -> int:
        """Return row index of previous (direction=-1) or next (direction=1) strict sibling."""
        if not (0 <= self.cursor < len(self._rows)):
            return self.cursor
        depth, fd, _, _, _, _, key_path = self._rows[self.cursor]
        if not key_path:
            return self.cursor
        parent_path = key_path[:-1]
        idx = self.cursor + direction
        while 0 <= idx < len(self._rows):
            rdepth, rfd, _, _, _, _, rkey_path = self._rows[idx]
            if rdepth < depth:
                break  # moved past parent — no more siblings
            if rdepth == depth and rfd is not None and rkey_path[:-1] == parent_path:
                return idx
            idx += direction
        return self.cursor  # no sibling found

    def _scroll_to(self, idx: int, list_rows: int) -> None:
        self.cursor = idx
        if self.cursor < self.scroll:
            self.scroll = self.cursor
        elif self.cursor >= self.scroll + list_rows:
            self.scroll = self.cursor - list_rows + 1

    def _jump_to_next_match(self, list_rows: int) -> None:
        """Move cursor to the next search match after the current position (wraps)."""
        if not self.search_matches:
            return
        for m in self.search_matches:
            if m > self.cursor:
                self._scroll_to(m, list_rows)
                return
        self._scroll_to(self.search_matches[0], list_rows)

    def _jump_to_prev_match(self, list_rows: int) -> None:
        """Move cursor to the previous search match before the current position (wraps)."""
        if not self.search_matches:
            return
        for m in reversed(self.search_matches):
            if m < self.cursor:
                self._scroll_to(m, list_rows)
                return
        self._scroll_to(self.search_matches[-1], list_rows)

    def handle_key(self, key: int, list_rows: int) -> str:
        n = len(self._rows)

        # ── Search mode ───────────────────────────────────────────────────────
        if self.search_mode:
            if key == 27:                                        # Escape – cancel
                self.search_mode = False
                self.search_query = ''
                self.search_matches = []
                self._search_match_set = set()
            elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):  # Enter – confirm
                self.search_mode = False
            elif key in (curses.KEY_BACKSPACE, 127, 8):          # Backspace
                self.search_query = self.search_query[:-1]
                self._update_search_matches()
                if self.search_matches:
                    self._scroll_to(self.search_matches[0], list_rows)
            elif 32 <= key <= 126:                               # printable character
                self.search_query += chr(key)
                self._update_search_matches()
                if self.search_matches:
                    self._scroll_to(self.search_matches[0], list_rows)
            return 'snapshot'

        if key == curses.KEY_UP:
            self.cursor = max(0, self.cursor - 1)
            if self.cursor < self.scroll:
                self.scroll = self.cursor

        elif key == curses.KEY_DOWN:
            self.cursor = min(n - 1, self.cursor + 1)
            if self.cursor >= self.scroll + list_rows:
                self.scroll = self.cursor - list_rows + 1

        elif key in (ord('['),):
            self._scroll_to(self._find_sibling(-1), list_rows)

        elif key in (ord(']'),):
            self._scroll_to(self._find_sibling(1), list_rows)

        elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
            if 0 <= self.cursor < n:
                depth, fd, sa, tb, is_exp, _, key_path = self._rows[self.cursor]
                if fd is not None:
                    self._toggle(key_path)
                    self._rebuild()

        elif key == curses.KEY_RIGHT:
            # Go to first child: expand if needed, then move cursor down one
            if 0 <= self.cursor < n:
                depth, fd, sa, tb, is_exp, _, key_path = self._rows[self.cursor]
                if fd is not None and sa:
                    if not is_exp:
                        self._toggle(key_path)
                        self._rebuild()
                    # After possible rebuild, cursor+1 is the first child
                    if self.cursor + 1 < len(self._rows):
                        self._scroll_to(self.cursor + 1, list_rows)

        elif key == ord('e'):
            # Recursively expand all descendants
            if 0 <= self.cursor < n:
                depth, fd, sa, tb, is_exp, _, key_path = self._rows[self.cursor]
                if fd is not None:
                    self._expand_recursive(key_path, sa, depth)
                    self._rebuild()

        elif key == ord('f'):
            # Focus: make the selected frame the new root.
            # Indentation resets to 0; 'q' pops back to the previous focus.
            if 0 <= self.cursor < n:
                depth, fd, sa, tb, is_exp, _, key_path = self._rows[self.cursor]
                if fd is not None and sa:
                    label = fd.get('name', '?')
                    self.focus_stack.append((
                        self.focus_sa, self.focus_level,
                        self.expanded_tree, self.cursor, self.scroll,
                        label,
                    ))
                    self.focus_sa    = sa
                    self.focus_level = depth + 1
                    self.expanded_tree = {}
                    self.cursor = 0
                    self.scroll = 0
                    self._rebuild()

        elif key == ord('r'):
            # Jump cursor to the first row (root of current tree)
            self.cursor = 0
            self.scroll = 0

        elif key == ord('c'):
            # Collapse all: clear expanded_tree so nothing is expanded
            self.expanded_tree = {}
            self.cursor = 0
            self.scroll = 0
            self._rebuild()

        elif key == curses.KEY_LEFT:
            # Go to parent: scan backward for first row with depth - 1
            if 0 <= self.cursor < n:
                depth = self._rows[self.cursor][0]
                if depth > self.focus_level:
                    for i in range(self.cursor - 1, -1, -1):
                        if self._rows[i][0] == depth - 1:
                            self._scroll_to(i, list_rows)
                            break

        elif key == 27:                                          # Escape – clear highlights
            self.search_query = ''
            self.search_matches = []
            self._search_match_set = set()

        elif key == ord('/'):
            self.search_mode = True
            self.search_query = ''
            self.search_matches = []
            self._search_match_set = set()

        elif key == ord('n'):
            self._jump_to_next_match(list_rows)

        elif key == ord('N'):
            self._jump_to_prev_match(list_rows)

        elif key in (ord('q'), ord('Q')):
            if self.focus_stack:
                prev = self.focus_stack.pop()
                (self.focus_sa, self.focus_level, self.expanded_tree,
                 self.cursor, self.scroll, _label) = prev
                self._rebuild()
            else:
                return 'timeline'

        return 'snapshot'

    def render(self, stdscr, height: int, width: int, extra_hint: str = ''):
        stdscr.erase()

        total_mem = sum(a.size for a in self.allocations)
        grand_total = total_mem or 1

        # Header – self.time_us is now an event index; look up the wall-clock time
        event_idx = int(self.time_us)
        tus = self.data.timeline_times_us
        if tus and 0 <= event_idx < len(tus):
            wall_us  = tus[event_idx]
            rel_us   = wall_us - tus[0]
            wall     = datetime.datetime.fromtimestamp(wall_us / 1e6).strftime('%Y-%m-%d %H:%M:%S.%f')
            time_tag = f"event {event_idx} / {wall} (+{fmt_time(rel_us)})"
        else:
            time_tag = f"event {max(0, event_idx)} (pre-recording baseline)"
        hdr = (f"Snapshot @ {time_tag}  |  "
               f"{len(self.allocations)} allocs  |  {fmt_bytes(total_mem)} total")
        safe_addstr(stdscr, 0, 0, hdr[:width - 1], curses.A_BOLD)

        # Breadcrumb line: focus path + selected row ancestry
        if self.focus_stack:
            focus_labels = [entry[5] for entry in self.focus_stack]
            focus_prefix = 'Focus: ' + ' > '.join(focus_labels) + '   '
        else:
            focus_prefix = ''
        crumb_line = 'Selected: (none)'
        if 0 <= self.cursor < len(self._rows):
            depth, fd, sa, tb, is_exp, _, key_path = self._rows[self.cursor]
            if key_path:
                crumb_line = 'Selected: ' + ' > '.join(k[0] or '?' for k in key_path)
        safe_addstr(stdscr, 1, 0, (focus_prefix + crumb_line)[:width - 1], curses.A_DIM)

        # Column header
        col_hdr = f"  {'Memory':>10}  {'%par':>5}  {'%tot':>5}  {'Allocs':>6}  Frame"
        safe_addstr(stdscr, 2, 0, col_hdr[:width - 1], curses.A_UNDERLINE)

        LIST_START = 3
        list_rows = height - LIST_START - 3   # 2 footer lines + 1 header
        if list_rows < 1:
            list_rows = 1

        for i, (depth, fd, sa, tb, is_exp, level_total, key_path) in enumerate(self._rows):
            if i < self.scroll:
                continue
            scr_row = LIST_START + (i - self.scroll)
            if scr_row >= LIST_START + list_rows:
                break

            pct_par = 100.0 * tb / (level_total or 1)
            pct_tot = 100.0 * tb / grand_total
            mem_s    = fmt_bytes(tb)
            pct_par_s = f'{pct_par:.1f}%'
            pct_tot_s = f'{pct_tot:.1f}%'
            count    = len(sa)
            indent_s = ' ' * (self.INDENT * (depth - self.focus_level))

            is_sel   = (i == self.cursor)
            is_match = bool(self.search_matches) and i in self._search_match_set
            heat     = curses.color_pair(heat_pair(tb / grand_total))
            sel_attr = curses.A_BOLD | curses.A_REVERSE

            if fd is None:
                # Leaf row: allocations with no further frames at this depth
                frame_s = f'({count} allocation{"s" if count != 1 else ""})'
                line = f'{indent_s}  {mem_s:>10}  {pct_par_s:>5}  {pct_tot_s:>5}  {count:>6}  {frame_s}'
                row_attr = (sel_attr | heat) if is_sel else (curses.A_DIM | heat)
                safe_addstr(stdscr, scr_row, 0, line[:width - 1], row_attr)
            else:
                fname = fd.get('name', '?')
                ffile = fd.get('filename', '')
                fline = fd.get('line', 0)
                has_children = any(len(a.frames) > depth + 1 for a in sa)

                if is_exp:
                    arrow = '▼'
                elif has_children:
                    arrow = '▶'
                else:
                    arrow = ' '

                prefix = f'{indent_s}{arrow} '
                # Fixed-width columns: prefix + mem(10) + pct_par(7) + pct_tot(7) + count(8) + gap(2)
                fixed_w = len(prefix) + 10 + 7 + 7 + 8 + 2
                frame_s = f'{fname}  ({ffile}:{fline})'
                max_frame_w = max(8, width - fixed_w - 2)
                if len(frame_s) > max_frame_w:
                    trim = len(frame_s) - max_frame_w + 3
                    frame_s = '...' + frame_s[trim:]

                line = f'{prefix}{mem_s:>10}  {pct_par_s:>5}  {pct_tot_s:>5}  {count:>6}  {frame_s}'
                if is_sel:
                    row_attr = sel_attr | heat
                elif is_match:
                    row_attr = curses.color_pair(heat_pair_match(tb / grand_total))
                else:
                    row_attr = heat
                safe_addstr(stdscr, scr_row, 0, line[:width - 1], row_attr)

        # Scrollbar
        n = len(self._rows)
        if n > list_rows:
            pct_pos = self.scroll / max(1, n - list_rows)
            safe_addch(stdscr, LIST_START + int(pct_pos * (list_rows - 1)), width - 1, '█', curses.A_DIM)

        # Footer / search bar (two lines)
        ctrl1 = ' ↑↓: Navigate   [ or ]: Prev/next sibling   →: First child   ←: Parent   Enter: Toggle   e: Expand all   c: Collapse all'
        ctrl2 = ' r: Root   f: Focus   /: Search   n/N: Next/prev match   Esc: Clear highlights   q: Unfocus/Back'
        if self.search_mode:
            n_matches = len(self.search_matches)
            if not self.search_query:
                hint = '  (type to search — Enter: confirm  Esc: cancel)'
            elif n_matches:
                hint = f'  ({n_matches} match{"es" if n_matches != 1 else ""} — Enter: confirm  Esc: cancel)'
            else:
                hint = '  (no matches — Esc: cancel)'
            bar = f'/{self.search_query}{hint}'
            safe_addstr(stdscr, height - 2, 0, ' ' * (width - 1), curses.A_REVERSE)
            safe_addstr(stdscr, height - 2, 0, bar[:width - 1],
                        curses.A_REVERSE | (0 if n_matches or not self.search_query
                                            else curses.color_pair(4)))
        else:
            safe_addstr(stdscr, height - 2, 0, (ctrl1 + extra_hint)[:width - 1], curses.A_REVERSE)
        safe_addstr(stdscr, height - 1, 0, (ctrl2 + extra_hint)[:width - 1], curses.A_REVERSE)

        stdscr.noutrefresh()


# ─── Main App Loop ────────────────────────────────────────────────────────────

def _init_colors() -> None:
    """Initialize all curses color pairs (call once after start_color)."""
    curses.start_color()
    curses.use_default_colors()
    # Color pairs: (fg, bg), -1 = terminal default
    curses.init_pair(1, curses.COLOR_CYAN,   -1)   # normal bar
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # selected column
    curses.init_pair(3, curses.COLOR_GREEN,  -1)   # positive / info
    curses.init_pair(4, curses.COLOR_RED,    -1)   # warning
    # Heat-map pairs for snapshot view: 5=blue(cold) … 9=red(hot)
    curses.init_pair(5, curses.COLOR_BLUE,   -1)
    curses.init_pair(6, curses.COLOR_CYAN,   -1)
    curses.init_pair(7, curses.COLOR_GREEN,  -1)
    curses.init_pair(8, curses.COLOR_YELLOW, -1)
    curses.init_pair(9, curses.COLOR_RED,    -1)
    # Heat pairs with white background for search match highlighting (10–14)
    curses.init_pair(10, curses.COLOR_BLUE,   curses.COLOR_WHITE)
    curses.init_pair(11, curses.COLOR_CYAN,   curses.COLOR_WHITE)
    curses.init_pair(12, curses.COLOR_GREEN,  curses.COLOR_WHITE)
    curses.init_pair(13, curses.COLOR_YELLOW, curses.COLOR_WHITE)
    curses.init_pair(14, curses.COLOR_RED,    curses.COLOR_WHITE)


def main(stdscr, data: MemoryProfileData):
    curses.curs_set(0)
    _init_colors()
    stdscr.keypad(True)

    timeline = TimelineView(data)
    snapshot: SnapshotView | None = None
    view = 'timeline'

    while True:
        height, width = stdscr.getmaxyx()

        if view == 'timeline':
            timeline.render(stdscr, height, width)
            curses.doupdate()
            key = stdscr.getch()
            result = timeline.handle_key(key, width - TimelineView.Y_AXIS_W - 1)
            if result == 'snapshot':
                snapshot = SnapshotView(data, timeline.current_time_us())
                view = 'snapshot'
            elif result == 'quit':
                break

        elif view == 'snapshot':
            assert snapshot is not None
            snapshot.render(stdscr, height, width)
            curses.doupdate()
            key = stdscr.getch()
            list_rows = max(1, height - 5)
            result = snapshot.handle_key(key, list_rows)
            if result == 'timeline':
                view = 'timeline'
            elif result == 'quit':
                break


def main_compare(stdscr, data1: MemoryProfileData, data2: MemoryProfileData):
    """Split-screen compare mode: two independent panes, vertical or horizontal split."""
    curses.curs_set(0)
    _init_colors()
    stdscr.keypad(True)

    datas     = [data1, data2]
    timelines = [TimelineView(data1), TimelineView(data2)]
    snapshots: list = [None, None]
    views     = ['timeline', 'timeline']
    active    = 0       # 0 = first pane, 1 = second pane
    vertical  = True    # True = left/right split, False = top/bottom split

    wins      = [None, None]
    prev_state = (0, 0, True)   # (height, width, vertical)

    while True:
        height, width = stdscr.getmaxyx()

        if vertical:
            sep     = (width - 1) // 2
            pane_h  = [height, height]
            pane_w  = [sep, width - sep - 1]
            pane_y  = [0, 0]
            pane_x  = [0, sep + 1]
        else:
            sep     = (height - 1) // 2
            pane_h  = [sep, height - sep - 1]
            pane_w  = [width, width]
            pane_y  = [0, sep + 1]
            pane_x  = [0, 0]

        # Recreate subwindows when terminal size or split direction changes
        if (height, width, vertical) != prev_state:
            for p in range(2):
                wins[p] = curses.newwin(max(1, pane_h[p]), max(1, pane_w[p]),
                                        pane_y[p], pane_x[p])
                wins[p].keypad(True)
            prev_state = (height, width, vertical)

        # Draw separator on stdscr
        stdscr.erase()
        if vertical:
            for row in range(height - 1):
                safe_addch(stdscr, row, sep, '│', curses.A_DIM)
            indicator = '◀' if active == 0 else '▶'
            safe_addch(stdscr, height // 2, sep, indicator, curses.A_BOLD)
        else:
            for col in range(width):
                safe_addch(stdscr, sep, col, '─', curses.A_DIM)
            indicator = '▲' if active == 0 else '▼'
            safe_addch(stdscr, sep, width // 2, indicator, curses.A_BOLD)
        stdscr.noutrefresh()

        # Render each pane into its subwindow
        compare_hint = '   { or }: Switch pane   s: Toggle split direction'
        for p in range(2):
            win = wins[p]
            win.bkgd(' ', curses.A_DIM if p != active else 0)
            if views[p] == 'timeline':
                timelines[p].render(win, pane_h[p], pane_w[p], extra_hint=compare_hint)
            elif views[p] == 'snapshot' and snapshots[p] is not None:
                snapshots[p].render(win, pane_h[p], pane_w[p], extra_hint=compare_hint)

        curses.doupdate()
        key = stdscr.getch()

        # Global compare-mode keys (intercepted before pane dispatch)
        active_snap = snapshots[active]
        in_search = (views[active] == 'snapshot' and
                     active_snap is not None and
                     active_snap.search_mode)
        if not in_search:
            if key in (ord('{'), ord('}')):
                active = 1 - active
                continue
            if key == ord('s'):
                vertical = not vertical
                prev_state = (0, 0, vertical)   # force window recreation
                for tl in timelines:
                    tl._auto_fit_done = False   # re-fit to new pane width
                continue

        # Dispatch key to the active pane
        ph, pw = pane_h[active], pane_w[active]
        if views[active] == 'timeline':
            result = timelines[active].handle_key(key, pw - TimelineView.Y_AXIS_W - 1)
            if result == 'snapshot':
                snapshots[active] = SnapshotView(datas[active],
                                                 timelines[active].current_time_us())
                views[active] = 'snapshot'
            elif result == 'quit':
                break
        elif views[active] == 'snapshot':
            list_rows = max(1, ph - 5)
            result = snapshots[active].handle_key(key, list_rows)
            if result == 'timeline':
                views[active] = 'timeline'
            elif result == 'quit':
                break


__version__ = '0.1.0'


def run():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] in ('-v', '--version'):
        print(f'ptmem {__version__}')
        sys.exit(0)

    # ── Compare mode: ptmem -c a.pkl b.pkl ────────────────────────────────────
    if sys.argv[1] in ('-c', '--compare'):
        if len(sys.argv) < 4:
            print('Usage: ptmem -c <snapshot1.pkl> <snapshot2.pkl>')
            sys.exit(1)
        paths = [sys.argv[2], sys.argv[3]]
        datas = []
        for p in paths:
            print(f'Loading {p} ...', end=' ', flush=True)
            try:
                d = MemoryProfileData(p)
            except FileNotFoundError:
                print(f'Error: file not found: {p}')
                sys.exit(1)
            except Exception as e:
                print(f'Error loading snapshot: {e}')
                sys.exit(1)
            print('(from cache)' if d.from_cache else '(parsed, cache saved)')
            datas.append(d)
        print('Starting compare viewer...')
        curses.wrapper(main_compare, datas[0], datas[1])
        return

    # ── Single mode ───────────────────────────────────────────────────────────
    path = sys.argv[1]
    print(f'Loading {path} ...', end=' ', flush=True)

    try:
        data = MemoryProfileData(path)
    except FileNotFoundError:
        print(f'Error: file not found: {path}')
        sys.exit(1)
    except Exception as e:
        print(f'Error loading snapshot: {e}')
        sys.exit(1)

    if not data.timeline_times:
        print('No timed memory events found in this snapshot.')
        print()
        print('Make sure you recorded memory history before dumping:')
        print('    torch.cuda.memory._record_memory_history(max_entries=100000)')
        print('    # ... run your model ...')
        print('    torch.cuda.memory._dump_snapshot("snapshot.pkl")')
        sys.exit(1)

    print('(from cache)' if data.from_cache else '(parsed, cache saved)')
    n_allocs = len(data.allocations)
    span = fmt_time(data.max_time - data.min_time)
    peak = fmt_bytes(data.max_memory)
    print(f'{n_allocs} allocations over {span} (peak: {peak})')
    print('Starting interactive viewer...')

    curses.wrapper(main, data)


if __name__ == '__main__':
    run()
