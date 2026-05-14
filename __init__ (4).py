"""
chimera/social/zscore.py
Rolling mention tracker and Z-score engine.

The Z-score answers: "How abnormal is the current mention rate
compared to the recent baseline?"

Formula:
    Z = (current_rate - μ_baseline) / σ_baseline

Where:
    current_rate   = mentions in the last SHORT_WINDOW minutes
    μ_baseline     = mean of hourly mention counts over BASELINE_HOURS
    σ_baseline     = std  of hourly mention counts over BASELINE_HOURS

A Z-score of 2.0 means the current mention rate is two standard
deviations above the recent average — a meaningful crowd signal.
The Sp score uses this value (normalised to 0-5 then clipped to 0-1).

Design notes:
  - One MentionWindow per tracked symbol.
  - Mentions are timestamped; old ones expire automatically on each call.
  - Thread-safe via a simple lock (written from async context, single writer).
  - The engine keeps a separate BASELINE deque of (window, count) snapshots
    taken every SNAPSHOT_INTERVAL minutes, from which μ and σ are derived.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any


# ── Configuration defaults ─────────────────────────────────────────────────────
SHORT_WINDOW_MINUTES  = 5     # current rate window
BASELINE_HOURS        = 24    # how far back the baseline looks
SNAPSHOT_INTERVAL_MIN = 10    # how often a baseline snapshot is taken
MIN_BASELINE_POINTS   = 6     # minimum snapshots before Z-score is meaningful
SPIKE_THRESHOLD       = 2.0   # Z-score above this is flagged as a spike


class MentionWindow:
    """
    Tracks raw mention timestamps for a single symbol.
    Exposes:
      .add(ts)          — record one mention
      .count_recent(minutes) — mentions in last N minutes
      .snapshot()       — take a baseline snapshot (call periodically)
      .zscore()         — current Z-score (float, or None if insufficient data)
      .is_spike()       — bool
    """

    def __init__(
        self,
        symbol:              str,
        short_window_min:    int   = SHORT_WINDOW_MINUTES,
        baseline_hours:      int   = BASELINE_HOURS,
        snapshot_interval:   int   = SNAPSHOT_INTERVAL_MIN,
        min_baseline_points: int   = MIN_BASELINE_POINTS,
        spike_threshold:     float = SPIKE_THRESHOLD,
    ):
        self.symbol             = symbol
        self.short_window_min   = short_window_min
        self.baseline_hours     = baseline_hours
        self.snapshot_interval  = snapshot_interval
        self.min_baseline_pts   = min_baseline_points
        self.spike_threshold    = spike_threshold

        # Rolling mention timestamps — expire after baseline_hours
        self._mentions: deque[datetime] = deque()
        # Baseline snapshots: (snapshot_dt, count_in_short_window)
        self._baseline: deque[tuple[datetime, int]] = deque()
        self._last_snapshot: datetime | None = None
        self._lock = threading.RLock()   # reentrant — snapshot() calls count_recent() under lock

    # ── Public API ─────────────────────────────────────────────────────────

    def add(self, ts: datetime | None = None) -> None:
        """Record one mention at the given timestamp (default: now UTC)."""
        if ts is None:
            ts = _now()
        with self._lock:
            self._mentions.append(ts)
            self._expire_mentions()

    def add_bulk(self, timestamps: list[datetime]) -> None:
        """Record multiple mentions at once (used when loading a page of results)."""
        with self._lock:
            for ts in timestamps:
                self._mentions.append(ts)
            self._expire_mentions()

    def count_recent(self, minutes: int | None = None) -> int:
        """Count mentions in the last N minutes (default: short_window_min)."""
        minutes = minutes or self.short_window_min
        cutoff  = _now() - timedelta(minutes=minutes)
        with self._lock:
            self._expire_mentions()
            return sum(1 for ts in self._mentions if ts >= cutoff)

    def snapshot(self) -> None:
        """
        Take a baseline snapshot of the current short-window count.
        Should be called every SNAPSHOT_INTERVAL minutes by the scraper.
        Snapshots older than baseline_hours are auto-expired.
        """
        now = _now()
        with self._lock:
            # Don't snapshot more frequently than the interval
            if (self._last_snapshot is not None and
                    (now - self._last_snapshot).total_seconds() < self.snapshot_interval * 60 * 0.9):
                return
            count = self.count_recent(self.short_window_min)
            self._baseline.append((now, count))
            self._last_snapshot = now
            # Expire old baseline points
            cutoff = now - timedelta(hours=self.baseline_hours)
            while self._baseline and self._baseline[0][0] < cutoff:
                self._baseline.popleft()

    def zscore(self) -> float | None:
        """
        Compute Z-score of the current mention rate vs the baseline.
        Returns None if insufficient baseline data.
        """
        with self._lock:
            if len(self._baseline) < self.min_baseline_pts:
                return None
            counts = [c for _, c in self._baseline]
            mu     = _mean(counts)
            sigma  = _std(counts)
            if sigma < 1e-9:
                # No variance — return 0 if current matches mean, else large value
                current = self.count_recent(self.short_window_min)
                return 0.0 if current == mu else float(self.spike_threshold + 1)
            current = self.count_recent(self.short_window_min)
            return round((current - mu) / sigma, 3)

    def is_spike(self) -> bool:
        z = self.zscore()
        return z is not None and z >= self.spike_threshold

    def stats(self) -> dict[str, Any]:
        """Full stats dict for logging and state injection."""
        z = self.zscore()
        return {
            "symbol":          self.symbol,
            "mentions_recent": self.count_recent(self.short_window_min),
            "mentions_1h":     self.count_recent(60),
            "baseline_points": len(self._baseline),
            "zscore":          z,
            "is_spike":        self.is_spike(),
            "short_window_min": self.short_window_min,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _expire_mentions(self) -> None:
        """Remove mentions older than baseline_hours (called under lock)."""
        cutoff = _now() - timedelta(hours=self.baseline_hours)
        while self._mentions and self._mentions[0] < cutoff:
            self._mentions.popleft()


class ZScoreEngine:
    """
    Manages a MentionWindow for every tracked symbol.
    Single instance shared across the social pipeline.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self._defaults = {
            "short_window_min":    cfg.get("social_short_window_min",    SHORT_WINDOW_MINUTES),
            "baseline_hours":      cfg.get("social_baseline_hours",      BASELINE_HOURS),
            "snapshot_interval":   cfg.get("social_snapshot_interval",   SNAPSHOT_INTERVAL_MIN),
            "min_baseline_points": cfg.get("social_min_baseline_points", MIN_BASELINE_POINTS),
            "spike_threshold":     cfg.get("social_spike_threshold",     SPIKE_THRESHOLD),
        }
        self._windows: dict[str, MentionWindow] = {}
        self._lock = threading.RLock()   # reentrant — snapshot() calls count_recent() under lock

    def track(self, symbol: str) -> MentionWindow:
        """Return (or create) the MentionWindow for a symbol."""
        with self._lock:
            if symbol not in self._windows:
                self._windows[symbol] = MentionWindow(symbol, **self._defaults)
            return self._windows[symbol]

    def add_mention(self, symbol: str, ts: datetime | None = None) -> None:
        self.track(symbol).add(ts)

    def add_mentions_bulk(self, symbol: str, timestamps: list[datetime]) -> None:
        self.track(symbol).add_bulk(timestamps)

    def zscore(self, symbol: str) -> float | None:
        with self._lock:
            if symbol not in self._windows:
                return None
        return self._windows[symbol].zscore()

    def snapshot_all(self) -> None:
        """Take baseline snapshots for all tracked symbols."""
        with self._lock:
            windows = list(self._windows.values())
        for w in windows:
            w.snapshot()

    def all_stats(self) -> dict[str, dict]:
        with self._lock:
            syms = list(self._windows.keys())
        return {s: self._windows[s].stats() for s in syms}

    def inject_into_state(self, state: Any) -> None:
        """
        Write current Z-scores into state.market.stocks[sym]["social_zscore"].
        Called by the scraper after each fetch cycle.
        """
        with self._lock:
            windows = dict(self._windows)
        for sym, window in windows.items():
            z = window.zscore()
            if z is None:
                continue
            # Ensure the symbol exists in stocks market data
            if sym not in state.market.stocks:
                state.market.stocks[sym] = {
                    "close": [], "high": [], "low": [], "volume": [],
                    "short_interest": 0.0, "rvol": 1.0, "social_zscore": 0.0,
                }
            state.market.stocks[sym]["social_zscore"] = z


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
