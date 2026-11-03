"""Measured timings; model reciprocal latency is distinct from throughput."""
from collections import deque
import time

import numpy as np


class TimingWindow:
    def __init__(self, size=120):
        self.rows = deque(maxlen=size)
        self.total = 0
        self.started = None

    def add(self, info, now=None):
        now = time.monotonic() if now is None else now
        if self.started is None:
            self.started = now - info.get('pipeline_ms', 0) / 1000
        self.total += 1
        self.rows.append((now, dict(info)))

    def report(self):
        result = {'processed_frames': self.total, 'window_frames': len(self.rows)}
        for key in ('yolo_ms', 'classifier_ms', 'pipeline_ms', 'freshness_ms'):
            values = [r[key] for _, r in self.rows if r.get(key) is not None and np.isfinite(r[key])]
            result[key] = None if not values else dict(mean=float(np.mean(values)),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)))
        for name, key in [('yolo_fps', 'yolo_ms'), ('classifier_fps', 'classifier_ms')]:
            stats = result[key]
            result[name] = 1000 / stats['mean'] if stats and stats['mean'] > 0 else None
        result['pipeline_fps'] = ((len(self.rows)-1)/(self.rows[-1][0]-self.rows[0][0])
            if len(self.rows) > 1 and self.rows[-1][0] > self.rows[0][0] else None)
        result['wall_fps'] = (self.total / (self.rows[-1][0]-self.started)
            if self.rows and self.rows[-1][0] > self.started else None)
        return result


class DisplayRate:
    """Count unique frames actually painted by Qt, using a two-second window."""
    def __init__(self):
        self.ticks = deque()
        self.total = 0
        self.first_tick = self.last_tick = None

    def painted(self, now=None):
        now = time.monotonic() if now is None else now
        self.total += 1
        if self.first_tick is None:
            self.first_tick = now
        self.last_tick = now
        self.ticks.append(now)
        self._expire(now)

    def _expire(self, now):
        while self.ticks and self.ticks[0] < now - 2:
            self.ticks.popleft()

    def fps(self, now=None):
        now = time.monotonic() if now is None else now
        self._expire(now)
        return ((len(self.ticks)-1)/(self.ticks[-1]-self.ticks[0])
                if len(self.ticks)>1 and self.ticks[-1]>self.ticks[0] else None)

    def average_fps(self):
        return ((self.total-1)/(self.last_tick-self.first_tick)
                if self.total>1 and self.last_tick>self.first_tick else None)
