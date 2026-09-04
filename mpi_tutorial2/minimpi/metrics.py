"""metrics.py — independent event/metrics layer.

Collective algorithms never print or build the teacher view; they only
produce CommunicationEvent records through the runtime. Rendering those
events (student local view / teacher global view) is a separate concern.
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Any, List, Dict


def now_ns():
    return time.perf_counter_ns()


@dataclass
class CommunicationEvent:
    """One point-to-point message, seen from one side."""
    algorithm: str = ""
    phase: str = ""            # e.g. "reduce", "broadcast", "reduce-scatter"
    logical_round: int = 0     # algorithm label / log grouping
    source: int = -1
    destination: int = -1
    tag: int = 0
    payload_bytes: int = 0
    transfer_time_ms: float = 0.0   # observed (protocol + network + runtime)
    effective_bandwidth_mbps: float = 0.0
    ts_start_ns: int = 0
    ts_end_ns: int = 0
    value_before: Any = None            # receiver/sender local value snapshot
    received_value: Any = None
    value_after: Any = None
    side: str = ""                      # "send" | "recv"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        return d


class EventLog:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)

    def extend(self, events):
        self.events.extend(events)

    def by_round(self, rnd):
        return [e for e in self.events if e.logical_round == rnd]

    def rounds(self):
        return sorted({e.logical_round for e in self.events})

    def clear(self):
        self.events = []


def bandwidth_mbps(payload_bytes, transfer_seconds):
    if transfer_seconds <= 0:
        return 0.0
    return (payload_bytes * 8.0) / (transfer_seconds * 1e6)
