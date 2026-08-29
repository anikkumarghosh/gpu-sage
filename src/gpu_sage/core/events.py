"""Event queue implementation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .models import SimulationEvent


@dataclass
class EventQueue:
    """Min-heap ordered by simulation time, then insertion order."""

    _heap: list[tuple[float, int, SimulationEvent]]
    _sequence: int = 0

    def __init__(self) -> None:
        self._heap = []
        self._sequence = 0

    def push(self, time: float, event_type: str, job_id: int) -> None:
        event = SimulationEvent(
            time=time,
            sequence=self._sequence,
            event_type=event_type,
            job_id=job_id,
        )
        self._sequence += 1
        heapq.heappush(self._heap, (event.time, event.sequence, event))

    def pop(self) -> SimulationEvent:
        if not self._heap:
            raise IndexError("Cannot pop from an empty event queue")
        return heapq.heappop(self._heap)[2]

    def peek(self) -> SimulationEvent | None:
        return self._heap[0][2] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
