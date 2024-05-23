from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class UsageInterface(Protocol):
    """An interface for a class that stores usage statistics."""

    n_calls: int
    """The number of calls to the model's forward pass."""

    n_tokens_context: int
    """The number of tokens in the context."""

    n_tokens_output: int
    """The number of tokens in the output."""

    def __add__(self, other: UsageInterface | None) -> UsageInterface: ...

    def __radd__(self, other: UsageInterface | None) -> UsageInterface: ...

    def __iadd__(self, other: UsageInterface | None) -> UsageInterface: ...

    def to_json(self) -> dict[str, int]: ...
