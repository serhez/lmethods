import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class GuardedValue:
    """A value guarded by a lock"""

    value: Any
    """The value to be guarded."""

    lock: threading.Lock = threading.Lock()
    """The lock to be acquired before accessing the value."""
