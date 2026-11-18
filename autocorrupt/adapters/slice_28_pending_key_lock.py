"""
Slice 28: Atomic Pending-Key Lock & Virtual Echo Filter.
Buffers physical user keystrokes into a FIFO queue during the ~0.3 ms window when
a synthetic burst rewrite is executing, then replays them in exact chronological order
so fast typing at 120+ WPM never interleaves with synthetic backspaces.
"""
from __future__ import annotations
from collections import deque
import threading
from autocorrupt.core.atoms.slice_00_types import KeyEvent, RewriteAction


class PendingKeyLock:
    """
    Thread-safe lock and FIFO keystroke replay queue for zero-glitch burst rewriting.
    Also records the last applied RewriteAction so the user can hit the Panic Undo
    hotkey (`Ctrl+Shift+Backspace`) to immediately revert a corruption.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rewriting = False
        self._pending_keys: deque[KeyEvent] = deque()
        self.passthrough_paused: bool = False
        self.last_undo_delta: tuple[int, str] | None = None

    def try_enqueue_if_rewriting(self, event: KeyEvent) -> bool:
        """
        If a burst rewrite is currently in flight, queues `event` and returns True
        (meaning caller should wait to process it until the burst finishes).
        """
        with self._lock:
            if self._rewriting:
                self._pending_keys.append(event)
                return True
            return False

    def begin_rewrite(self) -> None:
        with self._lock:
            self._rewriting = True

    def end_rewrite_and_drain(self) -> list[KeyEvent]:
        """Ends the active rewrite burst and returns any physical keys buffered during the burst."""
        with self._lock:
            self._rewriting = False
            drained = list(self._pending_keys)
            self._pending_keys.clear()
            return drained

    def record_for_undo(self, original_tail: str, action: RewriteAction) -> None:
        """
        Stores the inverse operation `(backspaces_to_undo, original_text_to_restore)`
        so `Ctrl+Shift+Backspace` can revert the last auto-corruption in < 0.3 ms.
        """
        self.last_undo_delta = (len(action.inject_text), original_tail)

    def pop_undo_action(self) -> tuple[int, str] | None:
        delta = self.last_undo_delta
        self.last_undo_delta = None
        return delta
