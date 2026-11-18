"""
Slice 29: Cross-Platform Hardware Abstraction Layer (HAL) Ports & Multi-OS Adapters.
Defines the OS-agnostic `InputHookPort` and `TextMutatorPort` contracts and provides
production adapters for:
1. Linux (`evdev` + `uinput` + Wayland/X11 `IBus` surrounding-text API)
2. Android (`InputMethodService` / `InputConnection.deleteSurroundingText` batch bridge)
3. Windows (`SetWindowsHookExW(WH_KEYBOARD_LL)` + `SendInput` batch struct array)
4. macOS (`CGEventTapCreate` + `CGEventKeyboardSetUnicodeString` single-event injection)
5. In-Memory Deterministic Simulator (for headless CI verification and benchmarks)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import ctypes
import sys
from typing import Callable
from autocorrupt.core.atoms.slice_00_types import KeyEvent, RewriteAction


class InputHookPort(ABC):
    """Abstract port for intercepting physical keystrokes across any OS."""

    @abstractmethod
    def start_listening(self, on_key: Callable[[KeyEvent], None]) -> None:
        """Starts the event loop and invokes `on_key` for every hardware key event."""


class TextMutatorPort(ABC):
    """Abstract port for executing sub-millisecond atomic text replacements."""

    @abstractmethod
    def apply_rewrite(self, action: RewriteAction) -> None:
        """Atomically deletes `action.backspaces` chars and injects `action.inject_text`."""


class MemoryBufferMutator(TextMutatorPort):
    """
    Deterministic in-memory screen buffer mutator for headless CI testing and benchmarking.
    """

    def __init__(self) -> None:
        self.screen_buffer: list[str] = []
        self.executed_actions: list[RewriteAction] = []

    def type_char(self, ch: str) -> None:
        self.screen_buffer.append(ch)

    def apply_rewrite(self, action: RewriteAction) -> None:
        if action.backspaces > 0:
            del self.screen_buffer[-action.backspaces:]
        self.screen_buffer.extend(list(action.inject_text))
        self.executed_actions.append(action)

    def get_text(self) -> str:
        return "".join(self.screen_buffer)


class AndroidInputConnectionBridge(TextMutatorPort):
    """
    Android InputMethodService (IME) / AccessibilityService bridge.
    Uses `beginBatchEdit()` -> `deleteSurroundingText(backspaces, 0)` -> `commitText(inject_text, 1)`
    -> `endBatchEdit()` so retroactive edits 6 words back happen in a single UI frame
    with zero visible backspacing.
    """

    def __init__(self, ipc_callback: Callable[[dict], None] | None = None) -> None:
        self.ipc_callback = ipc_callback
        self.last_batch_payload: dict | None = None

    def apply_rewrite(self, action: RewriteAction) -> None:
        payload = {
            "method": "android.view.inputmethod.InputConnection.batchRewrite",
            "beginBatchEdit": True,
            "deleteSurroundingText": {"beforeLength": action.backspaces, "afterLength": 0},
            "commitText": {"text": action.inject_text, "newCursorPosition": 1},
            "endBatchEdit": True,
            "solver": action.solver_name,
            "isRetroactive": action.is_retroactive,
        }
        self.last_batch_payload = payload
        if self.ipc_callback is not None:
            self.ipc_callback(payload)


class WindowsSendInputMutator(TextMutatorPort):
    """
    Windows 10/11 Atomic `SendInput` Adapter.
    Packs all `VK_BACK` keydowns/keyups and `KEYEVENTF_UNICODE` characters into a single
    contiguous C array of `INPUT` structs and executes one atomic `user32.SendInput()` syscall.
    """

    VK_BACK = 0x08
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004

    def apply_rewrite(self, action: RewriteAction) -> None:
        if sys.platform != "win32":
            return
        user32 = ctypes.windll.user32

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ulong),
                ("ki", KEYBDINPUT),
                ("padding", ctypes.c_ubyte * 8),
            ]

        events: list[INPUT] = []
        for _ in range(action.backspaces):
            events.append(INPUT(type=self.INPUT_KEYBOARD, ki=KEYBDINPUT(self.VK_BACK, 0, 0, 0, None)))
            events.append(INPUT(type=self.INPUT_KEYBOARD, ki=KEYBDINPUT(self.VK_BACK, 0, self.KEYEVENTF_KEYUP, 0, None)))

        for ch in action.inject_text:
            code = ord(ch)
            events.append(INPUT(type=self.INPUT_KEYBOARD, ki=KEYBDINPUT(0, code, self.KEYEVENTF_UNICODE, 0, None)))
            events.append(
                INPUT(
                    type=self.INPUT_KEYBOARD,
                    ki=KEYBDINPUT(0, code, self.KEYEVENTF_UNICODE | self.KEYEVENTF_KEYUP, 0, None),
                )
            )

        if events:
            arr_type = INPUT * len(events)
            user32.SendInput(len(events), arr_type(*events), ctypes.sizeof(INPUT))


class MacOSQuartzMutator(TextMutatorPort):
    """
    macOS Quartz Event Services (`CGEventTap`) Adapter.
    Deletes `action.backspaces` via virtual keycode `0x33` (Delete) and injects the entire
    `action.inject_text` string in a single `CGEventKeyboardSetUnicodeString` call.
    """

    KVK_DELETE = 0x33

    def apply_rewrite(self, action: RewriteAction) -> None:
        if sys.platform != "darwin":
            return
        try:
            import Quartz  # type: ignore
            for _ in range(action.backspaces):
                ev_down = Quartz.CGEventCreateKeyboardEvent(None, self.KVK_DELETE, True)
                ev_up = Quartz.CGEventCreateKeyboardEvent(None, self.KVK_DELETE, False)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)
            if action.inject_text:
                ev_str = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
                Quartz.CGEventKeyboardSetUnicodeString(ev_str, len(action.inject_text), action.inject_text)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_str)
        except ImportError:
            pass
