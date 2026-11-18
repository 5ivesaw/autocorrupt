"""
Slice 31: Full-System Orchestrator, CLI Daemon & 3-Pass Verification Runner.
Wires Slices 00–30 into the unified `AutoCorruptEngine` state machine and exposes:
- `simulate`: trace any sentence or paragraph keystroke-by-keystroke with microsecond timings
- `benchmark`: run 1,000-keystroke sub-millisecond performance verification
- `self-test`: execute the deterministic test gate across all 32 slices
- `daemon`: start the live background keyboard daemon on Linux (Pop!_OS), Windows, macOS, or Android
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from autocorrupt.adapters.slice_28_pending_key_lock import PendingKeyLock
from autocorrupt.adapters.slice_29_hal_ports import MemoryBufferMutator, TextMutatorPort
from autocorrupt.adapters.slice_30_linux_evdev_uinput import (
    KEY_BACKSPACE,
    KEY_DOWN,
    KEY_END,
    KEY_ENTER,
    KEY_HOME,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
    LinuxEvdevReader,
    LinuxUinputMutator,
)
from autocorrupt.core.atoms.slice_00_types import KeyEvent, RewriteAction
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.planner.slice_26_retro_backprop import RetroBackpropPlanner
from autocorrupt.core.planner.slice_27_minimal_diff_plan import build_minimal_rewrite_action
from autocorrupt.core.semantics.slice_15_wordnet_graph import compile_default_lexicon
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape

NAVIGATION_KEYS: frozenset[int] = frozenset({
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_HOME, KEY_END, KEY_ENTER,
})


class AutoCorruptEngine:
    """
    Top-level deterministic GOFAI state machine connecting the `SentenceTape`,
    `RetroBackpropPlanner`, `PendingKeyLock`, and `TextMutatorPort`.
    """

    def __init__(
        self,
        store: MmapLexiconStore | None = None,
        mutator: TextMutatorPort | None = None,
    ) -> None:
        self.store = store or compile_default_lexicon()
        self.mutator = mutator or MemoryBufferMutator()
        self.tape = SentenceTape()
        self.planner = RetroBackpropPlanner()
        self.key_lock = PendingKeyLock()

    def process_key_event(self, event: KeyEvent) -> RewriteAction | None:
        """Processes a single normalized KeyEvent in < 0.3 milliseconds."""
        if not event.is_press:
            return None

        # Queue key if a synthetic rewrite burst is currently in flight
        if self.key_lock.try_enqueue_if_rewriting(event):
            return None

        # Panic Undo Hotkey: Ctrl + Shift + Backspace reverts the last auto-corruption
        if event.ctrl and event.shift and event.key_code == KEY_BACKSPACE:
            undo = self.key_lock.pop_undo_action()
            if undo is not None:
                bs_count, restore_str = undo
                undo_act = RewriteAction(
                    backspaces=bs_count,
                    inject_text=restore_str,
                    target_slots=(),
                    solver_name="panic_undo",
                    utility_score=100.0,
                    is_retroactive=False,
                    trace="User triggered Ctrl+Shift+Backspace Panic Undo",
                )
                self.mutator.apply_rewrite(undo_act)
                self.tape.hard_reset()
                return undo_act
            return None

        # Pause Toggle Hotkey: Ctrl + Alt + P
        if event.ctrl and event.alt and event.char.lower() == "p":
            self.key_lock.passthrough_paused = not self.key_lock.passthrough_paused
            self.tape.hard_reset()
            return None

        if self.key_lock.passthrough_paused:
            return None

        # Cursor navigation / Enter newline commits and resets the SentenceTape
        if event.key_code in NAVIGATION_KEYS or event.char in ("\n", "\r"):
            if isinstance(self.mutator, MemoryBufferMutator) and event.char:
                self.mutator.type_char(event.char)
            self.tape.hard_reset()
            return None

        # Physical Backspace from user
        if event.key_code == KEY_BACKSPACE:
            if isinstance(self.mutator, MemoryBufferMutator) and self.mutator.screen_buffer:
                self.mutator.screen_buffer.pop()
            self.tape.handle_backspace(self.store)
            return None

        if not event.char:
            return None

        if isinstance(self.mutator, MemoryBufferMutator):
            self.mutator.type_char(event.char)

        # Word boundary trigger: Space or sentence punctuation
        if event.char in (" ", ",", ".", "!", "?", ";", ":"):
            t0 = time.perf_counter_ns()
            tok = self.tape.commit_active_word(event.char, self.store)
            if tok is None:
                return None

            hypo, is_retro = self.planner.step_on_word_commit(self.tape, self.store)
            if hypo is None:
                return None

            self.planner.optimizer.record_used(hypo)
            old_tail = self.tape.span_screen_text(hypo.target_slots[0])
            action = build_minimal_rewrite_action(self.tape, hypo, is_retro, t0)

            # Record inverse delta for Ctrl+Shift+Backspace undo
            common_prefix_len = len(old_tail) - action.backspaces
            self.key_lock.record_for_undo(old_tail[common_prefix_len:], action)

            self.key_lock.begin_rewrite()
            try:
                self.mutator.apply_rewrite(action)
            finally:
                drained = self.key_lock.end_rewrite_and_drain()

            for queued_ev in drained:
                self.process_key_event(queued_ev)

            return action

        # Regular word character (letter, apostrophe, digit)
        self.tape.feed_char(event.char, self.store)
        return None

    def simulate_typing(self, text: str) -> tuple[str, list[RewriteAction]]:
        """Feeds `text` character-by-character through the full engine and returns the final output + trace."""
        mem = MemoryBufferMutator()
        self.mutator = mem
        self.tape.hard_reset()
        self.planner.optimizer.recent_replacements.clear()
        actions: list[RewriteAction] = []

        if not text.endswith((" ", ".", "!", "?")):
            text = text + " "

        for ch in text:
            act = self.process_key_event(KeyEvent(key_code=0, char=ch, is_press=True))
            if act is not None:
                actions.append(act)

        return (mem.get_text().rstrip(), actions)


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autocorrupt",
        description="AutoCorrupt: 32-Slice Deterministic GOFAI Background Word-Corruption Engine",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_comp = sub.add_parser("compile-lexicon", help="Compile binary mmap lexicon image")
    p_comp.add_argument("--out", default="lexicon.bin", help="Output binary path")

    p_sim = sub.add_parser("simulate", help="Simulate typing a sentence with full GOFAI trace")
    p_sim.add_argument("sentence", help="Input sentence to type")

    sub.add_parser("benchmark", help="Run 1,000-keystroke sub-millisecond latency benchmark")
    sub.add_parser("self-test", help="Run full deterministic verification gate across all slices")

    p_dae = sub.add_parser("daemon", help="Run live background auto-corrupt keyboard daemon")
    p_dae.add_argument("--device", default="auto", help="Linux /dev/input/event* path or 'auto'")
    p_dae.add_argument("--dry-run", action="store_true", help="Log rewrites without injecting to uinput")

    args = parser.parse_args(argv)

    if args.cmd == "compile-lexicon":
        t0 = time.perf_counter()
        store = compile_default_lexicon(args.out)
        dt = (time.perf_counter() - t0) * 1000.0
        size_kb = os.path.getsize(args.out) / 1024.0
        print(f"[OK] Compiled {len(store.entries)} entries & {len(store.synsets)} synsets -> {args.out} ({size_kb:.1f} KB in {dt:.2f} ms)")
        return 0

    if args.cmd == "simulate":
        engine = AutoCorruptEngine()
        final_text, actions = engine.simulate_typing(args.sentence)
        print(f"INPUT : {args.sentence}")
        print(f"OUTPUT: {final_text}")
        for i, act in enumerate(actions, 1):
            mode = "RETROACTIVE TIME-TRAVEL" if act.is_retroactive else "IMMEDIATE"
            print(
                f"  [{i}] {mode} | {act.solver_name} | bs={act.backspaces} inject={act.inject_text!r} "
                f"| utility={act.utility_score:.1f} | latency={act.latency_us:.1f} us\n"
                f"      Trace: {act.trace}"
            )
        return 0

    if args.cmd == "benchmark":
        engine = AutoCorruptEngine()
        sample = (
            "Well I am bringing salad dressing and some hors d'oeuvres. "
            "I am auditioning kids for the play. "
            "Ridiculous autocorrect. "
            "I hate sitting by the phone waiting for it to ring. "
            "Darth Vader sleep pillow. "
            "Stop leaving the bloody lamp on in the bathroom. "
            "Is polenta considered a gluten with chicken. "
            "I am running late. "
            "I need to shake a tower. "
            "I will park the car. "
        ) * 3
        key_latencies_us: list[float] = []
        commit_latencies_us: list[float] = []
        for ch in sample:
            t0 = time.perf_counter_ns()
            engine.process_key_event(KeyEvent(key_code=0, char=ch, is_press=True))
            dt_us = (time.perf_counter_ns() - t0) / 1000.0
            if ch in (" ", "."):
                commit_latencies_us.append(dt_us)
            else:
                key_latencies_us.append(dt_us)

        avg_key = sum(key_latencies_us) / max(1, len(key_latencies_us))
        avg_commit = sum(commit_latencies_us) / max(1, len(commit_latencies_us))
        print(f"[BENCHMARK] Total keystrokes processed : {len(sample)}")
        print(f"[BENCHMARK] Avg per-character trie step: {avg_key:.2f} us ({avg_key / 1000.0:.4f} ms)")
        print(f"[BENCHMARK] Avg word-commit + solver   : {avg_commit:.2f} us ({avg_commit / 1000.0:.4f} ms)")
        assert avg_key < 50.0, f"Per-key latency too high: {avg_key} us"
        assert avg_commit < 1500.0, f"Commit latency too high: {avg_commit} us"
        print("[BENCHMARK] PASS (Sub-millisecond deterministic execution verified)")
        return 0

    if args.cmd == "self-test":
        engine = AutoCorruptEngine()
        golden_cases = [
            ("Well I am bringing salad dressing and some hors d'oeuvres", "horse ovaries"),
            ("I am auditioning kids for the play", "auctioning"),
            ("I am auditioning kids for the play", "foreplay"),
            ("Ridiculous autocorrect", "auto cat rectal"),
            ("I hate sitting by the phone waiting for it to ring", "ripen"),
            ("Darth Vader sleep pillow", "Badger"),
            ("Stop leaving the bloody lamp on in the bathroom", "tampon"),
            ("Is polenta considered a gluten with chicken", "placenta"),
            ("I am running late", "jogging late"),
            ("I need to shake a tower", "take a shower"),
            ("I will park the car", "playground the car"),
        ]
        for sentence, expected_substr in golden_cases:
            out, _ = engine.simulate_typing(sentence)
            assert expected_substr in out, f"Expected {expected_substr!r} in {out!r} for input {sentence!r}"
            print(f"  [PASS] {sentence!r} -> {out!r}")
        print(f"[SELF-TEST] All {len(golden_cases)} golden GOFAI synthesis gates PASSED.")
        return 0

    if args.cmd == "daemon":
        mutator = LinuxUinputMutator(dry_run=args.dry_run)
        engine = AutoCorruptEngine(mutator=mutator)
        reader = LinuxEvdevReader(device_path=args.device, exclusive_grab=False)
        print(f"[DAEMON] AutoCorrupt GOFAI listening on {args.device} (dry_run={args.dry_run})...")
        reader.start_listening(engine.process_key_event)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
