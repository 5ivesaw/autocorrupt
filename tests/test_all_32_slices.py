"""
Comprehensive 32-Slice Verification Suite for AutoCorrupt GOFAI Engine.
Tests every atomic layer (Slices 00–31), retroactive time-travel rewrites,
minimum-backspace screen diffing, panic undo (`Ctrl+Shift+Backspace`),
passthrough pause (`Ctrl+Alt+P`), FIFO pending-key lock replay, and cross-platform HAL ports.
"""
from __future__ import annotations
import os
import tempfile
import time
import unittest

from autocorrupt.adapters.slice_28_pending_key_lock import PendingKeyLock
from autocorrupt.adapters.slice_29_hal_ports import (
    AndroidInputConnectionBridge,
    MemoryBufferMutator,
)
from autocorrupt.adapters.slice_30_linux_evdev_uinput import (
    INPUT_EVENT_SIZE,
    KEY_BACKSPACE,
    KEY_UP,
    LinuxUinputMutator,
)
from autocorrupt.core.atoms.slice_00_types import (
    DomainMask,
    InflectionTag,
    KeyEvent,
    POS,
)
from autocorrupt.core.atoms.slice_01_articulatory_table import (
    consonant_skeleton,
    phoneme_distance,
    sequence_articulatory_distance,
)
from autocorrupt.core.atoms.slice_02_key_geometry import is_adjacent_key, key_distance
from autocorrupt.core.atoms.slice_03_case_prosody import (
    CASE_LOWER,
    CASE_MIXED,
    CASE_TITLE,
    CASE_UPPER,
    apply_case_profile,
    extract_case_profile,
)
from autocorrupt.core.morphology.slice_04_affix_stripper import strip_affixes
from autocorrupt.core.morphology.slice_05_conjugator import conjugate
from autocorrupt.core.morphology.slice_06_compound_splitter import split_compound
from autocorrupt.core.perception.slice_07_g2p_transducer import word_to_phonemes
from autocorrupt.core.perception.slice_08_syllable_nuclei import (
    segment_syllables,
    split_onset_rime,
)
from autocorrupt.core.perception.slice_09_mmap_trie import MmapLexiconStore
from autocorrupt.core.perception.slice_10_levenshtein_automaton import (
    find_orthographic_neighbors,
)
from autocorrupt.core.semantics.slice_15_wordnet_graph import compile_default_lexicon
from autocorrupt.core.semantics.slice_16_lesk_sense_detector import detect_intended_sense
from autocorrupt.core.semantics.slice_17_connotation_axes import (
    compute_meaning_vec_from_gloss,
)
from autocorrupt.core.semantics.slice_18_domain_collision import compute_domain_collision
from autocorrupt.core.syntax.slice_11_sentence_tape import SentenceTape
from autocorrupt.core.syntax.slice_12_viterbi_pos import update_pos_tags
from autocorrupt.core.syntax.slice_13_shift_reduce_parser import (
    get_retro_linked_slots,
    parse_dependencies,
)
from autocorrupt.daemon.slice_31_cli_and_verifier import AutoCorruptEngine


class TestAutoCorrupt32Slices(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = compile_default_lexicon()

    def test_slice_01_articulatory_features_and_skeleton(self) -> None:
        # Voiced/voiceless bilabial stops P and B are closer than P and K
        self.assertLess(phoneme_distance("P", "B"), phoneme_distance("P", "K"))
        # Rhotic equivalence ER <-> R
        self.assertAlmostEqual(phoneme_distance("ER", "R"), 0.15)
        # Nasal assimilation N <-> NG
        self.assertAlmostEqual(phoneme_distance("N", "NG"), 0.25)
        # French d'oeuvres and ovaries share exact consonant skeleton 'ovrs'
        self.assertEqual(consonant_skeleton("d'oeuvres"), "ovrs")
        self.assertEqual(consonant_skeleton("ovaries"), "ovrs")

    def test_slice_02_key_geometry(self) -> None:
        self.assertTrue(is_adjacent_key("c", "x"))
        self.assertTrue(is_adjacent_key("g", "t"))
        self.assertFalse(is_adjacent_key("q", "p"))
        self.assertLess(key_distance("d", "k"), 10.0)

    def test_slice_03_case_prosody(self) -> None:
        self.assertEqual(extract_case_profile("hello"), CASE_LOWER)
        self.assertEqual(extract_case_profile("FORPLAY"), CASE_UPPER)
        self.assertEqual(extract_case_profile("Vader"), CASE_TITLE)
        self.assertEqual(extract_case_profile("sPoNgE"), CASE_MIXED)
        self.assertEqual(apply_case_profile("badger", CASE_TITLE), "Badger")
        self.assertEqual(apply_case_profile("foreplay", CASE_UPPER), "FOREPLAY")
        self.assertEqual(apply_case_profile("badger", CASE_MIXED, "sPoNgE"), "bAdGeR")

    def test_slice_04_and_05_morphology_roundtrip(self) -> None:
        _, lemma, tag = strip_affixes("auditioning", self.store.known_words)
        self.assertEqual(lemma, "audition")
        self.assertEqual(tag, InflectionTag.GERUND_ING)
        self.assertEqual(conjugate("auction", tag), "auctioning")
        self.assertEqual(conjugate("jog", InflectionTag.GERUND_ING), "jogging")
        self.assertEqual(conjugate("ovary", InflectionTag.PLURAL_S), "ovaries")

    def test_slice_06_compound_splitter(self) -> None:
        res = split_compound("bathroom", {"bath", "room", "bed"})
        self.assertEqual(res, ("bath", "room"))

    def test_slice_07_and_08_g2p_and_syllables(self) -> None:
        phs = word_to_phonemes("autocorrect")
        self.assertEqual(phs[:3], ("AO", "T", "OW"))
        sylls = segment_syllables(phs)
        self.assertGreaterEqual(len(sylls), 3)
        onset, rime = split_onset_rime(word_to_phonemes("shake"))
        self.assertEqual(onset, ("SH",))
        self.assertEqual(rime, ("EY", "K"))

    def test_slice_09_and_10_mmap_binary_and_symspell(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = os.path.join(tmpdir, "lexicon.bin")
            self.store.save_binary(bin_path)
            loaded = MmapLexiconStore.load_binary(bin_path)
            self.assertEqual(len(loaded.entries), len(self.store.entries))
            nbs = find_orthographic_neighbors("lampon", loaded)
            words = {nb.word for nb in nbs}
            self.assertIn("tampon", words)

    def test_slice_11_to_18_tape_viterbi_lesk_and_collision(self) -> None:
        tape = SentenceTape()
        for w in ("I", "will", "park", "the", "car"):
            for ch in w:
                tape.feed_char(ch, self.store)
            tape.commit_active_word(" ", self.store)
        update_pos_tags(tape, self.store)
        # "park" after modal "will" and before "the car" must be tagged as VERB
        self.assertEqual(tape.tokens[2].pos, POS.VERB)
        arcs = parse_dependencies(tape)
        linked = get_retro_linked_slots(tape, 4, arcs)
        self.assertIn(2, linked)

        # Lesk disambiguation on "running late"
        tape2 = SentenceTape()
        for w in ("I", "am", "running", "late"):
            for ch in w:
                tape2.feed_char(ch, self.store)
            tape2.commit_active_word(" ", self.store)
        update_pos_tags(tape2, self.store)
        sid = detect_intended_sense(tape2, 2, self.store)
        self.assertIn("time", self.store.synsets[sid].gloss)

        # Connotation & Domain collision
        v_food = compute_meaning_vec_from_gloss("polenta", "cornmeal grain dish cooked in kitchen food")
        v_body = compute_meaning_vec_from_gloss("placenta", "vascular anatomical organ in uterus of mammal fetus")
        coll = compute_domain_collision(v_food, v_body, DomainMask.FOOD)
        self.assertGreater(coll, 70.0)

    def test_slice_19_to_27_all_golden_gofai_corruptions(self) -> None:
        engine = AutoCorruptEngine(store=self.store)
        cases = [
            ("Well I am bringing salad dressing and some hors d'oeuvres", "Well I am bringing salad dressing and some horse ovaries"),
            ("I am auditioning kids for the play", "I am auctioning kids foreplay"),
            ("Ridiculous autocorrect", "Ridiculous auto cat rectal"),
            ("I hate sitting by the phone waiting for it to ring", "I hate sitting by the phone waiting for it to ripen"),
            ("Darth Vader sleep pillow", "Darth Badger sleep pillow"),
            ("Stop leaving the bloody lamp on in the bathroom", "Stop leaving the bloody tampon in the bathroom"),
            ("Is polenta considered a gluten with chicken", "Is placenta considered a gluten with chicken"),
            ("I am running late", "I am jogging late"),
            ("I need to shake a tower", "I need to take a shower"),
            ("I will park the car", "I will playground the car"),
        ]
        for raw_in, expected_out in cases:
            actual_out, actions = engine.simulate_typing(raw_in)
            self.assertEqual(actual_out, expected_out)
            self.assertGreaterEqual(len(actions), 1)

    def test_slice_26_retroactive_time_travel_flag(self) -> None:
        engine = AutoCorruptEngine(store=self.store)
        # In "I will park the car", "park" is held until "car" arrives, then edited retroactively!
        out, actions = engine.simulate_typing("I will park the car")
        self.assertEqual(out, "I will playground the car")
        self.assertTrue(any(a.is_retroactive for a in actions))

    def test_slice_28_panic_undo_and_pause_hotkeys(self) -> None:
        mem = MemoryBufferMutator()
        engine = AutoCorruptEngine(store=self.store, mutator=mem)
        for ch in "Is polenta ":
            engine.process_key_event(KeyEvent(key_code=0, char=ch, is_press=True))
        self.assertEqual(mem.get_text(), "Is placenta ")

        # Trigger Ctrl + Shift + Backspace Panic Undo -> restores "Is polenta "!
        undo_act = engine.process_key_event(
            KeyEvent(key_code=KEY_BACKSPACE, char="", is_press=True, ctrl=True, shift=True)
        )
        self.assertIsNotNone(undo_act)
        self.assertEqual(mem.get_text(), "Is polenta ")

        # Trigger Ctrl + Alt + P passthrough pause toggle
        engine.process_key_event(KeyEvent(key_code=25, char="p", is_press=True, ctrl=True, alt=True))
        self.assertTrue(engine.key_lock.passthrough_paused)
        for ch in "Is polenta ":
            engine.process_key_event(KeyEvent(key_code=0, char=ch, is_press=True))
        # Unpaused toggle back
        engine.process_key_event(KeyEvent(key_code=25, char="p", is_press=True, ctrl=True, alt=True))
        self.assertFalse(engine.key_lock.passthrough_paused)

    def test_slice_28_pending_key_queue_during_rewrite(self) -> None:
        lock = PendingKeyLock()
        lock.begin_rewrite()
        ev = KeyEvent(key_code=30, char="a", is_press=True)
        self.assertTrue(lock.try_enqueue_if_rewriting(ev))
        drained = lock.end_rewrite_and_drain()
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0].char, "a")

    def test_slice_29_and_30_hal_and_linux_uinput_packets(self) -> None:
        # Verify Linux uinput binary packet builder produces exact struct input_event multiples
        pkt = LinuxUinputMutator.build_event_packets(backspaces=4, inject_text="Badger ")
        self.assertEqual(len(pkt) % INPUT_EVENT_SIZE, 0)
        self.assertGreater(len(pkt), INPUT_EVENT_SIZE * 10)

        # Verify Android InputConnection batch bridge
        android = AndroidInputConnectionBridge()
        engine = AutoCorruptEngine(store=self.store, mutator=android)
        for ch in "Is polenta ":
            engine.process_key_event(KeyEvent(key_code=0, char=ch, is_press=True))
        self.assertIsNotNone(android.last_batch_payload)
        self.assertEqual(android.last_batch_payload["deleteSurroundingText"]["beforeLength"], 7)
        self.assertEqual(android.last_batch_payload["commitText"]["text"], "lacenta ")

    def test_navigation_and_physical_backspace_handling(self) -> None:
        mem = MemoryBufferMutator()
        engine = AutoCorruptEngine(store=self.store, mutator=mem)
        for ch in "polen":
            engine.process_key_event(KeyEvent(key_code=0, char=ch, is_press=True))
        # User hits physical Backspace twice ("pol") then types "enta "
        engine.process_key_event(KeyEvent(key_code=KEY_BACKSPACE, char="", is_press=True))
        engine.process_key_event(KeyEvent(key_code=KEY_BACKSPACE, char="", is_press=True))
        for ch in "enta ":
            engine.process_key_event(KeyEvent(key_code=0, char=ch, is_press=True))
        self.assertEqual(mem.get_text(), "placenta ")

        # Cursor navigation key resets the tape cleanly
        engine.process_key_event(KeyEvent(key_code=KEY_UP, char="", is_press=True))
        self.assertEqual(len(engine.tape.tokens), 0)


if __name__ == "__main__":
    unittest.main()
