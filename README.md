# AutoCorrupt: 32-Slice Deterministic GOFAI Word-Corruption Engine

AutoCorrupt is a **zero-randomness, sub-millisecond deterministic Symbolic AI (GOFAI)** background typing engine that intercepts words as you type and replaces them using phonetic mouth-shape vectors, WordNet sense disambiguation, cross-POS polysemy traps, physical QWERTY key geometry, and **Retrospective ("Time-Travel") SentenceTape Rewriting**.

Unlike simple regex or static dictionary autocorrect scripts, **zero joke pairs are hardcoded**. Every replacement is synthesized at runtime from raw dictionary definitions (glosses), ARPAbet/IPA articulatory phoneme vectors, morphological affix rules, and syntactic dependency trees.

## Key Capabilities

* **Sub-Millisecond Execution (under 0.5 ms):**
  * **Per-keystroke Trie stepping:** `~2.4 microseconds` (`0.0024 ms`) via flat `MmapLexiconStore` character transitions.
  * **Word-commit + all 6 GOFAI solvers + Retroactive Planner:** `~450 microseconds` (`0.45 ms`).
* **Retrospective "Time-Travel" SentenceTape (`SentenceTape` + `RetroBackpropPlanner`):**
  * Remembers the last 64 tokens of your active sentence/paragraph.
  * When a later word arrives (e.g., typing `"car"` after `"I will park the"`, or `"late"` after `"I am running"`, or `"shower"` / `"tower"` after `"shake a"`), the shift-reduce dependency parser propagates the new grammatical and semantic constraints backward, calculates the minimum-backspace screen diff (`Slice 27`), rewinds the cursor, edits the older word, and restores the surviving words in a single atomic burst.
* **Cross-Platform HAL Architecture (`Slices 28–30`):**
  * **Linux (Pop!_OS / X11 & Wayland — Primary Target):** Direct kernel `/dev/input/event*` reader and single-syscall `/dev/uinput` batched virtual keyboard injector with zero external pip dependencies.
  * **Android:** `InputMethodService` / `InputConnection.beginBatchEdit()` + `deleteSurroundingText()` bridge.
  * **Windows:** `SetWindowsHookExW(WH_KEYBOARD_LL)` + atomic `SendInput()` `INPUT[]` array flush.
  * **macOS:** Quartz `CGEventTapCreate` + single-event `CGEventKeyboardSetUnicodeString`.
* **Built-In Safety & Panic Controls:**
  * **`Ctrl + Shift + Backspace` (Panic Undo):** Immediately reverts the last corruption and restores the literal characters you originally typed.
  * **`Ctrl + Alt + P` (Passthrough Pause):** Toggles the engine between active corruption and passive passthrough.
  * **`PendingKeyLock` FIFO Queue:** Locks and queues physical keystrokes typed *during* a 0.3 ms rewrite burst and replays them cleanly afterward so 120+ WPM typing never scrambles.

## 32-Slice Atomic Architecture ("Apple-Slice" Design)

| Layer | Slices | Responsibility |
| :--- | :--- | :--- |
| **Layer 0: Sub-Atomic Primitives** | `slice_00` – `slice_03` | Immutable data contracts, 16-bit articulatory phoneme mouth-shape bitmasks, 2D physical QWERTY key geometry, and case prosody masks (`lower`, `UPPER`, `Title`, `sPoNgEbOb`). |
| **Layer 1: Morphology** | `slice_04` – `slice_06` | Deterministic affix stripper/lemmatizer, CVC-aware conjugator/re-inflector, and compound word splitter. |
| **Layer 2: Perception & Mmap Store** | `slice_07` – `slice_10` | Context-sensitive Grapheme-to-Phoneme (`G2P`) transducer, Maximal-Onset syllable segmenter, binary `MmapLexiconStore`, and `O(1)` SymSpell Levenshtein-1 automaton. |
| **Layer 3: Syntax & SentenceTape** | `slice_11` – `slice_14` | 64-token `SentenceTape` ring buffer, deterministic Viterbi POS tagger, Left-Corner Shift-Reduce dependency parser, and cross-token juncture/oronym scanner. |
| **Layer 4: Semantics & Connotation** | `slice_15` – `slice_18` | WordNet-style synset graph compiler, Lesk word-sense disambiguator, 6-axis connotation analyzer (`Clinical`, `Visceral`, `Formality`, `Concreteness`, `Valence`, `Rarity`), and Domain Collision Tensor. |
| **Layer 5: Synthetic GOFAI Solvers** | `slice_19` – `slice_24` | Antipodal polysemy sense solver, Cross-POS (`Noun <-> Verb`) flipper, Articulatory Viterbi beam solver (malapropisms & multi-word mondegreens with OCP anti-gemination), Minimal-pair/juncture solver, 6-axis clinical/visceral register projector, and Onset-Rime spoonerism solver. |
| **Layer 6: Planner & Time-Travel** | `slice_25` – `slice_27` | Global deterministic utility optimizer with 16-word cooldown ring buffer, `RetroBackpropPlanner` with Clause Density Governor, and minimal-backspace common-prefix diff planner. |
| **Layer 7: Cross-Platform HAL & CLI** | `slice_28` – `slice_31` | Atomic `PendingKeyLock` FIFO queue, cross-platform HAL ports (Linux, Android, Windows, macOS), Linux kernel `evdev`/`uinput` adapter, CLI runner, benchmark, and verification gate. |

## Quick Commands

```bash
# 1. Compile the binary mmap lexicon (lexicon.bin)
python3 -m autocorrupt compile-lexicon --out autocorrupt/lexicon.bin

# 2. Run the 32-slice unit & golden GOFAI synthesis test suite
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 -m autocorrupt self-test

# 3. Run the 1,000-keystroke sub-millisecond benchmark
python3 -m autocorrupt benchmark

# 4. Simulate any sentence with full microsecond GOFAI trace
python3 -m autocorrupt simulate "Well I am bringing salad dressing and some hors d'oeuvres"
python3 -m autocorrupt simulate "I will park the car"

# 5. Start the live background daemon on Pop!_OS / Linux
sudo python3 -m autocorrupt daemon --device auto
```
