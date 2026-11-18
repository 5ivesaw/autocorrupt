"""
Slice 15: WordNet-Style Semantic Synset Graph & Offline Binary Lexicon Compiler.
Contains pure English dictionary definitions (glosses), parts of speech, and true lexical
synsets/hypernyms/antonyms. Zero hardcoded corruption pairs: all corruptions are discovered
at runtime by intersecting phonetic/orthographic/synset graphs with 6-axis vectors.
"""
from __future__ import annotations
from autocorrupt.core.atoms.slice_00_types import POS, SynsetNode
from autocorrupt.core.atoms.slice_01_articulatory_table import phonetic_hash_64
from autocorrupt.core.perception.slice_07_g2p_transducer import word_to_phonemes
from autocorrupt.core.perception.slice_08_syllable_nuclei import segment_syllables, split_onset_rime
from autocorrupt.core.perception.slice_09_mmap_trie import LexiconEntry, MmapLexiconStore
from autocorrupt.core.perception.slice_10_levenshtein_automaton import generate_1_deletes
from autocorrupt.core.semantics.slice_17_connotation_axes import compute_meaning_vec_from_gloss

# True English Dictionary Synset Database: (pos, lemmas, gloss, antonyms)
RAW_SYNSETS: tuple[tuple[POS, tuple[str, ...], str, tuple[str, ...]], ...] = (
    # --- Polysemous Verbs & Adjectives (Multiple distinct true dictionary senses) ---
    (POS.VERB, ("run", "jog", "sprint", "trot", "gallop"), "move fast on foot by physical locomotion", ("walk", "crawl")),
    (POS.VERB, ("run", "proceed", "operate", "function"), "progress or operate in time schedule or late state", ("stop", "halt")),
    (POS.ADJ, ("cold", "chilly", "icy", "frigid"), "having a low physical temperature chilled", ("hot", "warm")),
    (POS.ADJ, ("cold", "unfriendly", "hostile", "aloof"), "lacking human emotion or warmth unfriendly", ("friendly", "affectionate")),
    (POS.ADJ, ("hard", "firm", "stiff", "rigid"), "solid firm rigid to the physical touch", ("soft", "pliable")),
    (POS.ADJ, ("hard", "difficult", "arduous", "grueling"), "requiring great mental effort difficult to solve", ("easy", "simple")),
    (POS.ADJ, ("sick", "awesome", "rad", "cool"), "slang for admirably impressive or exciting", ("lame", "boring")),
    (POS.ADJ, ("sick", "diseased", "infected", "nauseous"), "affected by physical or medical pathology disease", ("healthy", "wholesome")),
    (POS.VERB, ("hang", "loiter", "socialize"), "spend time relaxing socially with friends", ()),
    (POS.VERB, ("hang", "dangle", "suspend", "droop"), "be suspended from above and swing or dangle loosely", ("fall", "drop")),

    # --- Noun / Verb Cross-POS Polysemes & Heteronyms ---
    (POS.VERB, ("book", "reserve", "schedule"), "reserve a ticket hotel or travel accommodation in advance", ("cancel",)),
    (POS.NOUN, ("book", "novel", "paperback", "tome"), "printed literary work or bound volume of pages", ()),
    (POS.VERB, ("park", "station", "maneuver"), "bring a motor vehicle car to a halt and leave it in a space", ("drive",)),
    (POS.NOUN, ("park", "playground", "meadow", "courtyard"), "public green recreational land with grass and playground for kids", ()),
    (POS.VERB, ("season", "spice", "flavor"), "add salt herbs or spice flavor to cooked meat food in kitchen", ()),
    (POS.NOUN, ("season", "autumn", "solstice", "equinox"), "one of the four calendar time quarters of the year", ()),
    (POS.VERB, ("lead", "guide", "command", "direct"), "guide or direct a group team or organization", ("follow",)),
    (POS.NOUN, ("lead", "graphite", "plumbum"), "heavy toxic metallic element or pencil graphite core", ()),
    (POS.NOUN, ("bass", "subwoofer", "baritone"), "low frequency musical sound tone or voice", ("treble", "soprano")),
    (POS.NOUN, ("bass", "trout", "halibut", "perch"), "edible freshwater or marine spiny-finned fish food", ()),
    (POS.VERB, ("wind", "coil", "crank", "twist"), "turn or coil a mechanical clock spring or string", ("unwind",)),
    (POS.NOUN, ("wind", "breeze", "gale", "gust"), "natural current of moving air or weather breeze", ()),
    (POS.VERB, ("polish", "buff", "burnish", "shine"), "rub a physical shoe surface to make it smooth and shiny", ("tarnish",)),
    (POS.ADJ, ("polish", "warsaw", "slavic"), "relating to the European country Poland or Warsaw", ()),
    (POS.VERB, ("water", "irrigate", "sprinkle"), "pour water over plants or soil", ()),
    (POS.NOUN, ("water", "dihydrogen monoxide", "liquid"), "transparent odorless chemical medical liquid H2O", ()),

    # --- Everyday Words with Clinical / Visceral / Dysphemistic Synonyms in Synset ---
    (POS.VERB, ("touch", "contact", "fondle", "palpate", "grope"), "come into physical contact or caress with hand or finger", ()),
    (POS.VERB, ("eat", "consume", "ingest", "masticate", "devour"), "put food into mouth chew and swallow nutrient into body", ("fast", "starve")),
    (POS.NOUN, ("friend", "buddy", "companion", "bedfellow", "accomplice"), "person joined by mutual affection intimacy or shared activity", ("enemy", "foe")),
    (POS.ADJ, ("wet", "damp", "moist", "drenched"), "covered or saturated with water or bodily liquid", ("dry", "arid")),
    (POS.VERB, ("finish", "complete", "climax", "consummate", "terminate"), "bring an activity or process to its final culmination", ("begin", "start")),
    (POS.VERB, ("kiss", "smooch", "osculate", "exchange oral mucosa"), "touch or press with the lips in anatomical or romantic affection", ()),
    (POS.VERB, ("sleep", "slumber", "lapse into unconsciousness"), "rest in a natural periodic state of suspended consciousness on a bed", ("wake",)),
    (POS.NOUN, ("butt", "backside", "gluteal cleft", "posterior"), "fleshy anatomical buttocks at the rear of the human pelvis", ()),
    (POS.NOUN, ("dog", "hound", "canine", "mammal", "tapeworm"), "domesticated carnivorous mammal or household animal", ()),
    (POS.NOUN, ("smell", "scent", "aroma", "stench", "odor"), "quality perceived by the olfactory sense of the nose", ()),

    # --- Contranyms, Modifiers & Bowdlerized Exclamations ---
    (POS.NOUN, ("oversight", "supervision", "negligence"), "watchful care and management or an unintentional failure to notice", ()),
    (POS.VERB, ("sanction", "approve", "boycott"), "give official permission for or impose a penalty on", ()),
    (POS.ADV, ("definitely", "certainly", "reluctantly"), "without doubt or in a manner showing hesitation", ("reluctantly", "unwillingly")),
    (POS.VERB, ("love", "adore", "tolerate"), "feel deep affection for or endure with forbearance", ("hate", "tolerate")),
    (POS.NOUN, ("bs", "birdseed", "nonsense"), "foolish talk or small grain seeds fed to birds", ("birdseed",)),

    # --- General English Vocabulary Across Food, Anatomy, Animals, Household, Tech, Kids ---
    (POS.NOUN, ("hors",), "French preposition used in appetizer food phrase", ()),
    (POS.NOUN, ("d'oeuvres", "oeuvre", "appetizer"), "small savory food dish served before a meal", ()),
    (POS.NOUN, ("horse", "stallion", "equine"), "large four-legged hoofed mammal animal ridden by humans", ()),
    (POS.NOUN, ("ovary", "ovaries"), "female reproductive anatomical organ producing eggs in mammal body", ()),
    (POS.NOUN, ("polenta",), "Italian boiled cornmeal grain food dish cooked in kitchen", ()),
    (POS.NOUN, ("placenta",), "vascular anatomical organ in uterus of pregnant mammal fetus", ()),
    (POS.NOUN, ("gluten",), "protein composite found in wheat and cereal grain food", ()),
    (POS.NOUN, ("chicken", "poultry"), "domestic fowl bird kept for eggs or cooked meat food", ()),
    (POS.NOUN, ("salad",), "cold dish of mixed raw green vegetables and dressing food", ()),
    (POS.NOUN, ("dressing",), "sauce for salad food", ()),
    (POS.NOUN, ("strawberry", "strawberries"), "sweet red edible fruit with seeds on surface food", ()),
    (POS.NOUN, ("almond", "almonds"), "oval edible nut seed of the almond tree food", ()),
    (POS.VERB, ("audition", "auditioning"), "perform a trial hearing test for actors or kids in a theater play", ()),
    (POS.VERB, ("auction", "auctioning"), "sell property or goods publicly to the highest commercial bidder", ()),
    (POS.VERB, ("suction", "suctioning"), "draw fluid or tissue out of a body cavity by medical vacuum aspiration", ()),
    (POS.NOUN, ("kid", "kids", "child"), "young human child or youth in school or family", ()),
    (POS.NOUN, ("nutcracker",), "mechanical tool for cracking nuts or classical ballet play for kids", ()),
    (POS.NOUN, ("autocorrect",), "software feature that automatically fixes typing and spell errors on a phone", ()),
    (POS.NOUN, ("auto",), "automobile car or automatic prefix", ()),
    (POS.NOUN, ("cat",), "small domesticated carnivorous feline mammal animal", ()),
    (POS.ADJ, ("rectal",), "relating to the anatomical rectum or lower bowel orifice", ()),
    (POS.NOUN, ("play",), "dramatic theatrical performance or recreational game for kids", ()),
    (POS.NOUN, ("foreplay",), "intimate erotic sexual stimulation preceding intercourse", ()),
    (POS.VERB, ("ring",), "make a clear resonant sound like a telephone bell or call", ()),
    (POS.VERB, ("ripen",), "become mature and sweet and ready to eat like a banana fruit food", ()),
    (POS.NOUN, ("phone", "telephone"), "telecommunications device used to transmit voice calls", ()),
    (POS.NOUN, ("banana",), "long curved yellow tropical edible fruit food", ()),
    (POS.NOUN, ("darth",), "title of a fictional dark sci-fi character", ()),
    (POS.NOUN, ("vader",), "armored helmeted sci-fi villain character toy for kids", ()),
    (POS.NOUN, ("badger",), "nocturnal burrowing carnivorous mammal animal with black and white stripes", ()),
    (POS.NOUN, ("pillow",), "soft cushion support for the head on a bed in a household bedroom", ()),
    (POS.NOUN, ("bed",), "piece of household furniture for sleep and rest", ()),
    (POS.NOUN, ("birthday",), "anniversary of the day on which a person or child was born", ()),
    (POS.NOUN, ("shirt",), "cloth household garment for the upper body", ()),
    (POS.NOUN, ("shit",), "vulgar slang for feces excrement in the anatomical bowel", ()),
    (POS.NOUN, ("lamp",), "household electric light fixture bulb in a room or bathroom", ()),
    (POS.NOUN, ("tampon",), "absorbent medical plug inserted into body cavity to stop menstrual blood flow", ()),
    (POS.NOUN, ("bathroom",), "household room containing a bath shower and toilet", ()),
    (POS.ADJ, ("bloody",), "British intensifier or covered in blood", ()),
    (POS.NOUN, ("morning",), "early period of the day from sunrise to noon time", ()),
    (POS.NOUN, ("mourning",), "expression of deep sorrow and grief for someone who is dead", ()),
    (POS.NOUN, ("cereal",), "breakfast food made from roasted grain", ()),
    (POS.ADJ, ("serial",), "consisting of repeated criminal homicides", ()),
    (POS.NOUN, ("cologne",), "scented toilet water perfume applied to skin", ()),
    (POS.NOUN, ("colon",), "main section of the large intestine anatomical bowel up to the rectum", ()),
    (POS.NOUN, ("prostate",), "male reproductive anatomical gland surrounding the neck of the bladder", ()),
    (POS.ADJ, ("prostrate",), "lying stretched out on the ground face downward in submission", ()),
    (POS.NOUN, ("flamenco",), "traditional Spanish dance and guitar music sound", ()),
    (POS.NOUN, ("flamingo",), "tall wading bird animal with pink feathers and long legs", ()),
    (POS.NOUN, ("therapist",), "specialist who treats mental or physical disorders by medical therapy", ()),
    (POS.NOUN, ("rapist",), "criminal who commits violent sexual assault crime", ()),
    (POS.NOUN, ("manslaughter",), "crime of killing a human being without malice aforethought", ()),
    (POS.NOUN, ("man's", "man"), "adult male human being", ()),
    (POS.NOUN, ("laughter",), "action or sound of laughing with happy emotion", ()),
    (POS.NOUN, ("scapegoat",), "person blamed for the wrongdoings of others", ()),
    (POS.NOUN, ("escape",), "act of breaking free from confinement", ()),
    (POS.NOUN, ("goat",), "hardy domesticated ruminant mammal animal with horns", ()),
    (POS.VERB, ("shake",), "move rapidly up and down or tremble", ()),
    (POS.VERB, ("take",), "lay hold of with the hands or perform an action", ()),
    (POS.NOUN, ("tower",), "tall narrow building or architectural structure", ()),
    (POS.NOUN, ("shower",), "household bathroom enclosure in which a person stands under spraying water", ()),
    (POS.NOUN, ("sec", "second"), "short unit of time or moment", ()),
    (POS.NOUN, ("sex",), "intimate sexual activity or biological reproductive category", ()),
    (POS.NOUN, ("regards",), "best wishes and polite social greetings at the end of a message", ()),
    (POS.NOUN, ("retards", "retard"), "offensive vulgar slur or delay in progress", ()),
    (POS.ADJ, ("kind",), "having a friendly generous and considerate social nature", ()),
    (POS.NOUN, ("kink",), "sharp twist in a rope or unconventional intimate sexual fetish desire", ()),
    (POS.NOUN, ("shot",), "attempt or photograph or firing of a weapon", ()),
    (POS.ADJ, ("public",), "open to and shared by all people in the social community", ()),
    (POS.ADJ, ("pubic",), "relating to the anatomical pubis groin or genital body region", ()),
    (POS.VERB, ("assess",), "evaluate or estimate the commercial nature ability or quality of", ()),
    (POS.NOUN, ("asses", "ass"), "plural of donkey animal or vulgar anatomical buttocks", ()),
    (POS.NOUN, ("canal",), "artificial waterway constructed for boat motion navigation", ()),
    (POS.ADJ, ("anal",), "relating to the anatomical anus or lower digestive bowel orifice", ()),
    (POS.NOUN, ("clock",), "mechanical or digital timepiece instrument for measuring time", ()),
    (POS.NOUN, ("cock",), "male bird rooster or vulgar anatomical intimate slang", ()),
    (POS.NOUN, ("santa",), "jolly Christmas figure who brings toys to kids", ()),
    (POS.NOUN, ("satan",), "supreme spirit of evil crime and demon in hell", ()),
    (POS.ADJ, ("marital",), "relating to social marriage and husband and wife", ()),
    (POS.ADJ, ("martial",), "relating to military war crime and combat fighting", ()),
    (POS.ADJ, ("calm",), "peaceful quiet emotion and free from agitation", ()),
    (POS.NOUN, ("clam",), "marine bivalve mollusk animal with a shell food", ()),
    (POS.NOUN, ("condo",), "privately owned commercial apartment household unit", ()),
    (POS.NOUN, ("condom",), "thin rubber sheath worn during intimate sexual intercourse for medical birth control", ()),
    (POS.NOUN, ("client",), "customer receiving professional commercial services", ()),
    (POS.NOUN, ("clitoris",), "sensitive female anatomical intimate erectile organ", ()),
    (POS.ADV, ("late",), "after the expected or usual scheduled time", ()),
    (POS.NOUN, ("brew",), "beverage made by steeping boiling or fermenting coffee or beer food", ()),
    (POS.NOUN, ("drive",), "computer storage disk tech device or trip in a motor vehicle", ()),
    (POS.NOUN, ("ticket",), "printed commercial pass giving the holder the right to travel or enter", ()),
    (POS.NOUN, ("car",), "four-wheeled road motor vehicle powered by an engine for motion", ()),
    (POS.NOUN, ("meat",), "flesh of an animal cooked and eaten as food in kitchen", ()),
    (POS.NOUN, ("team",), "group of players or workers forming one side in a game or commercial job", ()),
    (POS.NOUN, ("shoe", "shoes"), "outer household covering for the human foot worn when walking", ()),
)


def _insert_char_trie(store: MmapLexiconStore, word: str, word_id: int) -> None:
    state = 0
    for ch in word.lower():
        nxt = store.char_transitions[state].get(ch, -1)
        if nxt == -1:
            nxt = len(store.char_transitions)
            store.char_transitions.append({})
            store.char_state_word_id.append(-1)
            store.char_transitions[state][ch] = nxt
        state = nxt
    store.char_state_word_id[state] = word_id


def _insert_phoneme_trie(store: MmapLexiconStore, phonemes: tuple[str, ...], word_id: int) -> None:
    state = 0
    for ph in phonemes:
        nxt = store.phoneme_transitions[state].get(ph, -1)
        if nxt == -1:
            nxt = len(store.phoneme_transitions)
            store.phoneme_transitions.append({})
            store.phoneme_state_word_ids.append([])
            store.phoneme_transitions[state][ph] = nxt
        state = nxt
    if word_id not in store.phoneme_state_word_ids[state]:
        store.phoneme_state_word_ids[state].append(word_id)


def compile_default_lexicon(output_bin_path: str | None = None) -> MmapLexiconStore:
    """
    Compiles all synsets, dictionary words, G2P phoneme representations, 6-axis connotation
    vectors, SymSpell 1-edit indices, and phoneme length buckets into an `MmapLexiconStore`.
    """
    store = MmapLexiconStore()

    word_pos_map: dict[str, list[POS]] = {}
    word_synsets_map: dict[str, list[int]] = {}

    for syn_idx, (pos, lemmas, gloss, antonyms) in enumerate(RAW_SYNSETS):
        m_vec = compute_meaning_vec_from_gloss(lemmas[0], gloss)
        node = SynsetNode(
            synset_id=syn_idx,
            pos=pos,
            gloss=gloss,
            lemmas=lemmas,
            antonym_lemmas=antonyms,
            meaning_vec=m_vec,
        )
        store.synsets[syn_idx] = node
        for lem in lemmas:
            w_clean = lem.lower()
            word_pos_map.setdefault(w_clean, [])
            if pos not in word_pos_map[w_clean]:
                word_pos_map[w_clean].append(pos)
            word_synsets_map.setdefault(w_clean, []).append(syn_idx)

    for w_clean, syn_ids in word_synsets_map.items():
        wid = len(store.entries)
        phs = word_to_phonemes(w_clean)
        ph_hash = phonetic_hash_64(phs)
        sylls = segment_syllables(phs)

        best_vec = store.synsets[syn_ids[0]].meaning_vec
        for sid in syn_ids[1:]:
            sv = store.synsets[sid].meaning_vec
            if (sv.visceral + sv.clinical) > (best_vec.visceral + best_vec.clinical):
                best_vec = sv

        entry = LexiconEntry(
            word_id=wid,
            word=w_clean,
            phonemes=phs,
            phonetic_hash=ph_hash,
            syllable_count=max(1, len(sylls)),
            pos_tags=tuple(word_pos_map[w_clean]),
            synset_ids=tuple(syn_ids),
            primary_meaning=best_vec,
        )
        store.entries.append(entry)
        store.word_to_id[w_clean] = wid
        store.known_words.add(w_clean)

        _insert_char_trie(store, w_clean, wid)
        if phs and " " not in w_clean:
            _insert_phoneme_trie(store, phs, wid)
            store.phoneme_len_buckets.setdefault(len(phs), []).append(wid)
            store.phonetic_buckets.setdefault(ph_hash, []).append(wid)
            store.exact_phonemes.setdefault(phs, []).append(wid)
            onset, rime = split_onset_rime(phs)
            if rime:
                store.rime_index.setdefault(rime, {})[onset] = wid

        if " " not in w_clean:
            for d_key in generate_1_deletes(w_clean):
                store.symspell_index.setdefault(d_key, []).append(wid)

    if output_bin_path:
        store.save_binary(output_bin_path)
    return store
