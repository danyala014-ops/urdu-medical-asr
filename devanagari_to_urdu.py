"""Spell Omnilingual ASR output in Urdu script.

Meta's Omnilingual CTC model has no language setting; for Urdu speech it mostly writes Hindi
(Devanagari), because spoken Urdu and Hindi are nearly identical. The recognition itself is good,
so we convert character by character:

  * each Devanagari letter may only become its phonetically valid Urdu letter(s), e.g.
    थ -> تھ,  स -> س / ص / ث,  ज़ -> ز / ذ / ض / ظ,  ह -> ہ / ح
    and where several are valid, the model's own scores at that moment choose between them;
  * signs Urdu leaves unwritten (short vowels ि ु, virama ्) produce nothing;
  * vowel letters get their Urdu word-initial / medial / final forms (ए -> ای / ئ / ئے);
  * a few very common words whose Urdu spelling differs from Hindi are fixed afterwards.

Characters the model already writes in Urdu script are kept as-is.
"""

NUKTA = "़"

# Devanagari consonant (plus optional nukta) -> Urdu candidates. The first candidate is also the
# fallback; several candidates are ranked by the model's scores for their first letter.
CONSONANTS = {
    "क": ["ک", "ق"], "क" + NUKTA: ["ق"],
    "ख": ["کھ", "خ"], "ख" + NUKTA: ["خ"],
    "ग": ["گ", "غ"], "ग" + NUKTA: ["غ"],
    "घ": ["گھ"], "ङ": ["ن"],
    "च": ["چ"], "छ": ["چھ"],
    # Hindi usually writes ज without the nukta for z-sounds, so plain ज may be any of these.
    "ज": ["ج", "ز", "ذ", "ض", "ظ"], "ज" + NUKTA: ["ز", "ذ", "ض", "ظ"],
    "झ": ["جھ"], "ञ": ["ن"],
    "ट": ["ٹ"], "ठ": ["ٹھ"],
    "ड": ["ڈ"], "ड" + NUKTA: ["ڑ"],
    "ढ": ["ڈھ"], "ढ" + NUKTA: ["ڑھ"],
    "ण": ["ن"],
    "त": ["ت", "ط"], "थ": ["تھ"],
    "द": ["د"], "ध": ["دھ"],
    "न": ["ن"], "ऩ": ["ن"],
    "प": ["پ"], "फ": ["پھ", "ف"], "फ" + NUKTA: ["ف"],
    "ब": ["ب"], "भ": ["بھ"],
    "म": ["م"],
    "य": ["ی"], "र": ["ر"], "ऱ": ["ر"], "ल": ["ل"], "ळ": ["ل"], "व": ["و"],
    "श": ["ش"], "ष": ["ش"],
    "स": ["س", "ص", "ث"],
    "ह": ["ہ", "ح"],
}

# Vowel signs (matras) and independent vowels, by position in the word. A list means several
# spellings are valid there and the model's scores choose: Hindi has no letter for ع (عام, بعد,
# علاقہ are written with plain vowels), and a word-final ा is often Urdu ہ (زیادہ, ممکنہ).
#                     word start    middle        word end
VOWELS = {
    "ा": ("ا", ["ا", "ع"], ["ا", "ہ"]),
    "ी": ("ی", "ی", "ی"),
    "ू": ("و", "و", "و"),
    "े": ("ی", "ی", "ے"),
    "ै": ("ی", "ی", "ے"),
    "ो": ("و", "و", "و"),
    "ौ": ("و", "و", "و"),
    "ॉ": ("ا", "ا", "ا"),
    "ृ": ("ر", "ر", "ر"),
    "ं": ("ن", "ن", "ں"),
    "ँ": ("ں", "ں", "ں"),
    "अ": (["ا", "ع"], "ا", "ا"),
    "आ": (["آ", "عا"], "آ", "آ"),  # عام = ع + long a
    "इ": (["ا", "ع"], "ئ", "ئ"),
    "ई": ("ای", "ئی", "ئی"),
    "उ": (["ا", "ع"], "و", "و"),
    "ऊ": ("او", "و", "و"),
    "ए": ("ای", "ئ", "ئے"),
    "ऐ": ("ای", "ئ", "ئے"),
    "ओ": ("او", "و", "و"),
    "औ": ("او", "و", "و"),
    "ऑ": ("آ", "ا", "ا"),
    "ऋ": ("ر", "ر", "ر"),
}

SILENT = {"ि", "ु", "्", "ः", "ऽ", "॑", "॒"}  # left unwritten in Urdu
DIGITS = {d: str(i) for i, d in enumerate("०१२३४५६७८९")}

# Whole-word fixes for frequent words whose Urdu spelling isn't a letter-by-letter conversion.
WORD_FIXES = {
    "ک": "کہ", "ار": "اور", "تاک": "تاکہ", "جبک": "جبکہ", "بلک": "بلکہ",
    "کیونک": "کیونکہ", "چونک": "چونکہ", "حالانک": "حالانکہ", "لے": "لئے",
    "یے": "یہ", "وو": "وہ",  # colloquial Hindi ये / वो
}
# "لے" is also a real word ("take"); only fix it after کے / کی, where it means "for".
FIX_ONLY_AFTER = {"لے": {"کے", "کی"}}


def is_devanagari(ch):
    return "ऀ" <= ch <= "ॿ"


def to_urdu(seq, best_urdu, score):
    """Convert a collapsed CTC token sequence to Urdu script.

    seq:        list of (token, frame) after CTC collapse (no blanks, no repeats)
    best_urdu:  frame -> best Urdu letter at that frame (fallback for unmapped characters)
    score:      (frame, urdu_letter) -> model score, to rank alternative candidates
    """
    # Merge a base consonant with a following nukta (ज + ़ -> ज़).
    merged = []
    for tok, frame in seq:
        if tok == NUKTA and merged and merged[-1][0] + NUKTA in CONSONANTS:
            merged[-1] = (merged[-1][0] + NUKTA, merged[-1][1])
        elif tok != NUKTA:
            merged.append((tok, frame))

    out = []
    for i, (tok, frame) in enumerate(merged):
        # Doubled consonant (च्च in बच्चे): Urdu writes it once (بچے).
        if tok in CONSONANTS and i >= 2 and merged[i - 1][0] == "्" and merged[i - 2][0] == tok:
            continue
        at_start = not out or out[-1].endswith(" ")
        nxt = merged[i + 1][0] if i + 1 < len(merged) else " "
        at_end = nxt == " "
        if tok in CONSONANTS:
            cands = CONSONANTS[tok]
            out.append(max(cands, key=lambda c: score(frame, c[0])) if len(cands) > 1 else cands[0])
        elif tok in VOWELS:
            start, middle, end = VOWELS[tok]
            spelling = start if at_start else end if at_end else middle
            if isinstance(spelling, list):
                spelling = max(spelling, key=lambda c: score(frame, c[0]))
            out.append(spelling)
        elif tok in SILENT:
            continue
        elif tok in DIGITS:
            out.append(DIGITS[tok])
        elif is_devanagari(tok):
            out.append(best_urdu(frame))  # rare signs: let the model pick an Urdu letter
        elif tok == " " or tok.isdigit() or "؀" <= tok <= "ۿ" or tok in "۔،؟":
            out.append(tok)               # already Urdu script / space / digit
        else:
            out.append(best_urdu(frame))  # other scripts (Latin, ...): best Urdu letter
    return fix_words("".join(out))


def fix_words(text):
    words = text.split()
    for i, w in enumerate(words):
        fix = WORD_FIXES.get(w)
        if fix and (w not in FIX_ONLY_AFTER or (i > 0 and words[i - 1] in FIX_ONLY_AFTER[w])):
            words[i] = fix
    return " ".join(words)
